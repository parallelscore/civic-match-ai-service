# app/services/consistency_analyzer_service.py
#
# V2 — Generalised consistency analyzer.
# No hardcoded dimension IDs. Tension rules are derived dynamically from
# the actual dimensions present in each submission, using universal
# conflict patterns that apply to any election domain.

from typing import List, Tuple, Optional
from app.schemas.policy_matching_schema import PolicyPosition, LogicalTension, TensionSeverity
from app.core.matching_config import matching_config
from app.utils.logging_util import setup_logger


class ConsistencyAnalyzerService:
    """
    Detects logical tensions between a person's policy positions.

    V2 design: instead of hardcoding dimension IDs (which only worked for
    school-board elections), we inspect every pair of positions and apply
    three universal tension patterns:

      1. High-spend / low-fund  — wants more of something but opposes funding it
      2. Centralise / decentralise  — wants both central control AND local autonomy
      3. Expand / restrict  — wants to expand and restrict the same thing

    Keywords embedded in dimension names and descriptions are used to
    classify dimensions into these broad semantic buckets on the fly.
    """

    # ── Semantic bucket keyword sets ─────────────────────────────────────
    _SPENDING_KEYWORDS = {"fund", "invest", "spend", "budget", "resource", "program", "service", "support"}
    _REVENUE_KEYWORDS  = {"tax", "revenue", "levy", "fee", "fiscal", "finance"}
    _CENTRAL_KEYWORDS  = {"government", "federal", "state", "central", "national", "legislation", "mandate"}
    _LOCAL_KEYWORDS    = {"local", "community", "autonomy", "district", "school board", "municipal"}
    _EXPAND_KEYWORDS   = {"expand", "increase", "more", "universal", "access", "broaden", "extend"}
    _RESTRICT_KEYWORDS = {"restrict", "limit", "reduce", "less", "cut", "cap", "oppose"}

    def __init__(self):
        self.logger = setup_logger(__name__)

    # ── Public API ────────────────────────────────────────────────────────

    def analyze_position_consistency(
        self, policy_positions: List[PolicyPosition]
    ) -> Tuple[List[LogicalTension], float]:
        """
        Analyze logical consistency across all policy positions.
        Returns (tensions, consistency_score 0-1).
        """
        if not matching_config.enable_consistency_analysis or len(policy_positions) < 2:
            return [], 1.0

        self.logger.debug(f"Analyzing consistency for {len(policy_positions)} positions")

        position_lookup = {pos.dimension_id: pos for pos in policy_positions}
        detected_tensions: List[LogicalTension] = []

        # Check every unique pair of positions
        ids = list(position_lookup.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pos_a = position_lookup[ids[i]]
                pos_b = position_lookup[ids[j]]

                tension = self._evaluate_pair(pos_a, pos_b)
                if tension:
                    detected_tensions.append(tension)

        consistency_score = self._calculate_consistency_score(detected_tensions)
        self.logger.debug(
            f"Detected {len(detected_tensions)} tensions, "
            f"consistency score: {consistency_score:.3f}"
        )
        return detected_tensions, consistency_score

    def apply_consistency_penalty(
        self, base_match_score: float, consistency_score: float
    ) -> float:
        """Apply consistency penalty to a match score."""
        if not matching_config.enable_consistency_analysis:
            return base_match_score

        penalty_factor = 1.0 - (
            matching_config.consistency_penalty_rate * (1.0 - consistency_score)
        )
        adjusted = base_match_score * penalty_factor
        self.logger.debug(
            f"Consistency penalty: {base_match_score:.3f} → {adjusted:.3f} "
            f"(consistency={consistency_score:.3f})"
        )
        return adjusted

    # ── Internal helpers ──────────────────────────────────────────────────

    def _evaluate_pair(
        self, pos_a: PolicyPosition, pos_b: PolicyPosition
    ) -> Optional[LogicalTension]:
        """
        Apply universal tension patterns to a pair of positions.
        Returns a LogicalTension if a conflict is detected, else None.
        """
        # Pattern 1: high spending desire vs low revenue/funding support
        tension = self._check_spend_revenue_conflict(pos_a, pos_b)
        if tension:
            return tension

        # Pattern 2: central control vs local autonomy
        tension = self._check_centralise_decentralise_conflict(pos_a, pos_b)
        if tension:
            return tension

        # Pattern 3: expand vs restrict the same policy area (same keywords)
        tension = self._check_expand_restrict_conflict(pos_a, pos_b)
        if tension:
            return tension

        return None

    @staticmethod
    def _dim_keywords(pos: PolicyPosition) -> set:
        """
        Extract a lowercase keyword set from a position's dimension_id.
        e.g. "education_funding" → {"education", "funding"}
        """
        return set(pos.dimension_id.lower().replace("-", "_").split("_"))

    def _matches_bucket(self, pos: PolicyPosition, bucket: set) -> bool:
        """True if any of the position's dimension keywords appear in the bucket."""
        return bool(self._dim_keywords(pos) & bucket)

    def _check_spend_revenue_conflict(
        self, pos_a: PolicyPosition, pos_b: PolicyPosition
    ) -> Optional[LogicalTension]:
        """
        Tension: strongly wants more spending/services but strongly opposes
        the revenue mechanisms needed to fund them.
        """
        HIGH, LOW = 70, 30

        def _is_spender(p): return self._matches_bucket(p, self._SPENDING_KEYWORDS)
        def _is_revenue(p): return self._matches_bucket(p, self._REVENUE_KEYWORDS)

        spender, revenue = None, None
        if _is_spender(pos_a) and _is_revenue(pos_b):
            spender, revenue = pos_a, pos_b
        elif _is_spender(pos_b) and _is_revenue(pos_a):
            spender, revenue = pos_b, pos_a

        if spender and revenue:
            if spender.position_score > HIGH and revenue.position_score < LOW:
                impact = (spender.position_score - revenue.position_score) / 100
                return LogicalTension(
                    dimension_1_id=spender.dimension_id,
                    dimension_2_id=revenue.dimension_id,
                    tension_type="resource_conflict",
                    severity=TensionSeverity.HIGH,
                    impact_score=round(impact, 3),
                    explanation=(
                        f"Strong support for increased spending/services "
                        f"({spender.position_score:.0f}/100) conflicts with low support "
                        f"for the funding mechanisms needed ({revenue.position_score:.0f}/100)."
                    ),
                )
        return None

    def _check_centralise_decentralise_conflict(
        self, pos_a: PolicyPosition, pos_b: PolicyPosition
    ) -> Optional[LogicalTension]:
        """
        Tension: strongly supports both centralised control AND local autonomy
        on what appears to be the same policy area.
        """
        HIGH = 70

        def _is_central(p): return self._matches_bucket(p, self._CENTRAL_KEYWORDS)
        def _is_local(p):   return self._matches_bucket(p, self._LOCAL_KEYWORDS)

        central, local = None, None
        if _is_central(pos_a) and _is_local(pos_b):
            central, local = pos_a, pos_b
        elif _is_central(pos_b) and _is_local(pos_a):
            central, local = pos_b, pos_a

        if central and local:
            if central.position_score > HIGH and local.position_score > HIGH:
                impact = round(
                    min(central.position_score, local.position_score) / 100 * 0.8, 3
                )
                return LogicalTension(
                    dimension_1_id=central.dimension_id,
                    dimension_2_id=local.dimension_id,
                    tension_type="authority_conflict",
                    severity=TensionSeverity.MEDIUM,
                    impact_score=impact,
                    explanation=(
                        f"Strong support for centralised control "
                        f"({central.position_score:.0f}/100) conflicts with strong "
                        f"support for local autonomy ({local.position_score:.0f}/100)."
                    ),
                )
        return None

    def _check_expand_restrict_conflict(
        self, pos_a: PolicyPosition, pos_b: PolicyPosition
    ) -> Optional[LogicalTension]:
        """
        Tension: one position is clearly about expanding something and the
        other is clearly about restricting it, and they share a common
        policy keyword — suggesting the person wants to expand AND restrict
        the same thing.
        """
        HIGH, LOW = 70, 30

        def _is_expander(p): return self._matches_bucket(p, self._EXPAND_KEYWORDS)
        def _is_restrictor(p): return self._matches_bucket(p, self._RESTRICT_KEYWORDS)

        expander, restrictor = None, None
        if _is_expander(pos_a) and _is_restrictor(pos_b):
            expander, restrictor = pos_a, pos_b
        elif _is_expander(pos_b) and _is_restrictor(pos_a):
            expander, restrictor = pos_b, pos_a

        if expander and restrictor:
            # Only flag if they share at least one non-directional keyword
            # (i.e. they are about the same policy area)
            expander_kws = self._dim_keywords(expander) - self._EXPAND_KEYWORDS - self._RESTRICT_KEYWORDS
            restrictor_kws = self._dim_keywords(restrictor) - self._EXPAND_KEYWORDS - self._RESTRICT_KEYWORDS
            shared = expander_kws & restrictor_kws

            if shared and expander.position_score > HIGH and restrictor.position_score > HIGH:
                impact = round(
                    (expander.position_score + restrictor.position_score) / 200 * 0.6, 3
                )
                return LogicalTension(
                    dimension_1_id=expander.dimension_id,
                    dimension_2_id=restrictor.dimension_id,
                    tension_type="implementation_conflict",
                    severity=TensionSeverity.MEDIUM,
                    impact_score=impact,
                    explanation=(
                        f"Simultaneously wanting to expand "
                        f"({expander.position_score:.0f}/100) and restrict "
                        f"({restrictor.position_score:.0f}/100) related policy areas "
                        f"creates an implementation conflict."
                    ),
                )
        return None

    def _calculate_consistency_score(self, tensions: List[LogicalTension]) -> float:
        """Derive an overall consistency score (1.0 = fully consistent)."""
        if not tensions:
            return 1.0

        severity_weights = matching_config.consistency_severity_weights
        total_penalty = sum(
            tension.impact_score * severity_weights.get(tension.severity.value, 0.3)
            for tension in tensions
        )
        return max(0.0, 1.0 - total_penalty)


# Single global instance
consistency_analyzer_service = ConsistencyAnalyzerService()
