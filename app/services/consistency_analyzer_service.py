# app/services/consistency_analyzer_service.py
from typing import List, Dict, Tuple, Optional
from app.schemas.policy_matching_schema import PolicyPosition, LogicalTension, TensionSeverity
from app.core.matching_config import matching_config
from app.utils.logging_util import setup_logger

class ConsistencyAnalyzerService:
    """Service for analyzing logical consistency in policy positions"""

    def __init__(self):
        self.logger = setup_logger(__name__)
        self._initialize_tension_rules()

    def _initialize_tension_rules(self):
        """
        Define rules for detecting logical tensions between policy dimensions
        """
        self.tension_rules = [
            {
                "dimension_1": "education_funding",
                "dimension_2": "tax_policy",
                "tension_type": "resource_conflict",
                "description": "High education spending vs low tax support",
                "detector": self._detect_funding_tax_tension,
                "severity": TensionSeverity.HIGH
            },
            {
                "dimension_1": "student_support",
                "dimension_2": "school_safety",
                "tension_type": "approach_conflict",
                "description": "Mental health focus vs security focus",
                "detector": self._detect_support_safety_tension,
                "severity": TensionSeverity.MEDIUM
            },
            {
                "dimension_1": "government_role",
                "dimension_2": "local_autonomy",
                "tension_type": "authority_conflict",
                "description": "Central control vs local decision-making",
                "detector": self._detect_authority_tension,
                "severity": TensionSeverity.MEDIUM
            },
            {
                "dimension_1": "education_access",
                "dimension_2": "education_funding",
                "tension_type": "implementation_conflict",
                "description": "Expanding access without funding support",
                "detector": self._detect_access_funding_tension,
                "severity": TensionSeverity.MEDIUM
            }
        ]

    def analyze_position_consistency(self, policy_positions: List[PolicyPosition]) -> Tuple[List[LogicalTension], float]:
        """
        Analyze logical consistency across all policy positions
        """

        if not matching_config.enable_consistency_analysis:
            return [], 1.0

        self.logger.debug(f"Analyzing consistency for {len(policy_positions)} policy positions")

        # Create position lookup for easy access
        position_lookup = {pos.dimension_id: pos for pos in policy_positions}

        detected_tensions = []

        # Check each tension rule
        for rule in self.tension_rules:
            dim_1 = rule["dimension_1"]
            dim_2 = rule["dimension_2"]

            # Check if we have positions for both dimensions
            if dim_1 in position_lookup and dim_2 in position_lookup:
                pos_1 = position_lookup[dim_1]
                pos_2 = position_lookup[dim_2]

                # Apply the detector function
                tension = rule["detector"](pos_1, pos_2, rule)
                if tension:
                    detected_tensions.append(tension)

        # Calculate overall consistency score
        consistency_score = self._calculate_consistency_score(detected_tensions)

        self.logger.debug(f"Detected {len(detected_tensions)} tensions, consistency score: {consistency_score:.3f}")

        return detected_tensions, consistency_score

    def _detect_funding_tax_tension(self, education_pos: PolicyPosition, tax_pos: PolicyPosition, rule: Dict) -> Optional[LogicalTension]:
        """
        Detect tension between education funding support and tax policy
        """

        # High education support (>70) but low tax support (<30) = tension
        if education_pos.position_score > 70 and tax_pos.position_score < 30:
            impact_score = (education_pos.position_score - tax_pos.position_score) / 100

            return LogicalTension(
                dimension_1_id=education_pos.dimension_id,
                dimension_2_id=tax_pos.dimension_id,
                tension_type=rule["tension_type"],
                severity=rule["severity"],
                impact_score=impact_score,
                explanation=f"Strong support for education funding ({education_pos.position_score:.0f}/100) but weak support for taxes ({tax_pos.position_score:.0f}/100) creates implementation challenges."
            )

        return None

    def _detect_support_safety_tension(self, support_pos: PolicyPosition, safety_pos: PolicyPosition, rule: Dict) -> Optional[LogicalTension]:
        """
        Detect tension between student support approach and safety approach
        """

        # Strong support services (>70) but also strong security measures (>70) might indicate tension
        # This is a nuanced tension - some see them as complementary, others as conflicting
        if support_pos.position_score > 75 and safety_pos.position_score > 75:
            # Check intensity - if both are very strong, might indicate internal tension
            combined_intensity = support_pos.intensity_multiplier + safety_pos.intensity_multiplier

            if combined_intensity > 3.0:  # Both are strongly held
                impact_score = 0.3  # Moderate impact

                return LogicalTension(
                    dimension_1_id=support_pos.dimension_id,
                    dimension_2_id=safety_pos.dimension_id,
                    tension_type=rule["tension_type"],
                    severity=TensionSeverity.LOW,  # Override to low since this is philosophical
                    impact_score=impact_score,
                    explanation=f"Very strong support for both therapeutic approaches ({support_pos.position_score:.0f}/100) and security approaches ({safety_pos.position_score:.0f}/100) may indicate competing philosophies."
                )

        return None

    def _detect_authority_tension(self, govt_pos: PolicyPosition, local_pos: PolicyPosition, rule: Dict) -> Optional[LogicalTension]:
        """
        Detect tension between government role and local autonomy
        """

        # High government intervention (>70) but also high local autonomy (>70) = tension
        if govt_pos.position_score > 70 and local_pos.position_score > 70:
            impact_score = min(govt_pos.position_score, local_pos.position_score) / 100 * 0.8

            return LogicalTension(
                dimension_1_id=govt_pos.dimension_id,
                dimension_2_id=local_pos.dimension_id,
                tension_type=rule["tension_type"],
                severity=rule["severity"],
                impact_score=impact_score,
                explanation=f"Strong support for government involvement ({govt_pos.position_score:.0f}/100) conflicts with strong support for local autonomy ({local_pos.position_score:.0f}/100)."
            )

        return None

    def _detect_access_funding_tension(self, access_pos: PolicyPosition, funding_pos: PolicyPosition, rule: Dict) -> Optional[LogicalTension]:
        """
        Detect tension between expanding access and funding support
        """

        # High access expansion (>70) but low funding support (<40) = tension
        if access_pos.position_score > 70 and funding_pos.position_score < 40:
            impact_score = (access_pos.position_score - funding_pos.position_score) / 100

            return LogicalTension(
                dimension_1_id=access_pos.dimension_id,
                dimension_2_id=funding_pos.dimension_id,
                tension_type=rule["tension_type"],
                severity=rule["severity"],
                impact_score=impact_score,
                explanation=f"Strong support for expanding access ({access_pos.position_score:.0f}/100) but weak support for funding ({funding_pos.position_score:.0f}/100) creates implementation gap."
            )

        return None

    def _calculate_consistency_score(self, tensions: List[LogicalTension]) -> float:
        """
        Calculate overall consistency score from detected tensions
        """

        if not tensions:
            return 1.0

        # Weight tensions by severity and impact
        total_penalty = 0.0
        severity_weights = {
            TensionSeverity.LOW: 0.1,
            TensionSeverity.MEDIUM: 0.3,
            TensionSeverity.HIGH: 0.5
        }

        for tension in tensions:
            severity_weight = severity_weights[tension.severity]
            penalty = tension.impact_score * severity_weight
            total_penalty += penalty

        # Consistency score is 1.0 minus total penalty, bounded at 0.0
        consistency_score = max(0.0, 1.0 - total_penalty)

        return consistency_score

    def apply_consistency_penalty(self, base_match_score: float, consistency_score: float) -> float:
        """
        Apply consistency penalty to a match score
        """

        if not matching_config.enable_consistency_analysis:
            return base_match_score

        # Calculate penalty
        penalty_factor = 1.0 - (matching_config.consistency_penalty_rate * (1.0 - consistency_score))

        # Apply penalty
        adjusted_score = base_match_score * penalty_factor

        self.logger.debug(f"Applied consistency penalty: {base_match_score:.3f} -> {adjusted_score:.3f} (consistency: {consistency_score:.3f})")

        return adjusted_score

# Create service instance
consistency_analyzer_service = ConsistencyAnalyzerService()