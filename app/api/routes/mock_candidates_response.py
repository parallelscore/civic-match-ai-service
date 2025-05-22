from fastapi import status

from app.utils.logging_util import setup_logger
from app.api.routes.base_router import RouterManager


class MockCandidatesResponseRouter:
    """Router for mock API endpoints that simulate the external candidate API."""

    def __init__(self):
        self.router_manager = RouterManager()
        self.logger = setup_logger(__name__)

        # Register the mock API endpoint
        self.router_manager.add_route(
            path="/candidates/recommendation/{election_id}",
            handler_method=self.get_mock_candidates_response,
            methods=["GET"],
            tags=["Mock API"],
            status_code=status.HTTP_200_OK
        )

    async def get_mock_candidates_response(self, election_id: str):
        """
        Get mock candidates for a specific election.
        This endpoint has the same structure as the expected external API.
        """
        self.logger.info(f"Fetching mock candidates for election {election_id}")

        # Return mock data regardless of the election_id
        return [
            {
                "candidate_id": "c001",
                "name": "Jane Smith",
                "election_id": election_id,
                "responses": [
                    {
                        "id": "r001",
                        "question": "Should your neighborhood students have access to a language immersion middle "
                                    "school within a 30-minute commute?",
                        "answer": "Strongly Agree",
                        "comment": "Language immersion programs are crucial for our students' future success in a "
                                   "global economy.",
                        "election_id": election_id
                    },
                    {
                        "id": "r002",
                        "question": "Which educational programs should receive increased funding? "
                                    "(Select all that apply)",
                        "answer": ["STEM initiatives", "Arts and music", "Special education"],
                        "comment": "We need balanced funding across multiple educational areas.",
                        "election_id": election_id
                    },
                    {
                        "id": "r003",
                        "question": "Do you think it's essential for the your neighborhood council member to prioritize"
                                    "mental health resources for students?",
                        "answer": "Strongly Agree",
                        "comment": "Student mental health must be a top priority for all schools.",
                        "election_id": election_id
                    },
                    {
                        "id": "r004",
                        "question": "Do you believe School Resource Officers (SROs) effectively keep your neighborhood "
                                    "schools safe?",
                        "answer": "Disagree",
                        "comment": "We need more community-based approaches to school safety.",
                        "election_id": election_id
                    },
                    {
                        "id": "r005",
                        "question": "Should your your neighborhood council member actively pass legislation benefiting "
                                    "your neighborhood students and families?",
                        "answer": "Strongly Agree",
                        "comment": "Proactive legislation is essential for improving education.",
                        "election_id": election_id
                    },
                    {
                        "id": "r006",
                        "question": "Has education in your neighborhood improved over the last 20 years?",
                        "answer": "Disagree",
                        "comment": "Despite some progress, we still face significant challenges.",
                        "election_id": election_id
                    },
                    {
                        "id": "r007",
                        "question": "What is your top priority in your neighborhood?",
                        "answer": "Education",
                        "comment": "Quality education is the foundation for community success.",
                        "election_id": election_id
                    },
                    {
                        "id": "r008",
                        "question": "What specific improvements would you like to see in your neighborhood?",
                        "answer": "Better school funding and teacher retention programs",
                        "comment": "We need to invest in both facilities and staff to improve education.",
                        "election_id": election_id
                    }
                ]
            },
            {
                "candidate_id": "c002",
                "name": "Michael Johnson",
                "election_id": election_id,
                "responses": [
                    {
                        "id": "r009",
                        "question": "Should your neighborhood students have access to a language immersion middle "
                                    "school within a 30-minute commute?",
                        "answer": "Disagree",
                        "comment": "We should focus on core academics before expanding to immersion programs.",
                        "election_id": election_id
                    },
                    {
                        "id": "r010",
                        "question": "Which educational programs should receive increased funding? "
                                    "(Select all that apply)",
                        "answer": ["STEM initiatives", "Vocational training"],
                        "comment": "Technical skills are critical for future workforce needs.",
                        "election_id": election_id
                    },
                    {
                        "id": "r011",
                        "question": "Do you think it's essential for the your neighborhood council member to prioritize "
                                    "mental health resources for students?",
                        "answer": "Agree",
                        "comment": "Mental health support is important but must be balanced with other priorities.",
                        "election_id": election_id
                    },
                    {
                        "id": "r012",
                        "question": "Do you believe School Resource Officers (SROs) effectively keep your neighborhood "
                                    "schools safe?",
                        "answer": "Strongly Agree",
                        "comment": "SROs are an essential part of school safety.",
                        "election_id": election_id
                    },
                    {
                        "id": "r013",
                        "question": "Should your your neighborhood council member actively pass legislation benefiting "
                                    "your neighborhood students and families?",
                        "answer": "Agree",
                        "comment": "Legislation is important but should be carefully considered.",
                        "election_id": election_id
                    },
                    {
                        "id": "r014",
                        "question": "Has education in your neighborhood improved over the last 20 years?",
                        "answer": "Agree",
                        "comment": "We've made significant strides but still have work to do.",
                        "election_id": election_id
                    },
                    {
                        "id": "r015",
                        "question": "What is your top priority in your neighborhood?",
                        "answer": "Safety",
                        "comment": "Safe schools are prerequisite for effective learning.",
                        "election_id": election_id
                    },
                    {
                        "id": "r016",
                        "question": "What specific improvements would you like to see in your neighborhood?",
                        "answer": "Increased security and discipline in schools",
                        "comment": "Structure and order create the best environment for learning.",
                        "election_id": election_id
                    }
                ]
            },
            {
                "candidate_id": "c003",
                "name": "Aisha Washington",
                "election_id": election_id,
                "responses": [
                    {
                        "id": "r017",
                        "question": "Should your neighborhood students have access to a language immersion middle "
                                    "school within a 30-minute commute?",
                        "answer": "Strongly Agree",
                        "comment": "Multilingual education provides crucial advantages in today's world.",
                        "election_id": election_id
                    },
                    {
                        "id": "r018",
                        "question": "Which educational programs should receive increased funding? "
                                    "(Select all that apply)",
                        "answer": ["Arts and music", "Special education", "After-school programs"],
                        "comment": "We need a holistic approach that supports the whole child.",
                        "election_id": election_id
                    },
                    {
                        "id": "r019",
                        "question": "Do you think it's essential for the your neighborhood council member to "
                                    "prioritize mental health resources for students?",
                        "answer": "Strongly Agree",
                        "comment": "Mental health is the foundation of academic success and student wellbeing.",
                        "election_id": election_id
                    },
                    {
                        "id": "r020",
                        "question": "Do you believe School Resource Officers (SROs) effectively keep your neighborhood "
                                    "schools safe?",
                        "answer": "Strongly Disagree",
                        "comment": "We need counselors, not officers. Community-based safety approaches work better.",
                        "election_id": election_id
                    },
                    {
                        "id": "r021",
                        "question": "Should your your neighborhood council member actively pass legislation benefiting "
                                    "your neighborhood students and families?",
                        "answer": "Strongly Agree",
                        "comment": "Bold legislative action is needed to address systemic inequities.",
                        "election_id": election_id
                    },
                    {
                        "id": "r022",
                        "question": "Has education in your neighborhood improved over the last 20 years?",
                        "answer": "Disagree",
                        "comment": "Despite investments, achievement gaps persist and need urgent attention.",
                        "election_id": election_id
                    },
                    {
                        "id": "r023",
                        "question": "What is your top priority in your neighborhood?",
                        "answer": "Equity",
                        "comment": "Equal access to quality education must be guaranteed for all students.",
                        "election_id": election_id
                    },
                    {
                        "id": "r024",
                        "question": "What specific improvements would you like to see in your neighborhood?",
                        "answer": "More community input in educational decision-making",
                        "comment": "Parents and residents should have a stronger voice in school governance.",
                        "election_id": election_id
                    }
                ]
            }
        ]
