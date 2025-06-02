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
        Includes completion status fields and returns all candidates (eligible and ineligible).
        """
        self.logger.info(f"Fetching mock candidates for election {election_id}")

        # Return mock data with completion status - mix of eligible and ineligible candidates
        return {
            "data": [
                {
                    "candidateId": "c001",  # Using camelCase to match real API
                    "name": "Jane Smith",
                    "electionId": election_id,  # Using camelCase to match real API
                    "hasCompletedProfile": True,
                    "hasCompletedQuestionnaire": True,
                    "responses": [
                        {
                            "id": "r001",
                            "question": "Should your neighborhood students have access to a language immersion middle "
                                        "school within a 30-minute commute?",
                            "answer": "Strongly Agree",
                            "comment": "Language immersion programs are crucial for our students' future success in a "
                                       "global economy.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r002",
                            "question": "Which educational programs should receive increased funding? "
                                        "(Select all that apply)",
                            "answer": ["STEM initiatives", "Arts and music", "Special education"],
                            "comment": "We need balanced funding across multiple educational areas.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r003",
                            "question": "Do you think it's essential for the your neighborhood council member to prioritize"
                                        "mental health resources for students?",
                            "answer": "Strongly Agree",
                            "comment": "Student mental health must be a top priority for all schools.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r004",
                            "question": "Do you believe School Resource Officers (SROs) effectively keep your neighborhood "
                                        "schools safe?",
                            "answer": "Disagree",
                            "comment": "We need more community-based approaches to school safety.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r005",
                            "question": "Should your your neighborhood council member actively pass legislation benefiting "
                                        "your neighborhood students and families?",
                            "answer": "Strongly Agree",
                            "comment": "Proactive legislation is essential for improving education.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r006",
                            "question": "Has education in your neighborhood improved over the last 20 years?",
                            "answer": "Disagree",
                            "comment": "Despite some progress, we still face significant challenges.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r007",
                            "question": "What is your top priority in your neighborhood?",
                            "answer": "Education",
                            "comment": "Quality education is the foundation for community success.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r008",
                            "question": "What specific improvements would you like to see in your neighborhood?",
                            "answer": "Better school funding and teacher retention programs",
                            "comment": "We need to invest in both facilities and staff to improve education.",
                            "electionId": election_id  # Using camelCase
                        }
                    ]
                },
                {
                    "candidateId": "c002",  # Using camelCase
                    "name": "Michael Johnson",
                    "electionId": election_id,  # Using camelCase
                    "hasCompletedProfile": True,
                    "hasCompletedQuestionnaire": True,
                    "responses": [
                        {
                            "id": "r009",
                            "question": "Should your neighborhood students have access to a language immersion middle "
                                        "school within a 30-minute commute?",
                            "answer": "Disagree",
                            "comment": "We should focus on core academics before expanding to immersion programs.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r010",
                            "question": "Which educational programs should receive increased funding? "
                                        "(Select all that apply)",
                            "answer": ["STEM initiatives", "Vocational training"],
                            "comment": "Technical skills are critical for future workforce needs.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r011",
                            "question": "Do you think it's essential for the your neighborhood council member to prioritize "
                                        "mental health resources for students?",
                            "answer": "Agree",
                            "comment": "Mental health support is important but must be balanced with other priorities.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r012",
                            "question": "Do you believe School Resource Officers (SROs) effectively keep your neighborhood "
                                        "schools safe?",
                            "answer": "Strongly Agree",
                            "comment": "SROs are an essential part of school safety.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r013",
                            "question": "Should your your neighborhood council member actively pass legislation benefiting "
                                        "your neighborhood students and families?",
                            "answer": "Agree",
                            "comment": "Legislation is important but should be carefully considered.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r014",
                            "question": "Has education in your neighborhood improved over the last 20 years?",
                            "answer": "Agree",
                            "comment": "We've made significant strides but still have work to do.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r015",
                            "question": "What is your top priority in your neighborhood?",
                            "answer": "Safety",
                            "comment": "Safe schools are prerequisite for effective learning.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r016",
                            "question": "What specific improvements would you like to see in your neighborhood?",
                            "answer": "Increased security and discipline in schools",
                            "comment": "Structure and order create the best environment for learning.",
                            "electionId": election_id  # Using camelCase
                        }
                    ]
                },
                {
                    "candidateId": "c003",  # Using camelCase
                    "name": "Aisha Washington",
                    "electionId": election_id,  # Using camelCase
                    "hasCompletedProfile": True,
                    "hasCompletedQuestionnaire": True,
                    "responses": [
                        {
                            "id": "r017",
                            "question": "Should your neighborhood students have access to a language immersion middle "
                                        "school within a 30-minute commute?",
                            "answer": "Strongly Agree",
                            "comment": "Multilingual education provides crucial advantages in today's world.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r018",
                            "question": "Which educational programs should receive increased funding? "
                                        "(Select all that apply)",
                            "answer": ["Arts and music", "Special education", "After-school programs"],
                            "comment": "We need a holistic approach that supports the whole child.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r019",
                            "question": "Do you think it's essential for the your neighborhood council member to "
                                        "prioritize mental health resources for students?",
                            "answer": "Strongly Agree",
                            "comment": "Mental health is the foundation of academic success and student wellbeing.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r020",
                            "question": "Do you believe School Resource Officers (SROs) effectively keep your neighborhood "
                                        "schools safe?",
                            "answer": "Strongly Disagree",
                            "comment": "We need counselors, not officers. Community-based safety approaches work better.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r021",
                            "question": "Should your your neighborhood council member actively pass legislation benefiting "
                                        "your neighborhood students and families?",
                            "answer": "Strongly Agree",
                            "comment": "Bold legislative action is needed to address systemic inequities.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r022",
                            "question": "Has education in your neighborhood improved over the last 20 years?",
                            "answer": "Disagree",
                            "comment": "Despite investments, achievement gaps persist and need urgent attention.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r023",
                            "question": "What is your top priority in your neighborhood?",
                            "answer": "Equity",
                            "comment": "Equal access to quality education must be guaranteed for all students.",
                            "electionId": election_id  # Using camelCase
                        },
                        {
                            "id": "r024",
                            "question": "What specific improvements would you like to see in your neighborhood?",
                            "answer": "More community input in educational decision-making",
                            "comment": "Parents and residents should have a stronger voice in school governance.",
                            "electionId": election_id  # Using camelCase
                        }
                    ]
                },
                {
                    "candidateId": "c004",  # Using camelCase
                    "name": "Robert Davis",
                    "electionId": election_id,  # Using camelCase
                    "hasCompletedProfile": True,
                    "hasCompletedQuestionnaire": False,  # Incomplete questionnaire
                    "responses": []  # Empty responses array
                },
                {
                    "candidateId": "c005",  # Using camelCase
                    "name": "Sarah Martinez",
                    "electionId": election_id,  # Using camelCase
                    "hasCompletedProfile": False,  # Incomplete profile
                    "hasCompletedQuestionnaire": True,
                    "responses": [
                        {
                            "id": "r025",
                            "question": "Should your neighborhood students have access to a language immersion middle school?",
                            "answer": "Agree",
                            "comment": "This would be beneficial for students.",
                            "electionId": election_id  # Using camelCase
                        }
                    ]
                },
                {
                    "candidateId": "c006",  # Using camelCase
                    "name": "David Thompson",
                    "electionId": election_id,  # Using camelCase
                    "hasCompletedProfile": False,  # Incomplete profile
                    "hasCompletedQuestionnaire": False,  # Incomplete questionnaire
                    "responses": []  # Empty responses
                }
            ],
            "message": "Candidate and responses fetched successfully",
            "statusCode": 200
        }
