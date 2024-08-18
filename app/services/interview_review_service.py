from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from app.config import Settings
from typing import List

class TranscriptAnalysis(BaseModel):
    question_relevance: str = Field(description="Assessment of how well the answer relates to the question")
    answer_completeness: str = Field(description="Evaluation of how thoroughly the answer addresses all aspects of the question")
    content_analysis: dict = Field(description="Analysis of the content of the answer")
    communication_skills: dict = Field(description="Evaluation of the candidate's communication skills")
    critical_thinking: dict = Field(description="Assessment of the candidate's critical thinking abilities")
    professional_demeanor: dict = Field(description="Evaluation of the candidate's professional demeanor")
    technical_proficiency: dict = Field(description="Assessment of the candidate's technical knowledge")
    soft_skills: dict = Field(description="Evaluation of the candidate's soft skills")
    cultural_fit: dict = Field(description="Assessment of the candidate's cultural fit")

class InterviewReview(BaseModel):
    candidate_name: str = Field(description="Name of the candidate")
    job_profile: str = Field(description="Job position the candidate is applying for")
    interview_question: str = Field(description="The interview question asked")
    transcript_analysis: TranscriptAnalysis = Field(description="Detailed analysis of the interview transcript")
    areas_for_improvement: List[str] = Field(description="List of areas where the candidate can improve")
    scoring: dict = Field(description="Numerical scores for various aspects of the interview")
    summary: dict = Field(description="Overall summary of the interview")
    recommendation: str = Field(description="Final recommendation regarding the candidate")

class InterviewReviewService:
    def __init__(self):
        self.model = ChatOpenAI(api_key=Settings.OPENAI_API_KEY, temperature=Settings.OPENAI_TEMPERATURE, model_name=Settings.OPEANI_MODEL)
        self.parser = JsonOutputParser(pydantic_object=InterviewReview)
        
        self.prompt = PromptTemplate(
            template="""Analyze the provided interview question and transcript for the candidate applying for the specified job position. Provide a comprehensive evaluation based on the given information.

            {format_instructions}

            Candidate Name: {candidate_name}
            Job Profile: {job_profile}
            Interview Question: {interview_question}
            Transcript: {interview_transcription}

            Ensure all scores are on a scale of 1-5. The overall_score should be an average of the other scores, rounded to one decimal place. Include an assessment of how well the candidate understood and addressed the specific interview question.
            """,
            input_variables=["candidate_name", "job_profile", "interview_question", "interview_transcription"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.model | self.parser

    def generate_review(self, job_profile: str, candidate_name: str, interview_question: str, interview_transcription: str) -> InterviewReview:
        return self.chain.invoke({
            "candidate_name": candidate_name,
            "job_profile": job_profile,
            "interview_question": interview_question,
            "interview_transcription": interview_transcription
        })