from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.config import Settings

class InterviewReviewService:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=Settings.OPENAI_API_KEY)
        self.prompt_template = PromptTemplate(
            input_variables=["candidate_name", "job_profile", "interview_transcription"],
            template="""
            Analyze the provided interview transcript for {candidate_name} applying for the position {job_profile}. Follow these steps:

            Transcript Review: {interview_transcription}
            ...
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def generate_review(self, job_profile: str, candidate_name: str, interview_transcription: str) -> str:
        return self.chain.run(
            candidate_name=candidate_name,
            job_profile=job_profile,
            interview_transcription=interview_transcription
        )