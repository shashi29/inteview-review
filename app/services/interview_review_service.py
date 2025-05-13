import json
import os
import logging
import zlib
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
# from app.config import Settings
from typing import List
from diskcache import Cache, Disk, UNKNOWN
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

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

class JSONDisk(Disk):
    def __init__(self, directory, compress_level=1, **kwargs):
        self.compress_level = compress_level
        super().__init__(directory, **kwargs)

    def put(self, key):
        json_bytes = json.dumps(key).encode('utf-8')
        data = zlib.compress(json_bytes, self.compress_level)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        return json.loads(zlib.decompress(data).decode('utf-8'))

    def store(self, value, read, key=UNKNOWN):
        if not read:
            json_bytes = json.dumps(value).encode('utf-8')
            value = zlib.compress(json_bytes, self.compress_level)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = json.loads(zlib.decompress(data).decode('utf-8'))
        return data

class InterviewReviewService:
    def __init__(self):
        self.model = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                temperature=os.getenv('OPENAI_TEMPERATURE'),
                                model_name=os.getenv('OPENAI_MODEL'),
                                top_p=os.getenv('OPENAI_TOP_P'),
                                model_kwargs={ "response_format": { "type": "json_object" } })
        
        self.parser = JsonOutputParser(pydantic_object=InterviewReview)
        
        self.prompt = PromptTemplate(
            template="""You are an expert interview evaluator. Analyze the provided interview transcript comprehensively based on the candidate's response to the specific interview question for the given job position.

            {format_instructions}

            Candidate Name: {candidate_name}
            Job Profile: {job_profile}
            Interview Question: {interview_question}
            Transcript: {interview_transcription}

            ## Analysis Framework:

            ### 1. Question Relevance & Answer Completeness
            - Evaluate how directly the candidate addressed the specific question asked
            - Assess whether they understood what was being asked
            - Note if they provided concrete examples or remained too abstract
            - Check if they answered all parts of multi-part questions

            ### 2. Content Analysis
            Provide detailed analysis including:
            - **depth**: How thoroughly did they explore the topic?
            - **accuracy**: Were their statements factually correct?
            - **specificity**: Did they provide concrete examples or details?
            - **structure**: Was their response well-organized and logical?

            ### 3. Communication Skills
            Evaluate and score:
            - **clarity**: How clearly did they express their thoughts?
            - **articulation**: Were they easy to understand?
            - **listening**: Did they demonstrate good listening skills?
            - **responsiveness**: How well did they respond to follow-up questions?

            ### 4. Critical Thinking
            Assess:
            - **problem_solving**: How did they approach complex issues?
            - **analysis**: Did they break down problems systematically?
            - **creativity**: Did they show innovative thinking?
            - **reasoning**: Was their logic sound and well-supported?

            ### 5. Professional Demeanor
            Evaluate:
            - **confidence**: Were they appropriately confident without being arrogant?
            - **enthusiasm**: Did they show genuine interest in the role?
            - **composure**: How did they handle pressure or challenging questions?
            - **professionalism**: Did they maintain appropriate professional boundaries?

            ### 6. Technical Proficiency
            For the given job profile, assess:
            - **knowledge_depth**: How deep is their understanding of relevant concepts?
            - **practical_experience**: Did they demonstrate hands-on experience?
            - **current_awareness**: Are they up-to-date with industry trends?
            - **application**: Can they apply their knowledge to real scenarios?

            ### 7. Soft Skills
            Evaluate:
            - **teamwork**: How do they approach collaboration?
            - **leadership**: Do they show leadership potential?
            - **adaptability**: How flexible are they with changing requirements?
            - **emotional_intelligence**: Do they show self-awareness and empathy?

            ### 8. Cultural Fit
            Assess:
            - **values_alignment**: Do their values align with typical organizational values?
            - **work_style**: How does their work style match the role requirements?
            - **motivation**: What drives them and does it align with the position?
            - **growth_mindset**: Do they show desire for continuous learning?

            ## Scoring Guidelines (1-5 scale):
            - **5 - Exceptional**: Far exceeds expectations, demonstrates mastery
            - **4 - Strong**: Exceeds expectations in most areas
            - **3 - Satisfactory**: Meets expectations, solid performance
            - **2 - Below Average**: Falls short of expectations in several areas
            - **1 - Poor**: Significantly below expectations, major concerns

            ## Required Output Structure:

            ### Scoring Dictionary must include:
            - question_relevance_score
            - answer_completeness_score
            - communication_score
            - critical_thinking_score
            - professional_demeanor_score
            - technical_proficiency_score (rate based on job requirements)
            - soft_skills_score
            - cultural_fit_score
            - overall_score (calculated as weighted average)

            ### Areas for Improvement:
            Provide 3-5 specific, actionable recommendations for the candidate.

            ### Summary:
            Include:
            - **strengths**: Top 2-3 strengths demonstrated
            - **weaknesses**: Top 2-3 areas needing improvement
            - **key_insights**: Important observations about the candidate
            - **interview_highlights**: Notable moments or responses

            ### Recommendation:
            Choose one: "Strongly Recommend", "Recommend", "Consider with Reservations", "Do Not Recommend"
            Provide a brief justification (2-3 sentences) for your recommendation.

            ## Important Notes:
            - Be objective and evidence-based in your evaluation
            - Consider the specific requirements of the job profile
            - Look for both verbal and implied communication
            - Note any red flags or concerning patterns
            - Be constructive in feedback while being honest about weaknesses

            Return output in JSON format only, strictly following the provided schema.
            """,
            input_variables=["candidate_name", "job_profile", "interview_question", "interview_transcription"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.model | self.parser
        
        # Initialize cache with JSONDisk
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '29112024_interview_cache')
        self.cache = Cache(directory=cache_dir, disk=JSONDisk, disk_compress_level=6)

    def generate_review(self, job_profile: str, candidate_name: str, interview_question: str, interview_transcription: str, request_id: str) -> InterviewReview:
        # Create a unique key for caching
        # cache_key = self._create_cache_key(job_profile, candidate_name, interview_question, interview_transcription)
        
        # # Try to get the result from cache
        # cached_result = self.cache.get(cache_key)
        # if cached_result:
        #     logger.info("Retrieved result from cache")
        #     return InterviewReview(**cached_result)
        
        # If not in cache, generate the review
        logger.info("Generating new review using API")
        review = self.chain.invoke({
            "candidate_name": candidate_name,
            "job_profile": job_profile,
            "interview_question": interview_question,
            "interview_transcription": interview_transcription
        })
        review['request_id'] = request_id
        
        #Cache the result
        #self.cache.set(cache_key, review)
        
        return review

    def _create_cache_key(self, job_profile: str, candidate_name: str, interview_question: str, interview_transcription: str) -> str:
        # Create a unique key based on input parameters
        key_data = f"{job_profile}|{candidate_name}|{interview_question}|{interview_transcription}"
        return zlib.adler32(key_data.encode())