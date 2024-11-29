import json
import os
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List, Dict, Union
from diskcache import Cache, Disk, UNKNOWN
import hashlib
from logging import getLogger
from dotenv import load_dotenv

load_dotenv()
logger = getLogger(__name__)


class InterviewQuestionResult(BaseModel):
    questions: List[str]
    answers: List[str]
    
class InterviewQuestionService:
    def __init__(self, cache_dir: str = ".cache"):
        self.model = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=os.getenv('OPENAI_TEMPERATURE'),
            model_name=os.getenv('OPENAI_MODEL'),
            top_p=os.getenv('OPENAI_TOP_P'),
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        # Interview Question Parser
        self.interview_question_parser = JsonOutputParser(pydantic_object=InterviewQuestionResult)

        # Interview Question Prompt
        self.interview_question_prompt = PromptTemplate(
            template="""
            Generate interview questions and answers based on the provided job details.

            For each candidate interview, create questions that:
            - Align with the job title: {job_title}
            - Match experience level: {experience}
            - Test skills: {skills}
            - Reflect job description: {job_description}

            Number of questions: {number_of_questions}
            Question complexity: {question_level}

            Ensure questions demonstrate:
            - Practical scenario understanding
            - Skill proficiency
            - Relevant job knowledge

            # Steps

            1. **Analyze Inputs**: Carefully examine the `Job_title`, `Experience`, `Skills`, and `Job_description` to get a sense of what the job expectations are.
            2. **Generate Questions**:
                - Use the provided `Job_title`, `Experience`, and `Skills` to create questions that are the most relevant for the specified role.
                - Match the number of generated questions with `Number of questions` provided.
                - Make sure that the complexity aligns with the `Question_level` provided by the user.
            3. **Formulate Answers**:
                - Create clear and concise answers to each of the generated questions.
                - The answers should demonstrate comprehensive understanding and include relevant examples, aligned with the provided skills and job description to show practical knowledge.
            4. **Construct JSON Output**: Organize the questions and answers into the required JSON format.

            Make sure:
            - The number of elements in `"questions"` matches the `Number of questions` provided.
            - Each `"Question"` has a corresponding `"Answer"` that fits its level of complexity.

            # Notes
            - Ensure that the questions include practical scenarios whenever possible, to evaluate both theoretical understanding and hands-on experience.
            - The difficulty of each question should strictly match the desired `Question_level` to ensure consistency.
            - If the user-provided information is vague or contradictory (e.g., "beginner" level but extreme complexity in job requirements), try to find a balance and include a note.

            # Output Format
            {format_instructions}
            """,
            input_variables=[
                "job_title", 
                "experience", 
                "skills", 
                "job_description", 
                "number_of_questions", 
                "question_level"
            ],
            partial_variables={
                "format_instructions": self.interview_question_parser.get_format_instructions()
            }
        )

        # Create chains
        self.interview_question_chain = self.interview_question_prompt | self.model | self.interview_question_parser

        # Initialize cache
        self.cache = Cache(cache_dir)

    def generate_interview_questions(
        self, 
        job_title: str, 
        experience: str, 
        skills: str, 
        job_description: str, 
        number_of_questions: int = 3, 
        question_level: str = "intermediate"
    ) -> InterviewQuestionResult:
        """
        Generate interview questions for a specific job role.

        Args:
            job_title (str): Title of the job
            experience (str): Years of experience required
            skills (str): Key skills for the role
            job_description (str): Detailed job description
            number_of_questions (int, optional): Number of questions to generate. Defaults to 3.
            question_level (str, optional): Complexity of questions. Defaults to "intermediate".

        Returns:
            InterviewQuestionResult: Generated interview questions and answers
        """
        # Generate cache key
        cache_key = self.generate_cache_key(
            job_title + experience + skills + job_description, 
            str(number_of_questions) + question_level
        )

        # Check cache
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logger.info("Cache hit for interview questions")
            return cached_result

        # Generate new interview questions
        logger.info("Generating new interview questions")
        interview_questions = self.interview_question_chain.invoke({
            "job_title": job_title,
            "experience": experience,
            "skills": skills,
            "job_description": job_description,
            "number_of_questions": number_of_questions,
            "question_level": question_level
        })

        # Store in cache
        self.cache.set(cache_key, interview_questions)
        return interview_questions

    def generate_cache_key(self, *args: str) -> str:
        """
        Generate a unique cache key based on input arguments.

        Args:
            *args (str): Variable number of string arguments to generate cache key.

        Returns:
            str: The cache key.
        """
        key_data = ''.join(args)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    
if __name__ == "__main__":
    # Example input data
    job_title = "Software Engineer"
    experience = "3-4 years"
    skills = "Python, Data Structures, Algorithms, OOP"
    job_description = "Develop and manage backend services. Requires strong Python knowledge and understanding of OOP."
    number_of_questions = 5
    question_level = "intermediate"

    # Function call
    print(InterviewQuestionService().generate_interview_questions(
        job_title=job_title,
        experience=experience,
        skills=skills,
        job_description=job_description,
        number_of_questions=number_of_questions,
        question_level=question_level
    ))
