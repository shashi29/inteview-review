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
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class TranscriptAnalysis(BaseModel):
    question_relevance: str = Field(description="Assessment of how well the answer relates to the question")
    answer_completeness: str = Field(description="Evaluation of how thoroughly the answer addresses all aspects of the question")
    content_analysis: str = Field(description="Analysis of the content of the answer")
    communication_skills: str = Field(description="Evaluation of the candidate's communication skills")
    critical_thinking: str = Field(description="Assessment of the candidate's critical thinking abilities")
    professional_demeanor: str = Field(description="Evaluation of the candidate's professional demeanor")
    technical_proficiency: str = Field(description="Assessment of the candidate's technical knowledge")
    soft_skills: str = Field(description="Evaluation of the candidate's soft skills")
    cultural_fit: str = Field(description="Assessment of the candidate's cultural fit")

class InterviewReview(BaseModel):
    interview_question: str = Field(description="The interview question asked")
    transcript_analysis: TranscriptAnalysis = Field(description="Detailed analysis of the interview transcript")
    areas_for_improvement: List[str] = Field(description="List of areas where the candidate can improve")
    scoring: dict = Field(description="Numerical scores for various aspects of the interview")

class InterviewQuestionReviewService:
    def __init__(self):
        self.model = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                temperature=os.getenv('OPENAI_TEMPERATURE'),
                                model_name=os.getenv('OPENAI_MODEL'),
                                top_p=os.getenv('OPENAI_TOP_P'),
                                model_kwargs={ "response_format": { "type": "json_object" } })
        
        self.parser = JsonOutputParser(pydantic_object=InterviewReview)
        
        self.prompt = PromptTemplate(
            template="""You are an expert interview evaluator. Analyze the provided interview question and transcript for the candidate comprehensively and objectively.

            IMPORTANT INSTRUCTIONS:
            1. If the transcript is empty, very short (less than 10 words), or contains no meaningful response, clearly identify this in your analysis
            2. For empty/inadequate responses, assign scores of 0-2 and explain the lack of response
            3. For substantial responses, evaluate thoroughly using the full 1-10 scale
            4. Be specific and detailed in your analysis - avoid generic feedback
            5. Consider the interview level and technologies when setting expectations

            EVALUATION CONTEXT:
            - Technologies: {interview_technologies}
            - Tags: {interview_tags}
            - Level: {interview_level}
            - Interview Question: {interview_question}
            - Interview Question Explanation: {interview_question_explanation}
            - Candidate Response/Transcript: {interview_transcription}

            SCORING GUIDELINES (1-10 scale):
            - 0-2: No response, completely inadequate, or completely incorrect
            - 3-4: Minimal effort, major gaps, fundamentally flawed understanding
            - 5-6: Basic understanding but incomplete, several important issues
            - 7-8: Good response with minor gaps, demonstrates solid understanding
            - 9-10: Excellent, comprehensive, demonstrates deep understanding and expertise

            ANALYSIS REQUIREMENTS:
            1. Question Relevance: Does the response directly address what was asked?
            2. Answer Completeness: How thoroughly does the response cover all aspects?
            3. Content Analysis: Evaluate technical accuracy, depth, and examples provided
            4. Communication Skills: Clarity, structure, and articulation of ideas
            5. Critical Thinking: Evidence of analytical reasoning and problem-solving
            6. Professional Demeanor: Appropriateness and professionalism in response
            7. Technical Proficiency: Demonstration of relevant technical knowledge for the level
            8. Soft Skills: Evidence of collaboration, adaptability, and interpersonal skills
            9. Cultural Fit: Alignment with professional values and team dynamics

            SPECIFIC REQUIREMENTS:
            - If transcript is empty/inadequate: Explicitly state "No response provided" or "Inadequate response" in relevant analysis fields
            - For each scoring category, provide specific reasoning based on the actual content (or lack thereof)
            - Areas for improvement should be actionable and specific to the candidate's performance
            - Consider the interview level when evaluating - junior candidates aren't expected to have senior-level expertise

            {format_instructions}

            Return output in JSON format only. Ensure all scores reflect the actual quality of the response, with 0-2 for no/poor responses and higher scores only for genuinely good responses.
            """,
            input_variables=["interview_technologies", "interview_tags", "interview_level", "interview_question", "interview_question_explanation", "interview_transcription"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.model | self.parser

    def _is_transcript_empty(self, transcript_text: str) -> bool:
        """Check if transcript is empty or contains only whitespace/minimal content."""
        if not transcript_text or not transcript_text.strip():
            return True
        # Consider transcript empty if it's too short to be meaningful
        return len(transcript_text.strip()) < 10

    def _generate_empty_review(self, interview_question_details) -> dict:
        """Generate a default review for empty transcripts."""
        return {
            "interview_question": interview_question_details["questionText"],
            "transcript_analysis": {
                "question_relevance": "No response provided",
                "answer_completeness": "No response provided", 
                "content_analysis": "No response provided",
                "communication_skills": "No response provided",
                "critical_thinking": "No response provided",
                "professional_demeanor": "No response provided",
                "technical_proficiency": "No response provided",
                "soft_skills": "No response provided",
                "cultural_fit": "No response provided"
            },
            "areas_for_improvement": ["Provide a response to the interview question"],
            "scoring": {
                "question_relevance": 0,
                "answer_completeness": 0,
                "content_analysis": 0,
                "communication_skills": 0,
                "critical_thinking": 0,
                "professional_demeanor": 0,
                "technical_proficiency": 0,
                "soft_skills": 0,
                "cultural_fit": 0
            },
            "average_points": 0.0,
            "average_percentage": 0.0,
            "qid": interview_question_details["qid"],
            "level": interview_question_details["level"],
            "technologies": interview_question_details["technologies"],
            "tags": interview_question_details["tags"]
        }
        

    def calculate_scores(self, scoring_data, question_points, max_score_per_category=10):
        """
        Calculate the average score and percentage based on scoring data.
        
        Parameters:
            scoring_data (dict): A dictionary of scoring categories with their respective scores.
            max_score_per_category (int): The maximum possible score for each category. Default is 10.
        
        Returns:
            dict: A dictionary containing the total score, average score, and average percentage.
        """
        total_score = sum(scoring_data.values())
        num_categories = len(scoring_data)
        average_score = total_score / num_categories
        average_percentage = (average_score / max_score_per_category) * 100
        average_points = (average_score / max_score_per_category) * question_points
        
        # Round the results to two decimal places
        average_points = round(average_points, 2)
        average_percentage = round(average_percentage, 2)

        return average_points, average_percentage

    def generate_review(self, interview_question_details) -> InterviewReview:
        # Always make API call - let the improved prompt handle empty transcripts intelligently
        logger.info("Generating review using API with enhanced analysis")
        
        review = self.chain.invoke({
            "interview_technologies":interview_question_details["technologies"], 
            "interview_tags":interview_question_details["tags"], 
            "interview_level": interview_question_details["level"],
            "interview_question": interview_question_details["questionText"],
            "interview_question_explanation":interview_question_details["questionExplanation"],
            "interview_transcription": interview_question_details["transcriptText"]
        })
        
        print(type(review['scoring']))
        print(review['scoring'])
        
        average_points, average_percentage = self.calculate_scores(review['scoring'], interview_question_details["point"])
        review["average_points"] = average_points
        review["average_percentage"] = average_percentage
        review["qid"] = interview_question_details["qid"]
        review["level"] = interview_question_details["level"]
        review["technologies"] = interview_question_details["technologies"]
        review["tags"] = interview_question_details["tags"]
        
        return review

class InterviewOverallReview(BaseModel):
    summary: list = Field(description="Overall summary of the interview")
    areas_for_improvement: List[str] = Field(description="List of areas where the candidate can improve")
    recommendation: str = Field(description="Final recommendation regarding the candidate")

class InterviewOverallReviewService:
    def __init__(self):
        self.model = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                temperature=os.getenv('OPENAI_TEMPERATURE'),
                                model_name=os.getenv('OPENAI_MODEL'),
                                top_p=os.getenv('OPENAI_TOP_P'),
                                model_kwargs={ "response_format": { "type": "json_object" } })
        
        self.parser = JsonOutputParser(pydantic_object=InterviewOverallReview)
        
        self.prompt = PromptTemplate(
            template="""You are an expert interview analyst. Analyze the provided interview reviews and create a comprehensive overall evaluation of the candidate's performance.

            IMPORTANT INSTRUCTIONS:
            1. Synthesize information from all individual interview question reviews
            2. Identify patterns across multiple responses (consistency, growth, areas of strength/weakness)
            3. If most responses were empty/inadequate, reflect this in your overall assessment
            4. Provide specific, actionable recommendations based on actual performance observed
            5. Consider the candidate's overall trajectory and potential

            ANALYSIS CONTEXT:
            {candidate_interview_review}

            EVALUATION REQUIREMENTS:
            1. Summary: Provide 3-5 key observations about the candidate's overall performance
               - Highlight strengths demonstrated across responses
               - Note consistent patterns or issues
               - Comment on overall engagement level
               - Assess readiness for the role based on evidence

            2. Areas for Improvement: List specific, actionable areas where the candidate can grow
               - Base recommendations on actual gaps observed in responses
               - Prioritize the most critical improvements needed
               - Provide constructive, specific guidance
               - If no responses were given, focus on interview participation and preparation

            3. Recommendation: Provide a clear, evidence-based recommendation
               - Strong Hire: Excellent performance across multiple areas, ready for role
               - Hire: Good performance with minor gaps, likely to succeed with support
               - No Hire: Significant gaps or concerns that outweigh strengths
               - No Assessment Possible: Insufficient responses to make a determination

            QUALITY STANDARDS:
            - Be specific and evidence-based in all assessments
            - Avoid generic feedback - reference actual performance patterns
            - Balance honesty about gaps with constructive guidance
            - Consider the cumulative evidence across all interview questions
            - Ensure recommendations align with observed performance levels

            {format_instructions}

            Return output in JSON format only. Ensure your analysis accurately reflects the candidate's actual performance across all interview responses.
            """,
            input_variables=["candidate_interview_review"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.model | self.parser

    def _has_meaningful_content(self, data: List[Dict]) -> bool:
        """Check if any individual results have meaningful content (non-zero scores)."""
        return any(
            result.get('average_percentage', 0) > 0 
            for result in data
        )

    def _generate_empty_overall_review(self) -> dict:
        """Generate a default overall review when no meaningful responses exist."""
        return {
            "summary": ["No responses provided to interview questions"],
            "areas_for_improvement": [
                "Participate actively in the interview",
                "Provide responses to interview questions",
                "Demonstrate engagement with the interview process"
            ],
            "recommendation": "Cannot recommend based on lack of participation"
        }
        

    def calculate_performance(self, average_percentage: float) -> str:
        """
        Determines the overall performance category based on average percentage.

        Parameters:
        - average_percentage (float): The calculated average percentage.

        Returns:
        - str: The performance category.
        """
        if 90 <= average_percentage <= 100:
            return "excellent"
        elif 75 <= average_percentage < 90:
            return "good"
        elif 50 <= average_percentage < 75:
            return "average"
        elif 30 <= average_percentage < 50:
            return "below_average"
        else:
            return "poor"

    def extract_and_process_interview_data(self, data: List[Dict]) -> Dict:
        """
        Processes the interview data, extracts required fields, and calculates total score,
        average percentage, and overall performance.

        Parameters:
        - data (List[Dict]): The interview review data containing interview question, transcript analysis, and areas for improvement.

        Returns:
        - Dict: A dictionary containing the processed results.
        """
        total_score = 0
        total_percentage = 0
        technologies = set()
        tags = set()
        overall_performance = ""

        for entry in data:
            total_score += entry['average_points']
            total_percentage += entry['average_percentage']
            
            # Combine technologies and tags
            technologies.update(entry['technologies'])
            tags.update(entry['tags'])

        # Calculate average percentage
        average_percentage = total_percentage / len(data)
        average_percentage = round(average_percentage, 2)

        # Calculate overall performance
        overall_performance = self.calculate_performance(average_percentage)

        # Prepare the result as a dictionary
        result = {
            "total_score": round(total_score, 2),
            "average_percentage": average_percentage,
            "overall_performance": overall_performance,
            # "technologies": list(technologies),
            # "tags": list(tags)
        }

        return result

    def extract_interview_details(self, data: List[Dict]) -> str:
        """
        Extracts interview question, transcript analysis, and areas for improvement from the provided data
        and formats them into a human-readable text format.

        Parameters:
        - data (List[Dict]): The interview review data containing interview question, transcript analysis, and areas for improvement.

        Returns:
        - str: The formatted details as a string.
        """
        result = ""
        
        for entry in data:
            interview_question = entry['interview_question']
            transcript_analysis = entry['transcript_analysis']
            areas_for_improvement = entry['areas_for_improvement']

            # Format the transcript analysis as text
            transcript_analysis_text = "\n".join([f"{key}: {value}" for key, value in transcript_analysis.items()])

            # Format areas for improvement as text
            areas_for_improvement_text = "\n- ".join(areas_for_improvement)

            # Append the formatted result
            result += f"Interview Question: {interview_question}\n\n"
            result += "Transcript Analysis:\n"
            result += transcript_analysis_text
            result += f"\n\nAreas for Improvement:\n- {areas_for_improvement_text}"
            result += "\n" + "="*40 + "\n"
        
        return result

    def add_average_scoring(self, data):
        # Access the list of individual results
        individual_results = data.get('individual_result', [])
        
        if not individual_results:
            # No individual results to calculate averages from
            return data  # Return the data as is
        
        # Collect all scoring dictionaries
        scorings = [result['scoring'] for result in individual_results if 'scoring' in result]
        
        if not scorings:
            # No scoring data available
            return data  # Return the data as is
        
        # Determine the scoring keys
        scoring_keys = scorings[0].keys()
        
        # Calculate the sum for each scoring key
        sum_scores = {key: 0 for key in scoring_keys}
        for scoring in scorings:
            for key in scoring_keys:
                sum_scores[key] += scoring[key]
        
        # Calculate the average for each scoring key
        count = len(scorings)
        average_scoring = {key: sum_scores[key] / count for key in scoring_keys}
        
        # Add the 'scoring' key to 'overall_result'
        overall_result = data.get('overall_result', {})
        overall_result['scoring'] = average_scoring
        data['overall_result'] = overall_result
        return data

    def generate_review(self, interview_question_details):
        # Check if there's meaningful content before making API call
        if not self._has_meaningful_content(interview_question_details):
            logger.info("No meaningful content detected, skipping overall API call")
            review = self._generate_empty_overall_review()
        else:
            # If not in cache, generate the review
            logger.info("Generating new review using API")
            candidate_interview_review = self.extract_interview_details(interview_question_details)
            review = self.chain.invoke({
                "candidate_interview_review":candidate_interview_review
            })
        
        final_response = dict()
        overall_result = self.extract_and_process_interview_data(interview_question_details)
        final_response["individual_result"] = interview_question_details
        final_response["overall_result"] = overall_result
        final_response["overall_result"]["summary"] = review["summary"]
        final_response["overall_result"]["recommendation"] = review["recommendation"]
        final_response = self.add_average_scoring(final_response)
        
        return final_response

if __name__ == "__main__":
    service1 = InterviewQuestionReviewService()
    service2 = InterviewOverallReviewService()
    
    response = {
    "id": "6749d0cebb9525f2254a7160",
    "profile": "student",
    "interview": [
        {
        "qid": "123e4567-e89b-12d3-a456-426614174001",
        "level": "intermediate",
        "technologies": ["Java"],
        "tags": ["thread-safety", "synchronized", "Java basics"],
        "point": 2,
        "questionText": "Explain the use of the synchronized keyword in Java and how it ensures thread safety.",
        "questionExplanation": "The synchronized keyword in Java is used to control access to critical sections of code, ensuring thread safety by preventing multiple threads from accessing the same resource simultaneously.",
        "transcriptText": "The synchronized keyword is directly related to thread safety but lacks examples and in-depth explanation."
        },
        {
        "qid": "123e4567-e89b-12d3-a456-426614174002",
        "level": "advanced",
        "technologies": ["Java"],
        "tags": ["concurrency", "thread-pooling", "Java performance"],
        "point": 3,
        "questionText": "What is the purpose of thread pooling in Java, and how does it improve application performance?",
        "questionExplanation": "Thread pooling in Java is a technique to manage threads efficiently by reusing a pool of worker threads to execute multiple tasks, reducing the overhead of creating and destroying threads repeatedly.",
        "transcriptText": "Thread pooling optimizes resource usage but requires understanding of thread management to avoid deadlocks."
        }
    ]
    }


        
    question_level_list = list()
    for interview_question_details in response["interview"]:
        interview_question_review = service1.generate_review(interview_question_details)
        question_level_list.append(interview_question_review)
        
    final_response = service2.generate_review(question_level_list)
    
    final_response["id"] = response["id"]
    final_response["profile"] = response["profile"]
    
    print(final_response)
    
    #Step 1    
    # request = {
    # "qid": "123e4567-e89b-12d3-a456-426614174002",
    # "level": "advanced",
    # "technologies": ["Java"],
    # "tags": ["concurrency", "thread-pooling", "Java performance"],
    # "point": 3,
    # "questionText": "What is the purpose of thread pooling in Java, and how does it improve application performance?",
    # "questionExplanation": "Thread pooling in Java is a technique to manage threads efficiently by reusing a pool of worker threads to execute multiple tasks, reducing the overhead of creating and destroying threads repeatedly.",
    # "transcriptText": "Thread pooling optimizes resource usage but requires understanding of thread management to avoid deadlocks."
    # }
    # request = {
    #         "qid": "123e4567-e89b-12d3-a456-426614174001",
    #         "level": "intermediate",
    #         "technologies": ["Java"],
    #         "tags": ["thread-safety", "synchronized", "Java basics"],
    #         "point": 2,
    #         "questionText": "Explain the use of the synchronized keyword in Java and how it ensures thread safety.",
    #         "questionExplanation": "The synchronized keyword in Java is used to control access to critical sections of code, ensuring thread safety by preventing multiple threads from accessing the same resource simultaneously.",
    #         "transcriptText": "The synchronized keyword is directly related to thread safety but lacks examples and in-depth explanation."
    #     }
    
    # print(service.generate_review(request))
    
    #Step 2
    # response = [{'interview_question': 'What is the purpose of thread pooling in Java, and how does it improve application performance?', 'transcript_analysis': {'question_relevance': "The candidate's response is relevant as it addresses the purpose of thread pooling in Java.", 'answer_completeness': 'The answer is somewhat incomplete as it does not fully explain how thread pooling improves application performance beyond optimizing resource usage.', 'content_analysis': 'The candidate mentions optimizing resource usage and the need for understanding thread management, which indicates some knowledge of the topic but lacks depth.', 'communication_skills': 'The candidate communicates their ideas clearly but could elaborate more on the concepts.', 'critical_thinking': 'The response shows some critical thinking regarding the implications of thread management but lacks a comprehensive analysis.', 'professional_demeanor': 'The candidate maintains a professional tone throughout the response.', 'technical_proficiency': 'The candidate demonstrates a basic understanding of thread pooling but does not delve into technical details or examples.', 'soft_skills': 'The candidate exhibits good soft skills by articulating their thoughts clearly.', 'cultural_fit': "The candidate's ability to communicate technical concepts suggests a potential fit for a collaborative environment."}, 'areas_for_improvement': ['Provide a more detailed explanation of how thread pooling improves application performance.', 'Include examples or scenarios where thread pooling is beneficial.', 'Discuss potential pitfalls or challenges associated with thread pooling, such as deadlocks.'], 'scoring': {'question_relevance': 8, 'answer_completeness': 5, 'content_analysis': 6, 'communication_skills': 7, 'critical_thinking': 5, 'professional_demeanor': 8, 'technical_proficiency': 6, 'soft_skills': 7, 'cultural_fit': 7}, 'average_points': 1.97, 'average_percentage': 65.56, 'qid': '123e4567-e89b-12d3-a456-426614174002', 'level': 'advanced', 'technologies': ['Java'], 'tags': ['concurrency', 'thread-pooling', 'Java performance']},{'interview_question': 'Explain the use of the synchronized keyword in Java and how it ensures thread safety.', 'transcript_analysis': {'question_relevance': "The candidate's response was relevant to the question, directly addressing the use of the synchronized keyword in relation to thread safety.", 'answer_completeness': 'The answer was incomplete as it lacked examples and a deeper explanation of the concept.', 'content_analysis': 'The candidate mentioned the relationship between the synchronized keyword and thread safety but did not elaborate on its implementation or provide practical examples.', 'communication_skills': 'The candidate communicated their thoughts clearly but could have structured their answer better to enhance understanding.', 'critical_thinking': 'The candidate demonstrated basic critical thinking by identifying the importance of the synchronized keyword but did not explore its implications or alternatives.', 'professional_demeanor': 'The candidate maintained a professional demeanor throughout the interview.', 'technical_proficiency': 'The candidate showed a basic understanding of the synchronized keyword but lacked depth in technical knowledge.', 'soft_skills': 'The candidate exhibited good soft skills, such as maintaining eye contact and being polite, but could improve on engaging the interviewer more effectively.', 'cultural_fit': "The candidate's approach seemed to align with the company's values of collaboration and knowledge sharing, but the lack of depth in the answer may raise concerns."}, 'areas_for_improvement': ['Provide examples to illustrate technical concepts.', 'Expand on explanations to demonstrate deeper understanding.', 'Engage more with the interviewer to clarify points.'], 'scoring': {'question_relevance': 8, 'answer_completeness': 5, 'content_analysis': 6, 'communication_skills': 7, 'critical_thinking': 5, 'professional_demeanor': 8, 'technical_proficiency': 6, 'soft_skills': 7, 'cultural_fit': 7}, 'average_points': 1.31, 'average_percentage': 65.56, 'qid': '123e4567-e89b-12d3-a456-426614174001', 'level': 'intermediate', 'technologies': ['Java'], 'tags': ['thread-safety', 'synchronized', 'Java basics']}]
    
    #service = InterviewOverallReviewService()
    # print(service.generate_review(response))
    
    
    
    
    