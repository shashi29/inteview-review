
import json
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional

from rabbitmq_interview_review_service import ProcessingService, ServiceConfig, ProcessingStatus 
from generate_interview_question import InterviewQuestionService
from utils.rabbitmq_utils import RabbitMQClient
import logging
import os

# Security Constants
SECURITY_CONSTANTS = {
    "API_KEY_HEADER_NAME": "X-API-Key",
    "MAX_REQUEST_BODY_SIZE": 10 * 1024 * 1024,  # 10 MB
    "REQUEST_TIMEOUT_SECONDS": 30,
    "MAX_CONCURRENT_REQUESTS": 100,
    "RATE_LIMIT_PER_MINUTE": 60,
    "REDIS_RESPONSE_EXPIRY_SECONDS": 3600  # 1 hour
}

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InterviewRequest(BaseModel):
    job_title: str
    experience: str
    skills: str
    job_description: str
    number_of_questions: Optional[int] = 3
    question_level: Optional[str] = "intermediate"
    
class InterviewSubmissionRequest(BaseModel):
    """
    Pydantic model for interview submission request.
    
    Attributes:
        request_id (str): Unique identifier for the submission.
        job_profile (str): Profile for which interview is conducted.
        candidate_name (str): Name of the candidate.
        interview_question (str): Main interview question.
        interview_transcription (str): Full transcription of the interview.
    """
    request_id: str
    job_profile: str
    candidate_name: str
    interview_question: str
    interview_transcription: str

# API Key authentication
api_key_header = APIKeyHeader(name=SECURITY_CONSTANTS["API_KEY_HEADER_NAME"])

def validate_api_key(api_key: str = Security(api_key_header)):
    """
    Validate the provided API key against expected value.
    
    Args:
        api_key (str): API key to validate.
    
    Raises:
        HTTPException: If API key is invalid.
    """
    expected_api_key = os.getenv('HARD_CODED_TOKEN', 'default_secure_key')
    if api_key != expected_api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")

app = FastAPI(
    title="Interview Review Service",
    description="Secure service for submitting and retrieving interview reviews",
    docs_url="/docs",
    redoc_url="/redoc"
)

config = ServiceConfig()
processing_service = ProcessingService()
output_queue = RabbitMQClient(
    config.rabbitmq_host, 
    config.input_queue, 
    config.rabbitmq_user, 
    config.rabbitmq_pass
)

@app.post("/submit-interview", dependencies=[Security(validate_api_key)], tags=["Interview Score Service"])
async def submit_interview(request: InterviewSubmissionRequest):
    """
    Submit an interview for processing.
    
    Args:
        request (InterviewSubmissionRequest): Details of the interview submission.
    
    Returns:
        dict: Submission status and request ID.
    
    Raises:
        HTTPException: If submission fails.
    """
    try:
        # Validate request size
        if len(request.interview_transcription) > SECURITY_CONSTANTS["MAX_REQUEST_BODY_SIZE"]:
            raise HTTPException(status_code=413, detail="Request payload too large")
        
        # Prepare message for queue
        message = {
            "request_id": request.request_id,
            "job_profile": request.job_profile,
            "candidate_name": request.candidate_name,
            "interview_question": request.interview_question,
            "interview_transcription": request.interview_transcription
        }
        
        # Send message to RabbitMQ
        output_queue.publish_message(message)
        
        return {
            "status": "success", 
            "message": "Interview submitted for processing",
            "request_id": request.request_id
        }
    except Exception as e:
        logger.error(f"Interview submission error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interview-response/{request_id}", dependencies=[Security(validate_api_key)], tags=["Interview Score Service"])
async def get_interview_response(request_id: str):
    """
    Retrieve interview response by request ID.
    
    Args:
        request_id (str): Unique identifier for the interview response.
    
    Returns:
        Dict[str, Any]: Parsed interview response.
    
    Raises:
        HTTPException: If response retrieval fails or no response found.
    """
    try:
        # Retrieve Redis data
        response_data = processing_service.redis_client.client.hgetall(f"interview_response:{request_id}")
        
        # Check if response exists
        if not response_data:
            raise HTTPException(
                status_code=404, 
                detail=f"No interview response found for request ID: {request_id}"
            )
        
        # Parse and decode response
        decoded_response: Dict[str, Any] = {}
        for key, value in response_data.items():
            try:
                # Decode key and value
                decoded_key = key.decode('utf-8')
                decoded_value = value.decode('utf-8')
                
                # Special handling for specific keys
                if decoded_key == 'request_id':
                    decoded_response[decoded_key] = decoded_value
                else:
                    try:
                        # Attempt to parse JSON for complex types
                        parsed_value = json.loads(decoded_value)
                        decoded_response[decoded_key] = parsed_value
                    except json.JSONDecodeError:
                        # Fallback to string if JSON parsing fails
                        decoded_response[decoded_key] = decoded_value
            
            except Exception as decode_error:
                # Log individual key decoding errors
                logger.warning(f"Error decoding key: {key}. Error: {decode_error}")
        
        # Log successful retrieval
        logger.info(f"Successfully retrieved interview response for request ID: {request_id}")
        
        return decoded_response
    
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error retrieving interview response: {e}", exc_info=True)
        
        # Return a 500 internal server error
        raise HTTPException(
            status_code=500, 
            detail="Internal server error occurred while retrieving interview response"
        )
        
@app.post("/generate-interview-questions/",  tags=["Generate Interview Questions"] , dependencies=[Security(validate_api_key)])
async def generate_questions(request: InterviewRequest):
    try:
        questions = InterviewQuestionService().generate_interview_questions(
            job_title=request.job_title,
            experience=request.experience,
            skills=request.skills,
            job_description=request.job_description,
            number_of_questions=request.number_of_questions,
            question_level=request.question_level,
        )
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        


@app.get("/health", tags=["Health"], dependencies=[Security(validate_api_key)])
async def health_check():
    """
    Health check endpoint to verify if the service is running.
    """
    try:
        # Add any additional checks here (e.g., database connection, third-party service availability)
        return {"status": "healthy", "message": "Service is up and running"}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}
    
def start_api_server():
    """
    Start the FastAPI server using Uvicorn.
    
    Runs the server on all interfaces at port 8082.
    """
    uvicorn.run(app, host="0.0.0.0", port=8082)

if __name__ == "__main__":
    start_api_server()