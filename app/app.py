import json
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rabbitmq_interview_review_service import ProcessingService, ServiceConfig,ProcessingStatus 
from utils.rabbitmq_utils import RabbitMQClient
import logging

# Create a router for better modularity
# interview_router = APIRouter(prefix="/interview", tags=["Interview"])

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InterviewSubmissionRequest(BaseModel):
    request_id: str
    job_profile: str
    candidate_name: str
    interview_question: str
    interview_transcription: str

app = FastAPI(title="Interview Review Service")
config = ServiceConfig()
processing_service = ProcessingService()
output_queue = RabbitMQClient(
    config.rabbitmq_host, 
    config.input_queue, 
    config.rabbitmq_user, 
    config.rabbitmq_pass
)

@app.post("/submit-interview")
async def submit_interview(request: InterviewSubmissionRequest):
    try:
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/interview-response/{request_id}")
async def get_interview_response(request_id: str):
    """
    Retrieve interview response by request ID
    
    Args:
        request_id (str): Unique identifier for the interview response
    
    Returns:
        Dict[str, Any]: Parsed interview response
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
        
# async def get_interview_response(request_id: str):
#     response_data = processing_service.redis_client.client.hgetall(f"interview_response:{request_id}")
    

#     if not response_data:
#         return None
    
#     # Convert the Redis response to a more usable dictionary
#     decoded_response = {}
#     for key, value in response_data.items():
#         # Decode key and value
#         decoded_key = key.decode('utf-8')
#         decoded_value = value.decode('utf-8')
        
#         # Special handling for specific keys
#         if decoded_key == 'request_id':
#             decoded_response[decoded_key] = decoded_value
#         else:
#             try:
#                 # Try to parse JSON for complex types
#                 decoded_response[decoded_key] = json.loads(decoded_value)
#             except json.JSONDecodeError:
#                 # If JSON parsing fails, store as string
#                 decoded_response[decoded_key] = decoded_value
    
    
#     return response_data

def start_api_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_api_server()