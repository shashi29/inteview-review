import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import json
from urllib.parse import urlparse

import redis
from dotenv import load_dotenv
from utils.rabbitmq_utils import RabbitMQClient
from services.interview_review_service import InterviewReviewService
from services.interview_question_level_service import InterviewQuestionReviewService, InterviewOverallReviewService

import zlib
import numpy as np


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import requests

def send_ai_result(data):
    """
    Sends a PATCH request to the specified URL with the provided data.

    Args:
        data (dict): The JSON data to send in the request body.

    Returns:
        Response: The response from the API.
    """
    url = f"https://demo.exams.api.jobprep.io/exam/api/interview-activity/external/ai-result/{data['id']}"
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": "4de1c3b5a83f3d3eb0e5f7d2422ecd5c653824e34feef35ee28a65f340be449f"
    }

    response = requests.patch(url, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        print("Request successful:", response.json())
    else:
        print("Request failed:", response.status_code, response.text)
    
class ProcessingStatus(Enum):
    """Enhanced enum for processing status"""
    STARTED = "STARTED"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    MESSAGE_QUEUED = "MESSAGE_QUEUED"
    MESSAGE_SENT = "MESSAGE_SENT"
    MESSAGE_FAILED = "MESSAGE_FAILED"
    
    def __str__(self):
        return self.value

class ServiceConfig:
    """Service configuration from environment variables"""
    
    def __init__(self):
        # RabbitMQ Configuration
        self.rabbitmq_host = self._get_env("RABBITMQ_HOST")
        self.input_queue = self._get_env("RABBITMQ_QUEUE")
        # self.output_queue = self._get_env("OUTPUT_QUEUE")
        self.rabbitmq_user = self._get_env("RABBITMQ_DEFAULT_USER")
        self.rabbitmq_pass = self._get_env("RABBITMQ_DEFAULT_PASS")
        
        # S3 Configuration
        self.s3_access_key = self._get_env("AWS_ACCESS_KEY_ID")
        self.s3_secret_key = self._get_env("AWS_SECRET_ACCESS_KEY")
        
        # Redis Configuration
        self.redis_host = self._get_env("REDIS_HOST")
        self.redis_port = int(self._get_env("REDIS_PORT"))
        self.redis_db = int(self._get_env("REDIS_DB"))
        
        # Service Configuration
        self.service_name = self._get_env("SERVICE_NAME", "interview-review-service")
        self.max_retries = int(self._get_env("MAX_RETRIES", "3"))
        self.retry_delay = int(self._get_env("RETRY_DELAY", "5"))
        self.status_check_timeout = int(self._get_env("STATUS_CHECK_TIMEOUT", "7200"))
        self.temp_dir = self._get_env("TEMP_DIR", "/tmp")
        

        
    @staticmethod
    def _get_env(key: str, default: Optional[str] = None) -> str:
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Environment variable {key} is not set")
        return value

@dataclass
class ProcessingEvent:
    """Data class for processing events"""
    service_name: str
    cid: str
    organization: str
    status: ProcessingStatus
    datetime: str
    details: Dict[str, Any]
    error: Optional[str] = None
    error_description: Optional[str] = None
    retry_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert ProcessingEvent to a Redis-compatible dictionary"""
        data = asdict(self)
        data['status'] = str(data['status'])
        data['details'] = str(data['details'])  # Convert dict to string for Redis
        return {k: str(v) if v is not None else '' for k, v in data.items()}

class RedisClient:
    """Enhanced Redis client for storing processing events"""
    
    def __init__(self, host: str, port: int, db: int):
        self.client = redis.Redis(host=host, port=port, db=db)
    
    def get_latest_status(self, service_name: str, cid: str) -> Tuple[Optional[str], Optional[str]]:
        """Get latest processing status for a CID"""
        key = f"status:{cid}"
        status_data = self.client.hgetall(key)
        
        if not status_data:
            return None, None
            
        status = status_data.get(b'status', b'').decode()
        timestamp = status_data.get(b'timestamp', b'').decode()
        return status, timestamp
    
    def store_event(self, event: ProcessingEvent) -> None:
        """Store processing event in Redis"""
        key = f"status:{event.cid}"
        event_dict = event.to_dict()
        self.client.hmset(key, event_dict)
        self.client.expire(key, 2592000)  # 30 days expiry

class ProcessingService:
    """Enhanced OCR Processing Service"""
    
    def __init__(self):
        self.config = ServiceConfig()
        self.input_queue = RabbitMQClient(self.config.rabbitmq_host, 
                                          self.config.input_queue, 
                                          self.config.rabbitmq_user, 
                                          self.config.rabbitmq_pass)

        self.redis_client = RedisClient(
            self.config.redis_host,
            self.config.redis_port,
            self.config.redis_db
        )
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.config.temp_dir, exist_ok=True)
        
        self.interview_service = InterviewReviewService()


    def log_event(
        self,
        service_name: str,
        cid: str,
        organization: str,
        status: ProcessingStatus,
        details: Dict[str, Any],
        error: Optional[str] = None,
        error_description: Optional[str] = None,
        retry_count: Optional[int] = None
    ) -> None:
        """Log processing event to Redis"""
        event = ProcessingEvent(
            service_name=service_name,
            cid=cid,
            organization=organization,
            status=status,
            datetime=datetime.utcnow().isoformat(),
            details=details,
            error=error,
            error_description=error_description,
            retry_count=retry_count
        )
        self.redis_client.store_event(event)
        logger.info(f"Logged event: {event}")

    def process_message(self, message: Dict[str, Any]) -> None:
        """Process a single message with enhanced error handling and status tracking"""
        message_id = str(uuid.uuid4())
        local_input_path = None
        local_output_path = None
        message = eval(message)
        id = message.get('id', {})

        try:
            logger.info(f"Processing message: {message}")
            
            # self.log_event(
            #     service_name="interview-review-service",
            #     cid=cid,
            #     organization="",
            #     status=ProcessingStatus.STARTED,
            #     details=message
            # )

            # Process the document
            result_flag = self._process_single_document(message, local_input_path, local_output_path)
            
            # Log success and send result
            # self.log_event(
            #     service_name="interview-review-service",
            #     cid=cid,
            #     organization="",
            #     status=ProcessingStatus.SUCCESS,
            #     details=message
            # )
            
            logger.info(f"Successfully processed document for CID: {id}")

        except Exception as e:
            error_description = traceback.format_exc()
            logger.error(f"Error processing message: {str(e)}\n{error_description}")
            
            self.log_event(
                service_name="InterviewReviewService",
                cid=id,
                organization="",
                status=ProcessingStatus.FAILED,
                details=message,
                error="ProcessingError",
                error_description=str(error_description)
            )

        finally:
            self._clean_up_temp_files(local_input_path, local_output_path)

    def _process_single_document(
        self,
        message: Dict[str, Any],
        local_input_path: Optional[str],
        local_output_path: Optional[str]
    ) -> Dict[str, Any]:
        """Process a single document with enhanced path handling"""

        # request_id = message.get('request_id')
        # job_profile = message.get('job_profile')
        # candidate_name = message.get('candidate_name')
        # interview_question = message.get('interview_question')
        # interview_transcription = message.get('interview_transcription')

        # candidate_response = self.interview_service.generate_review(job_profile, candidate_name, interview_question, interview_transcription, request_id)
        
        # # Store the candidate response in Redis without expiration
        # self.redis_client.client.hmset(f"interview_response:{request_id}", {
        #     k: json.dumps(v) if isinstance(v, (dict, list)) else v 
        #     for k, v in candidate_response.items()
        # })
        
        service1 = InterviewQuestionReviewService()
        service2 = InterviewOverallReviewService()
        
        question_level_list = list()
        for interview_question_details in message["interview"]:
            interview_question_review = service1.generate_review(interview_question_details)
            question_level_list.append(interview_question_review)
            
        final_response = service2.generate_review(question_level_list)
        
        final_response["id"] = message["id"]
        final_response["profile"] = message["profile"]        
        
        #Now save this full response using the api
        send_ai_result(final_response)
        
        return True        


    def _generate_output_key(self, input_key: str, output_filename: str) -> str:
        """Generate output key with OCR folder structure"""
        path_components = input_key.split('/')
        last_folder_index = len(path_components) - 2
        path_components[last_folder_index] = "ocr"
        path_components[-1] = output_filename
        return '/'.join(path_components[:-1]) + '/' + output_filename

    def _clean_up_temp_files(self, *file_paths: str) -> None:
        """Clean up temporary files"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")

    def start(self) -> None:
        """Start the OCR Processing Service"""
        logger.info("Starting OCR Processing Service")
        self.input_queue.start_consumer(self.process_message)

    def check_health(self) -> bool:
        """Check service health"""
        try:
            # Check Redis connection
            #self.redis_client.client.ping()
            
            # Check RabbitMQ connections
            rabbitmq_health = self.input_queue.check_health(),
            
            return rabbitmq_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

def main():
    """Main entry point"""
    try:
        service = ProcessingService()
        
        if service.check_health():
            logger.info("Health check passed. Starting Interview Review Service Service.")
            service.start()
        else:
            logger.error("Health check failed. Please check your connections.")
            exit(1)
    except Exception as e:
        logger.critical(f"Service failed to start: {e}")
        exit(1)

if __name__ == "__main__":
    main()