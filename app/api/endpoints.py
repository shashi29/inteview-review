import os
import tempfile
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status
from app.models.schemas import InterviewReviewRequest, InterviewReviewResponse, TranscriptionResponse
from app.services.audio_service import AudioService
from app.services.transcription_service import TranscriptionService
from app.services.interview_review_service import InterviewReviewService
from app.core.dependencies import get_interview_review_service
from app.config import Settings
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload-video/", response_model=TranscriptionResponse)
async def upload_video(video: UploadFile = File(...)):
    try:
        # Extract audio from the uploaded video and save it to the temp file
        temp_audio_path = AudioService.extract_audio_from_video(video)
        
        # Transcribe the audio using the appropriate model
        if Settings.AUDIO_MODEL == "GCP":
            transcription_text = TranscriptionService.audio_to_text_gcp(temp_audio_path)
        else:
            transcription_text = TranscriptionService.audio_to_text(temp_audio_path)
    
        # Generate a structured response payload with a timestamp
        response_payload = TranscriptionResponse(
            status="success",
            message="Video processed successfully",
            transcription=transcription_text,
            timestamp=datetime.utcnow().isoformat()
        )
        
        return response_payload

    except Exception as e:
        # Log the error and return a structured error payload with a timestamp
        logger.error(f"Error during video upload and transcription: {str(e)}")
        error_payload = TranscriptionResponse(
            status="error",
            message="An error occurred during video processing",
            transcription=None,
            timestamp=datetime.utcnow().isoformat()
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_payload.dict())

    finally:
        # Clean up the temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    
@router.post("/evaluate-response/", response_model=dict())
async def evaluate_response(
    request: InterviewReviewRequest,
    interview_review_service: InterviewReviewService = Depends(get_interview_review_service)
):

    review = interview_review_service.generate_review(
        job_profile=request.job_profile,
        candidate_name=request.candidate_name,
        interview_question=request.interview_question,
        interview_transcription=request.interview_transcription
    )
    return review#InterviewReviewResponse(review=review)
