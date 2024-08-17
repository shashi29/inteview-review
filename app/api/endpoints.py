import os
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from app.models.schemas import InterviewReviewRequest, InterviewReviewResponse, TranscriptionResponse
from app.services.audio_service import AudioService
from app.services.transcription_service import TranscriptionService
from app.services.interview_review_service import InterviewReviewService
from app.core.dependencies import get_interview_review_service
from app.config import Settings

router = APIRouter()

@router.post("/upload-video/", response_model=TranscriptionResponse)
async def upload_video(video: UploadFile = File(...)):
    try:
        temp_audio_path = AudioService.extract_audio_from_video(video)
        if Settings.AUDIO_MODEL == "GCP":
            text = TranscriptionService.audio_to_text_gcp(temp_audio_path)
        else:
            text = TranscriptionService.audio_to_text(temp_audio_path)

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        return TranscriptionResponse(transcription=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate-response/", response_model=InterviewReviewResponse)
async def evaluate_response(
    request: InterviewReviewRequest,
    interview_review_service: InterviewReviewService = Depends(get_interview_review_service)
):
    try:
        review = interview_review_service.generate_review(
            request.job_profile,
            request.candidate_name,
            request.interview_transcription
        )
        return InterviewReviewResponse(review=review)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))