from pydantic import BaseModel

class InterviewReviewRequest(BaseModel):
    job_profile: str
    candidate_name: str
    interview_transcription: str

class InterviewReviewResponse(BaseModel):
    review: str

class TranscriptionResponse(BaseModel):
    transcription: str