from app.services.interview_review_service import InterviewReviewService

def get_interview_review_service() -> InterviewReviewService:
    return InterviewReviewService()