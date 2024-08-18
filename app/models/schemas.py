from pydantic import BaseModel, Field
from typing import Optional

class TranscriptionResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(..., example="Video processed successfully")
    transcription: Optional[str] = Field(None, example="This is the transcribed text from the audio.")
    timestamp: str = Field(..., example="2024-08-18T12:34:56.789Z")

class InterviewReviewRequest(BaseModel):
    job_profile: str = Field(..., example="Azure Cloud Engineer")
    candidate_name: str = Field(..., example="John Doe")
    interview_transcription: str = Field(..., example=""" what will be question "ok so basically setting up disaster recovery solution in Azure cloud includes a several steps that ensure that your application and data are protected and can be quickly recovered in the event of disaster so first of all we have to access our needs so identifying the critical application and data is one of the most important task second one is the RPO and recovery point objective and recovery time objective which is the maximum acceptable amount of data loss or measure in time of and RTO is the maximum acceptable basically choose the right as your services so one of the predominant services as your site recovery which is comprehensive so these are like recovery solution that allows you to replicate on premises physical servers and virtual machines to assure or two additionally we can use Azure traffic Manager which help us to automatic failover by routing the traffic to best available and your backup which is one of the services that is provided like a backup solutions for your answer application of virtual machines we can customise the replication policies by defining replication policies including RPO retention period and failover and ones can we can implement the network connectivity like we ensure that a network supports the disaster recovery VPN or Express route connections between on premises and environmental show traffic manager as well so by setting up the traffic manager to root the traffic to appropriate region in case of failure done with this we can test or disaster recovery plan like by performing a test recovery plan so based on the result of test failover in order to address is shows and after that we can monitor and maintain by continuing monitoring with the help of your monitor service which can help us out continuously monitoring your disaster recovery environment and based on that we can update our recovery plans of periodically so this is what""")
    interview_question: str = Field(..., example="Can you explain how to set up a disaster recovery solution in Azure Cloud?")

class InterviewReviewResponse(BaseModel):
    review: str = Field(..., example="The candidate showed strong knowledge in API development but lacked experience in cloud deployment.")
