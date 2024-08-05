import ffmpeg
import speech_recognition as sr
import os
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

def extract_audio_from_video(video_file_path, output_audio_file_path):
    stream = ffmpeg.input(video_file_path)
    stream = ffmpeg.output(stream, output_audio_file_path)
    ffmpeg.run(stream)

def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)  # Correct method to use recognizer
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        
def generate_interview_review(api_key, job_profile, candidate_name, interview_transcription):
    llm = ChatOpenAI(api_key=api_key)
    prompt_template = PromptTemplate(
        input_variables=["candidate_name", "job_profile", "interview_transcription"],
        template="""
        Analyze the provided interview transcript for {candidate_name} applying for the position {job_profile}. Follow these steps:

        Transcript Review:

        Read the entire transcript carefully
        Note any unclear or potentially erroneous sections


        Response Evaluation:

        Assess relevance to questions asked
        Evaluate depth of knowledge demonstrated
        Rate clarity and coherence (1-5 scale)


        Skills Assessment:

        Identify examples of:
        a) Problem-solving
        b) Critical thinking
        c) Communication skills
        d) Technical expertise
        Rate each skill area (1-5 scale)


        Cultural Fit:

        Note indicators of:
        a) Teamwork
        b) Adaptability
        c) Alignment with company values
        Provide an overall cultural fit score (1-5 scale)


        Quantitative Analysis:

        Calculate:
        a) Average response length
        b) Frequency of industry-specific terms
        c) Ratio of concrete examples to general statements


        STAR Method:

        Identify responses following the Situation, Task, Action, Result format
        Rate the effectiveness of STAR responses (1-5 scale)


        Red Flags:

        List any concerning statements or inconsistencies


        Strengths and Areas for Improvement:

        Summarize top 3 strengths
        Identify 2-3 areas for potential development


        Overall Assessment:

        Provide a brief overall evaluation (2-3 sentences)
        Assign a total score (1-100 scale)


        Recommendation:

        Suggest next steps (e.g., additional interview, skills test, reference check, job offer)

        Remember to:

        Provide specific quotes from the transcript to support your analysis
        Maintain objectivity throughout the evaluation
        Consider the specific requirements of the {job_profile} role
        Compare the candidate's responses to industry standards and best practices

        After completing your analysis, summarize your findings in a concise report, highlighting key points that will aid in the hiring decision.

        Here is candidate transacript: {interview_transcription}
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    return chain.run(candidate_name=candidate_name, job_profile=job_profile, interview_transcription=interview_transcription)

# Main execution
if __name__ == "__main__":
    video_file_path = '/workspaces/inteview-review/test_video/CJ9c1ab6ee541386b003bb224ed0e5e532.mp4'
    output_audio_file_path = 'output_audio.wav'

    # Step 1: Extract Audio from Video
    extract_audio_from_video(video_file_path, output_audio_file_path)

    # Step 2: Convert Audio to Text
    text = audio_to_text(output_audio_file_path)
    print(text)
    os.remove(output_audio_file_path) 

    # Step 3: Generate Interview Review
    api_key = ""  # Replace with your actual API key
    job_profile = "Python Developer"  # Replace with the actual job profile
    candidate_name = "Shashi Raj"  # Replace with the actual candidate name

    review = generate_interview_review(api_key, job_profile, candidate_name, text)
    print(review)