import streamlit as st
import ffmpeg
import speech_recognition as sr
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

@st.cache_data
def extract_audio_from_video(video_file):
    temp_video_path = "temp_video.mp4"
    temp_audio_path = "temp_audio.wav"
    
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    stream = ffmpeg.input(temp_video_path)
    stream = ffmpeg.output(stream, temp_audio_path)
    ffmpeg.run(stream)
    
    os.remove(temp_video_path)
    return temp_audio_path

@st.cache_data
def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
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

        Here is candidate transcript: {interview_transcription}
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    return chain.run(candidate_name=candidate_name, job_profile=job_profile, interview_transcription=interview_transcription)

st.title("Interview Review Application")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
job_profile = st.text_input("Enter the Job Profile")
candidate_name = st.text_input("Enter the Candidate Name")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with st.spinner("Processing video..."):
        # Extract audio
        temp_audio_path = extract_audio_from_video(uploaded_file)
        
        # Convert audio to text
        text = audio_to_text(temp_audio_path)
        st.subheader("Transcription")
        st.text_area("", text, height=200)
        
        # Clean up temporary files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    if st.button("Generate Review"):
        if api_key and job_profile and candidate_name:
            with st.spinner("Generating review..."):
                review = generate_interview_review(api_key, job_profile, candidate_name, text)
                st.subheader("Interview Review")
                st.markdown(review)
        else:
            st.warning("Please fill in all the required fields.")

