import streamlit as st
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from fpdf import FPDF
import streamlit_lottie as st_lottie
from streamlit_ace import st_ace
import pyaudio
import wave
import readtext
import sp

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize variables and arrays
technology_arr = []
levels_arr = ["basic", "normal", "hard"]
candidate_response = ""
ai_answer = ""
response_ans = [""]
message_text = ""
memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI()
response=[]
# Load Lottie animation
def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

human_talking = load_lottie(r"D:\abhijith\ML\pravaah\assets\coat.json")

# Load rules from file
with open(r"D:\abhijith\ML\pravaah\production\rules.txt", "r") as f:
    for line in f:
        key, value = line.split(': ')
        value = value.strip()
        if key == "Job Description":
            job_description = value
        elif key == "Technology":
            technology_arr.append(value)
        elif key == "Instructions":
            instructions = value


class AudioRecorder:
    def __init__(self):
        print("CALLEDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        self.filename = "recorded_audio.wav"
        self.channels = 1
        self.sample_rate = 44100
        self.chunk = 1024
        self.audio_format = pyaudio.paInt16
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.input_device_index = None

        self.list_input_devices()

    def list_input_devices(self):
        info = self.p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        for i in range(num_devices):
            device_info = self.p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                self.input_device_index = i  # Choose the first available input device

    def start_recording(self):
        self.stream = self.p.open(format=self.audio_format,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  input=True,
                                  input_device_index=self.input_device_index,
                                  frames_per_buffer=self.chunk)

        st.session_state['recording'] = True
        self.frames = []
        while st.session_state['recording']:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
        
    def stop_recording(self):
        st.session_state['recording'] = False
        self.stream.stop_stream()
        self.stream.close()

        # Save the recorded audio as a PCM WAV file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

@st.cache_resource
def instantiate_class():
    return AudioRecorder()












# Functions for AI interaction
def get_ai_response(candidate_response, k):
    if k == 0:
        template = """You are an expert recruiter conducting a technical interview for a software engineering position.
        Previous conversation:
        {chat_history}
        AI: You ask 1 question based on the {tech_level} of questions provided. also provide very short answer. you must return question:"",answer:""
        """
        prompt = PromptTemplate(input_variables=["chat_history", "tech_level"], template=template)
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
        tech = f"Technologies required: {technology_arr} Job Description: {job_description} Other Instructions: {instructions}"
        data = llm_chain({"tech_level": tech})
    else:
        template = """You are an expert recruiter conducting a technical interview for a software engineering position.
        Previous conversation:
        {chat_history}
        AI: You ask 1 question based on the user input {candidate_response} of questions provided. also provide very short answer. you must return first question:"",answer:""
        """
        prompt = PromptTemplate(input_variables=["chat_history", "candidate_response"], template=template)
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
        data = llm_chain({"candidate_response": candidate_response})

    question, answer = data['text'].split("\n")[1], data['text'].split("\n")[2]
    response.append(question)
    response_ans.append(answer)
    memory.chat_memory.add_user_message(candidate_response)
    return response[-1]

def get_project_response(candidate_response):
    def load_pdf(file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    pdf_file_path = 'path/to/resume.pdf'
    documents = load_pdf(pdf_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    llm = OpenAI()
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    query = "Act as AI interviewer, ask 1 question about his projects"
    docs = vectorstore.similarity_search(query)
    answer = qa_chain.run(input_documents=docs, question=query)
    return answer

def save_transcript(chat_history):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Chat Transcript', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_title('Chat Transcript')
    pdf.set_author('Generated by Python Script')
    pdf.set_font('Arial', '', 12)
    for line in chat_history.split('\n'):
        pdf.cell(0, 10, line, 0, 1)
    pdf.output('chat_transcript.pdf')
    st.success("PDF created successfully!")

# Load chat messages
def load_messages():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'username': "AI", 'text': "I am your AI interviewer. Enter 'ready' to start the interview.", 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]
    return st.session_state['messages']

# Save a new message
def save_message(username, text):
    messages = load_messages()
    messages.append({'username': username, 'text': text, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    st.session_state['messages'] = messages

def get_code():
    global response 
    template = """You are an expert coder conducting a technical interview for a software engineering position. 
    Previous conversation:
    {chat_history}
    AI:You ask 1 coding question where user need to write code on editor based on the DSA of questions provided. you MUST question:""
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "tech_level"], template=template
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    tech = str("Technologies required :" + str(technology_arr) + " Job Descrption:" + job_description + " Other Instructions: " + instructions)
    data = llm_chain({"tech_level": tech})
    question = data['text'].split("\n")[1]
    response.append(question)
    candidate_response = ""
    memory.chat_memory.add_user_message(candidate_response)
    return question

def check_code( code_ans):
    template = """You are an expert coder conducting a technical interview for a software engineering position. 
    Previous conversation:
    {chat_history}
    AI:Check whether answer is correct code for given question {tech_level}, If YES, then add some trick in question,If NO ask technical question based on user mistake
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "tech_level"], template=template
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    tech = str("previous question"+ " User Response" + str(code_ans))
    data = llm_chain({"tech_level": tech})
    question = data['text'].split("\n")[1]
    response.append(question)
    candidate_response = ""
    memory.chat_memory.add_user_message(candidate_response)
    return question

def call_code_check():
    code_answer = ""
    with open("./code_answer.txt", "r") as f:
        for line in f:
            code_answer += line
    ai_question = check_code(code_answer)
    save_message("AI", ai_question)

    return ai_question


# Main app
def main():
    st.set_page_config(page_title="Chat App", page_icon=":speech_balloon:")

    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            color: #FFA500;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar .block-container {
            padding-top: 2rem;
        }
        .chat-message {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            background-color: #333333;
            color: white;
            display: flex;
            align-items: center;
        }
        .chat-message:hover {
            background-color: #2b2b2b;
        }
        .message-time {
            font-size: 0.8rem;
            color: #888;
            margin-left: auto;
        }
        .form-container {
            margin-top: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .form-container .css-1cpxqw2 {
            flex-grow: 1;
            margin-right: 10px;
        }
        .form-container .stButton button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        .form-container .stButton button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)
    if st.button("Submit", key="transcript_button"):
        messages = memory.load_memory_variables({})
        print(messages)
        save_transcript(messages['chat_history'])
    st.markdown('<h1 class="main-title">AI Bot</h1>', unsafe_allow_html=True)
    st.markdown('<div class="lottie-container">', unsafe_allow_html=True)
    # sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
    # with sub_col1:
    st_lottie.st_lottie(human_talking, key="1", height=200, width=200)
    # with sub_col2:
    #     st_lottie.st_lottie(human_talking, key="2", height=200, width=200)
    # with sub_col3:
    #     st_lottie.st_lottie(human_talking, key="3", height=200, width=200)
    st.markdown('</div>', unsafe_allow_html=True)

    # Toggle button for code editor visibility
    if 'editor_visible' not in st.session_state:
        st.session_state['editor_visible'] = False

    if st.button("Toggle Code Editor"):
        st.session_state['editor_visible'] = not st.session_state['editor_visible']

    if st.session_state['editor_visible']:
        st.markdown('<h3>Code Editor</h3>', unsafe_allow_html=True)
        with st.form(key='code_form'):
            code = st_ace(value='print("Hello, Streamlit!")', language='python', theme='monokai', key='ace-editor', height=300)
            submit_code_button = st.form_submit_button(label='Submit Code')
            if submit_code_button:
                st.session_state['saved_code'] = code
                with open('./code_answer.txt', "w") as f:
                    f.write(f"{code}\n")
                ai_question=call_code_check()
                save_message(st.session_state['username'], code)

                st.success('Code saved!')


    # Sidebar for user information
    with st.sidebar:
        st.header("User Info")
        username = st.text_input("Username", key="username")
        if not username:
            st.warning("Please enter a username to join the chat")
            st.stop()


    st.sidebar.header("Audio Controls")
    if "my_class" not in st.session_state:
        st.session_state.my_class = instantiate_class()
    
    if st.sidebar.button("Start Recording"):
        st.session_state.my_class.start_recording()

    if st.sidebar.button("Stop Recording"):
        st.session_state.my_class.stop_recording()

    # if st.sidebar.button("Reattempt"):
    #     st.session_state.my_class.reattempt()

    # Display chat messages
    messages = load_messages()
    for message in messages:
        st.markdown(
            f"""
            <div class="chat-message">
                <div>
                    <strong>{message['username']}</strong>: {message['text']}
                </div>
                <div class="message-time">{message['time']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Input form for new messages
    with st.form(key='message_form', clear_on_submit=True):
        
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        message_text=readtext.read_text_file()
        # save_message(st.session_state['username'], message_text)
        submit_button = st.form_submit_button(label='Send')

        st.markdown('</div>', unsafe_allow_html=True)
    if os.path.exists("recorded_audio.wav"):
        try:
            sp.speech_to_text()
        except Exception as e:
            st.error(f"Error processing audio: {e}")
    else:
        st.warning("No recorded audio found to process.")

    if submit_button and message_text:
        global k
        save_message(st.session_state['username'], message_text)
        # save_message(st.session_state['username'], message_text)
        latest_user_message = message_text

            # Run speech-to-text
        print("kkkkkkkkkkkkkkkkkk",k)
        if k == 0:
            ai_question = get_ai_response("", k)
        elif k < 2:
            ai_question = get_ai_response(latest_user_message, k)
        elif k==3:
            ai_question=get_project_response("")
        
        elif k >= 4:
            ai_question = get_code()
        

        k += 1
        save_message("AI", ai_question)
        st.experimental_rerun()

if __name__ == "__main__":
    k = 3
    response = []

    main()