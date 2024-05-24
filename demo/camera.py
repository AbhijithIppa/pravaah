import streamlit as st
from datetime import datetime, timedelta
import pyaudio
import wave
import os
import readtext
import sp
import speech_recognition as sr
from streamlit_ace import st_ace

class AudioRecorder:
    def _init_(self):
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

# Function to load messages
def load_messages():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    return st.session_state['messages']

# Function to save a new message
def save_message(username, text):
    messages = load_messages()
    messages.append({'username': username, 'text': text, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    st.session_state['messages'] = messages
    messages.append({'username':"AI", 'text':"Hey!!!", 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

# Main app
def main():
    st.set_page_config(page_title="Chat App", page_icon=":speech_balloon:")        
    # Custom CSS for styling
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

    st.markdown('<h1 class="main-title">AI Bot</h1>', unsafe_allow_html=True)

    # Sidebar for user information
    with st.sidebar:
        st.header("User Info")
        username = st.text_input("Username", key="username")
        if not username:
            st.warning("Please enter a username to join the chat")
            st.stop()

    st.header("Chat")
    if "my_class" not in st.session_state:
        st.session_state.my_class = instantiate_class()
    
    if st.button("Start Recording"):
        st.session_state.my_class.start_recording()

    if st.button("Stop Recording"):
        st.session_state.my_class.stop_recording()
        st.session_state['reattempt_timer'] = datetime.now() + timedelta(seconds=60)
        st.session_state['reattempt_available'] = True  # Set flag to indicate reattempt is available

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
       # message_text = st.text_input("Your Message")  # Provide a non-empty label
        submit_button = st.form_submit_button(label='Send')
        st.markdown('</div>', unsafe_allow_html=True)

    if submit_button and message_text:
        save_message(st.session_state['username'], message_text)
        st.experimental_rerun()

    # Run speech-to-text
    if os.path.exists("recorded_audio.wav"):
        try:
            sp.speech_to_text()
        except Exception as e:
            st.error(f"Error processing audio: {e}")
    else:
        st.warning("No recorded audio found to process.")
    st.markdown('<h3>Code Editor</h3>', unsafe_allow_html=True)
    with st.form(key='code_form'):
        code = st_ace(value='print("Hello, Streamlit!")', language='python', theme='monokai', key='ace-editor', height=300)
        submit_code_button = st.form_submit_button(label='Submit Code')
        if submit_code_button:
            st.session_state['saved_code'] = code
            with open('./code_answer.txt', "w") as f:
                f.write(f"{code}\n")
            st.success('Code saved!')

if _name_ == "_main_":
    main()