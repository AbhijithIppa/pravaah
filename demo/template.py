import streamlit as st
from datetime import datetime
import json
import streamlit_lottie as st_lottie
from streamlit_ace import st_ace

# Function to load Lottie animation
def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

human_talking = load_lottie(r"D:\abhijith\ML\pravaah\assets\coat.json")

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
    messages.append({'username': "AI", 'text': "Hey!!!", 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

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
        .lottie-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .lottie-column {
            display: flex;
            justify-content: center;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">AI Bot</h1>', unsafe_allow_html=True)

    # Layout with Lottie animations and code editor
    # col1, col2 = st.columns([2, 1])
    # with col1:
    st.markdown('<div class="lottie-container">', unsafe_allow_html=True)
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
    with sub_col1:
        st_lottie.st_lottie(human_talking, key="1", height=200, width=200)
    with sub_col2:
        st_lottie.st_lottie(human_talking, key="2", height=200, width=200)
    with sub_col3:
        st_lottie.st_lottie(human_talking, key="3", height=200, width=200)
    st.markdown('</div>', unsafe_allow_html=True)
    # with col2:
    st.markdown('<h3>Code Editor</h3>', unsafe_allow_html=True)
    with st.form(key='code_form'):
        code = st_ace(
            value='print("Hello, Streamlit!")',
            language='python',
            theme='monokai',
            key='ace-editor',
            height=300,
        )
        submit_code_button = st.form_submit_button(label='Submit Code')
        if submit_code_button:
            st.session_state['saved_code'] = code
            st.success('Code saved!')



    # Display last chat message
    messages = load_messages()
    if messages:
        last_message = messages[-1]
        st.markdown(
            f"""
            <div class="chat-message">
                <div>
                    <strong>{last_message['username']}</strong>: {last_message['text']}
                </div>
                <div class="message-time">{last_message['time']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Input form for new messages
    with st.form(key='message_form', clear_on_submit=True):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        username = "User"  # Hardcoded username
        message_text = st.text_input("Your message")
        submit_button = st.form_submit_button(label='Send')

        st.markdown('</div>', unsafe_allow_html=True)

    if submit_button and message_text:
        save_message(username, message_text)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
