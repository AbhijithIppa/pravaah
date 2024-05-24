import speech_recognition as sr
import streamlit as st
# import chat1
def speech_to_text():
    recognizer = sr.Recognizer()

    # Provide the path to your audio file
    audio_file_path = r"D:\abhijith\ML\pravaah\recorded_audio.wav"  

    with sr.AudioFile(audio_file_path) as audio_file:
        recognizer.adjust_for_ambient_noise(audio_file, duration=0.2)
        audio = recognizer.record(audio_file)
        try:
            text = recognizer.recognize_google(audio)
            text = text.lower()
            with open(r"D:\abhijith\ML\pravaah\client\user_chatbot_msg.txt", "w") as file:
                file.write(text)
            
            print(f"{text}.")
        except sr.UnknownValueError:
            print("Could not understand audio")