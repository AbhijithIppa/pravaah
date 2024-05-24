import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# Set the page configuration
st.set_page_config(page_title="Creative Dashboard", layout="wide")

# Add centered page heading
st.markdown("<h1 style='text-align: center;'>Candidate Interview Analysis</h1>", unsafe_allow_html=True)

# Define and display two circular progress bars side by side
score1 = 75
score2 = 72  

def emotion():
    face_classifier = cv2.CascadeClassifier(r'D:\abhijith\ML\pravaah\models\emotion\haarcascade_frontalface_default.xml')
    classifier = load_model(r'D:\abhijith\ML\pravaah\models\emotion\model.h5')

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    video_path = r"D:\abhijith\ML\pravaah\video.avi"  # Change this to the path of your video file
    cap = cv2.VideoCapture(video_path)

    emotion_counts = {label: 0 for label in emotion_labels}  # Initialize counts to zero

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                emotion_counts[label] += 1  # Increment count for detected emotion
            else:
                pass

    print("Emotion Counts:")
    for emotion, count in emotion_counts.items():
        if(count>30):
            print(f"{emotion}: {count}")
            emotion_counts[emotion]=int(count/30)

    cap.release()
    cv2.destroyAllWindows()

    return emotion_counts


emotion_counts=emotion()


fig_progress1 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score1,
    title={'text': "Resume Score"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "#4CAF50", 'thickness': 0.2},
           'steps': [
               {'range': [0, 58], 'color': "#FF6347"},
               {'range': [58, 100], 'color': "#90EE90"}]}))

fig_progress2 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score2,
    title={'text': "AI Score"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "#4CAF50", 'thickness': 0.2},
           'steps': [
               {'range': [0, 72], 'color': "#FF6347"},
               {'range': [72, 100], 'color': "#90EE90"}]}))

# Custom CSS for centering and responsiveness
st.markdown("""
    <style>
        .plotly-container {
            display: flex;
            justify-content: space-between;
        }
        .plotly-chart {
            width: 48%;
        }
    </style>
""", unsafe_allow_html=True)

# Display the circular progress bars side by side
progress_container = st.container()
with progress_container:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_progress1, use_container_width=True)
    with col2:
        st.plotly_chart(fig_progress2, use_container_width=True)

# Sample data for the increasing line plot with constant sections
np.random.seed(42)
x = np.arange(25)
y = np.cumsum(np.random.choice([0, 1], size=25))  # Increasing with some constant values
data_line = pd.DataFrame({'x': x, 'y': y})
fig_line = px.line(data_line, x='x', y='y', title='Iris Analysis')
fig_line.update_layout(xaxis_title='No of Deviations', yaxis_title='No of Warnings')

# New bar chart with emojis on x-axis and number of times they were expressed on y-axis
data_binary_bar = pd.DataFrame({
    'x': ['Happy😊', 'Disgust😢', 'Angry😡', 'Surprised😲', 'Fear😎'], 
    'y': [emotion_counts['Happy'], emotion_counts['Disgust'], emotion_counts['Angry'], emotion_counts['Surprise'], emotion_counts['Fear']]
})
fig_binary_bar = go.Figure(go.Bar(x=data_binary_bar['x'], y=data_binary_bar['y'], marker_color='lightgreen'))
fig_binary_bar.update_layout(title='Emotion Detection Analysis', yaxis_title='No of Times Expressed', xaxis_title='Emotions', width=400)

# Adding annotations for descriptions of each emoji

# annotations = [
#     dict(x='Happy😊', y=emotion_counts['Happy'], text='6', showarrow=False, font=dict(size=15)),
#     dict(x='Tensed😢', y=emotion_counts['Disgust'], text='3', showarrow=False, font=dict(size=15)),
#     dict(x='Angry😡', y=emotion_counts['Angry'], text='0', showarrow=False, font=dict(size=15)),
#     dict(x='Surprised😲', y=emotion_counts['Surprise'], text='5', showarrow=False, font=dict(size=15)),
#     dict(x='Fear😎', y=emotion_counts['Fear'], text='2', showarrow=False, font=dict(size=15)),
# ]

fig_binary_bar.update_layout()

# Display the plots
st.plotly_chart(fig_line, use_container_width=True)
st.plotly_chart(fig_binary_bar, use_container_width=True)
