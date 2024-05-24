import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

import time
import numpy as np
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2,math
import numpy as np
import mediapipe as mp
# Set the page configuration
st.set_page_config(page_title="Creative Dashboard", layout="wide")

# Add centered page heading
st.markdown("<h1 style='text-align: center;'>Candidate Interview Analysis</h1>", unsafe_allow_html=True)

with open(r"D:\abhijith\ML\pravaah\client\answer.txt","r") as f:
    score1=f.readline()
# Define and display two circular progress bars side by side
score1 = int(score1)
# score2 = 72  
emotion_counts={}
face_classifier = cv2.CascadeClassifier(r'D:\abhijith\ML\pravaah\models\emotion\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\abhijith\ML\pravaah\models\emotion\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

video_path = r"D:\abhijith\ML\pravaah\video.avi"  # Change this to the path of your video file
cap = cv2.VideoCapture(0)

emotion_counts = {label: 0 for label in emotion_labels}  # Initialize counts to zero

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5,refine_face_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [33] #right eye right most landmark
L_H_RIGHT = [133] #right eye leftmost landmark
R_H_LEFT = [362]  #left eye rightmost landmark
R_H_RIGHT = [263] #left eye leftmost landmark

cnt=0
tc=0
pTime=0
obs=0
h=0
bp=True


def euclidean_distance(point1, point2):
    x1, y1 =point1.ravel()
    x2, y2 =point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance


def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position =""
    if ratio >= 2.95:
        iris_position="left"

    elif ratio > 2.77 and ratio <= 2.95:
        iris_position="center"

    else:
        iris_position = "right"
    

    return iris_position, ratio
# import time
# last_active_time = time.time()

# # Main loop for processing video frames
# while True:
#     current_time = time.time()
#     # Check if bp is false and 5 seconds have passed since the last activity
#     if not bp and current_time - last_active_time > 5:
#         print("The interview has ended. Thank you for your participation.")
#         # Add code here to turn off the video or handle the end of the interview
#         break  # Exit the loop

#     # Your existing code for video processing goes here
    
#     # Update the last active time whenever there's activity
#     last_active_time = current_time

if 'running' not in st.session_state:
    st.session_state.running = False
if 'deviations' not in st.session_state:
    st.session_state.deviations = [0]
if 'warnings' not in st.session_state:
    st.session_state.warnings = [0]

while True:
    current_time = time.time()
    last_active_time = time.time()

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
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]

    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.face_landmarks:
        mesh_points = np.array(
        [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.face_landmarks.landmark])

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)

        iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
    else:
        # Handle case when face landmarks are not detected
        iris_pos = None
        ratio = None

    if (iris_pos == "left" or iris_pos == "right"):
        tc += 1
        cnt += 1
    # /////////////////HANDS //////////////////////////////////
    if results.right_hand_landmarks or results.left_hand_landmarks:
        cv2.putText(image, f'Hands are detected', (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (121, 22, 76), 3)
        h += 1
    # //////////////////////////////////////////////////////////////////
    if (cnt > 200 and cnt < 250):
        cv2.putText(image, f'pls face towards screen', (20, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    if (cnt > 250):
        cnt = 0
        obs += 1
    st.session_state.deviations.append(tc)
    st.session_state.warnings.append(obs)
    # ///////////////////////////////////////POSTURE/////////////////////////////////////////////////////////////////////////
    if results.pose_landmarks:
        bp=True
        for num,pose in enumerate(results.pose_landmarks.landmark):
            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
            if(num==12 or num==13):
                if((pose.visibility)<0.15):
                    print(pose.visibility)
                    print("adjust camera to your shoulders level")
    
    else:
        print("Body is not present")
        bp=False
    cv2.imshow('Video', image)
    if not bp and current_time - last_active_time > 5:
        print("The interview has ended. Thank you for your participation.")
        # Add code here to turn off the video or handle the end of the interview
        break
    last_active_time = current_time



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(iris_pos,tc,obs,h,bp)

print("Emotion Counts:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")

cap.release()
cv2.destroyAllWindows()






fig_progress1 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score1,
    title={'text': "Resume Score"},
    gauge={'axis': {'range': [0, 100]},
           'bar': {'color': "#4CAF50", 'thickness': 0.2},
           'steps': [
               {'range': [0, 58], 'color': "#FF6347"},
               {'range': [58, 100], 'color': "#90EE90"}]}))

# fig_progress2 = go.Figure(go.Indicator(
#     mode="gauge+number",
#     value=score2,
#     title={'text': "AI Score"},
#     gauge={'axis': {'range': [0, 100]},
#            'bar': {'color': "#4CAF50", 'thickness': 0.2},
#            'steps': [
#                {'range': [0, 72], 'color': "#FF6347"},
#                {'range': [72, 100], 'color': "#90EE90"}]}))

# # Custom CSS for centering and responsiveness
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

st.plotly_chart(fig_progress1, use_container_width=True)


# Sample data for the increasing line plot with constant sections
def generate_line_chart_data():
    return pd.DataFrame({
        'deviations': st.session_state.deviations,
        'warnings': st.session_state.warnings
    })

# Generate the line chart data
data_line = generate_line_chart_data()

# Create the line chart with Plotly Express
fig_line = px.line(data_line, x='deviations', y='warnings', title='Iris Analysis')

# Update the layout to dynamically adjust the y-axis
fig_line.update_layout(
    xaxis_title='No of Deviations',
    yaxis_title='No of Warnings',
    yaxis=dict(range=[0, max(data_line['warnings']) + 5]),  # Dynamically adjust y-axis range
    xaxis=dict(range=[0, max(data_line['deviations']) + 1000])   # Dynamically adjust x-axis range
)

# Display the plot
st.plotly_chart(fig_line, use_container_width=True)

#tarun---------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# Sample emotion counts dictionary (replace with your actual data)
emotion_counts = {"Happy": emotion_counts["Happy"], "Sad": emotion_counts["Sad"], "Angry": emotion_counts["Angry"], "Surprised": emotion_counts["Surprise"], "Fearful": emotion_counts["Fear"]}


# Sample emotion counts dictionary (replace with your actual data)

# Convert emotion counts to DataFrame

import plotly.express as px
# emotion_counts = {"Happy": 40, "Sad": 25, "Angry": 10, "Surprised": 15, "Fearful": 5, "Neutral": 10}
data = {"Emotion": list(emotion_counts.keys()), "Count": list(emotion_counts.values())}    
df = pd.DataFrame(data)
st.title("Bar Chart - Emotion Distribution")
fig = px.bar(df, x='Emotion', y='Count', labels={'Count': 'Emotion Count'}, title="Distribution of Emotions")
fig.update_layout(width=1000, height=600)
st.plotly_chart(fig)

#tarun---------------------------------------------------------

#anshu-------------------------------------------------------

#anshu-------------------------------------------------------

#manoj---------------------------------------------------

#manoj---------------------------------------------------



#