import streamlit as st
import plotly.graph_objects as go

# Title of the application
st.title("Resume Upload and Score Visualization")

# File uploader
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "doc", "docx"])

if uploaded_file is not None:
    # Display the file name
    st.write("File uploaded successfully:", uploaded_file.name)
    

    
    # Assuming some process to determine the score
    # For demonstration, let's use a sample score
    score = 75  # Sample score out of 100

    # Create a circular progress bar using Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Resume Score"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#4CAF50"},
               'steps': [
                   {'range': [0, 50], 'color': "#FF6347"},
                   {'range': [50, 100], 'color': "#90EE90"}]}))

    # Display the circular progress bar
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Please upload your resume to see the score.")