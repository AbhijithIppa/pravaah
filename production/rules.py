import streamlit as st
import os 
import base64  # Import the base64 module

def save_details(job_description, technology, experience, cgpa_filteration):
    with open("production/rules.txt", "w") as f:
        f.write(f"Job Description: {job_description}\n")
        f.write(f"Technology: {technology}\n")
        f.write(f"Experience in Industry: {experience}\n")
        f.write(f"CGPA Filteration: {cgpa_filteration}\n")

def main():
    st.title("Job Details Entry")

    job_description = st.text_input("Enter Job Description:")
    technology = st.text_input("Enter Technology:")
    experience = st.text_input("Enter Experience in Industry:")
    cgpa_filteration = st.text_input("Enter CGPA Filteration:")

    if st.button("Save Details"):
        save_details(job_description, technology, experience, cgpa_filteration)
        st.success("Details saved successfully!")

    st.subheader("Saved Details:")
    try:
        with open("production/rules.txt", "r") as f:
            details = f.read()
        st.text(details)
    except FileNotFoundError:
        st.text("No details saved yet.")
    
    if st.button("Analysis"):
        # Add your analysis code here
        st.write("Placeholder for analysis")
        
     # Check if transcript.pdf exists and generate a download button for it
    if os.path.exists("D:/abhijith/ML/pravaah/chat_transcript.pdf"):
        with open("D:/abhijith/ML/pravaah/chat_transcript.pdf", "rb") as f:
            pdf_bytes = f.read()
        st.markdown(
            f'<a href="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" download="chat_transcript.pdf">Download chat_transcript.pdf</a>',
            unsafe_allow_html=True
        )



if __name__ == "__main__":
    main()
