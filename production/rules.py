import streamlit as st
import os
import base64

def save_details(user_type, job_description, technology, experience, cgpa_filteration, instructions):
    with open("production/rules.txt", "w") as f:
        f.write(f"User Type: {user_type}\n")
        f.write(f"Job Description: {job_description}\n")
        f.write(f"Technology: {technology}\n")
        if user_type == "Graduate":
            f.write(f"CGPA Filteration: {cgpa_filteration}\n")
        elif user_type == "Employee":
            f.write(f"Experience in Industry: {experience}\n")
        f.write(f"Instructions: {instructions}\n")

def main():
    st.title("Job Details Entry")

    user_type = st.radio("Select User Type:", ("Graduate", "Employee"))

    job_description = st.text_input("Enter Job Description:")
    technology = st.text_input("Enter Technology:")
    
    if user_type == "Graduate":
        cgpa_filteration = st.text_input("Enter CGPA Filteration:")
        experience = ""  # No experience required for graduates
    elif user_type == "Employee":
        experience = st.text_input("Enter Experience in Industry:")
        cgpa_filteration = ""  # No CGPA required for employees
    
    instructions = st.text_input("Add extra instructions:")

    if st.button("Save Details"):
        save_details(user_type, job_description, technology, experience, cgpa_filteration, instructions)
        st.success("Details saved successfully!")

    st.subheader("Saved Details:")
    try:
        with open("production/rules.txt", "r") as f:
            details = f.read()
        st.text(details)
    except FileNotFoundError:
        st.text("No details saved yet.")
    
    # if st.button("Analysis"):
    #     # Add your analysis code here
    #     st.write("Placeholder for analysis")
        
    # Check if transcript.pdf exists and generate a download button for it
    # if os.path.exists("D:/abhijith/ML/pravaah/chat_transcript.pdf"):
    #     with open("D:/abhijith/ML/pravaah/chat_transcript.pdf", "rb") as f:
    #         pdf_bytes = f.read()
    #     st.markdown(
    #         f'<a href="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" download="chat_transcript.pdf">Download chat_transcript.pdf</a>',
    #         unsafe_allow_html=True
    #     )

if __name__ == "__main__":
    main()