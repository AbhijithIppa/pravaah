import streamlit as st
import os

def save_pdf_to_vscode(uploaded_file):
    if uploaded_file is not None:
        # Save the uploaded PDF file to the Visual Studio Code workspace
        file_path = os.path.join("D:/abhijith/ML/pravaah", "resume.pdf")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Resume saved successfully in Visual Studio Code workspace!")

def main():
    st.title("Upload Resume")

    # Add a file uploader for PDF files
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    # Button to save the uploaded file to Visual Studio Code
    if st.button("Save to Visual Studio Code"):
        save_pdf_to_vscode(uploaded_file)

if __name__ == "__main__":
    main()
