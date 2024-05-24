import streamlit as st
import plotly.graph_objects as go
import os 
from langchain_openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

# Title of the application
st.title("Resume Upload and Score Visualization")
technology_arr=[]

# File uploader
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "doc", "docx"])
rules_path=r"D:\abhijith\ML\pravaah\production\rules.txt"
with open(rules_path,"r") as f:
    for line in f:
        key, value = line.split(': ')
        value = value.strip()  
        
        if key == "Job Description":
            job_description = value
        elif key == "Technology":
            technology_arr.append(value)
        # elif key == "Experience in Industry":
        #     experience_in_industry = value
        # elif key == "CGPA Filteration":
        #     cgpa_filteration = value
        elif key == "Instructions":
            instructions = value


def save_pdf_to_vscode(uploaded_file):
    if uploaded_file is not None:
        # Save the uploaded PDF file to the Visual Studio Code workspace
        file_path = os.path.join("D:/abhijith/ML/pravaah", "resume.pdf")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Resume saved successfully in Visual Studio Code workspace!")
def llm():
    def load_pdf(file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    pdf_file_path = r'D:\abhijith\ML\pravaah\resume.pdf'
    documents = load_pdf(pdf_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = OpenAI()
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    tech=str("Technologies required :"+str(technology_arr)+" Job Descrption:"+job_description+" Other Instructions: "+instructions)

    query = f" MUST RETURN OUTPUT , SCORE:/100,As per {tech} check whether the employee eligible for next round,If resume doesnt have required informarion return 0 , MUST RETURN OUTPUT ONLY AND ONLY AS : Score : /100"
    docs = vectorstore.similarity_search(query)
    answer = qa_chain.run(input_documents=docs, question=query)
    print("ANSWER       :       ",answer)
    if(len(answer)>20):
        answer=55
    st.write(answer)

    return int(answer.split(':')[1].split("/")[0].strip())



if uploaded_file is not None:
    # Display the file name
    save_pdf_to_vscode(uploaded_file)
    st.write("File uploaded successfully:", uploaded_file.name)
    sc=llm()
    
    


    # Assuming some process to determine the score
    # For demonstration, let's use a sample score
    score = sc  # Sample score out of 100

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