from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def main():
    pdf_file_path = 'traditional_resume.pdf'
    documents = load_pdf(pdf_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    llm = OpenAI()
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    
    query = "Your question about the PDF content"
    docs = vectorstore.similarity_search(query)
    answer = qa_chain.run(input_documents=docs, question=query)
    
    print("Answer:", answer)

if __name__ == "__main__":
    main()
