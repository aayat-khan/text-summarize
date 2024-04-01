import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

def main():
    st.title("Text Summarization with LangChain and Streamlit")

    # Sidebar
    st.sidebar.title("Options")
    task = st.sidebar.selectbox("Choose a task", ["Summarize Speech", "Summarize PDF"])

    if task == "Summarize Speech":
        summarize_speech()
    elif task == "Summarize PDF":
        summarize_pdf()

def summarize_speech():
    st.subheader("Summarize Speech")
    
    # Define the speech
    speech = st.text_area("Enter the speech", "")

    # Initialize the model
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')

    # Generate summary
    if st.button("Generate Summary"):
        chat_messages = [
            SystemMessage(content='You are an expert assistant with expertise in summarizing speeches'),
            HumanMessage(content=f'Please provide a short and concise summary of the following speech:\nTEXT: {speech}')
        ]
        summary = llm(chat_messages).content
        st.write(summary)

def summarize_pdf():
    st.subheader("Summarize PDF")
    
    # Upload PDF file
    pdf_file = st.file_uploader("Upload PDF file", type=['pdf'])

    if pdf_file is not None:
        # Read text from PDF
        text = ""
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Initialize the model
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.create_documents([text])

        # Summarize chunks
        chain = load_summarize_chain(llm, chain_type='map_reduce')
        summary = chain.run(chunks)

        # Display summary
        st.write(summary)

if __name__ == "__main__":
    main()
