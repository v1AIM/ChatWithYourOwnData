import streamlit as st
import os
import csv
import docx
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader





# Extract text from different document formats
def extract_text_from_documents(docs):
    text = ""
    
    for doc in docs:
        file_extension = os.path.splitext(doc.name)[1].lower()  # Use doc.name to get the filename
        
        if file_extension == '.pdf':
            # Reset the pointer to the start of the file
            doc.seek(0)
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        elif file_extension == '.csv':
            doc.seek(0)
            reader = csv.reader(doc.read().decode('utf-8').splitlines())
            for row in reader:
                text += ' '.join(row) + '\n'
        
        elif file_extension == '.txt':
            doc.seek(0)
            text += doc.read().decode('utf-8') + '\n'
        
        
        elif file_extension == '.xlsx':
            doc.seek(0)
            excel_data = pd.read_excel(doc)
            text += excel_data.to_string(index=False) + '\n'
        
        elif file_extension == '.docx':
            doc.seek(0)
            docx_reader = docx.Document(doc)
            for paragraph in docx_reader.paragraphs:
                text += paragraph.text + '\n'
        
        else:
            text += f"Unsupported file format: {file_extension}\n"
    
    return text


# Split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vector store
def get_vector_store(text_chunks):
    embedding = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vectorstore


# Create a conversational chain
def get_conversational_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        temperature=0,
    )

    chat_model = ChatHuggingFace(llm=llm)

    condense_question_template = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    condense_question_prompt = ChatPromptTemplate.from_template(condense_question_template)

    qa_template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer
    the question. If you don't know the answer, say that you
    don't know. Use five sentences maximum and keep the answer concise.

    Chat History:
    {chat_history}

    Other context:
    {context}

    Question: {question}
    """

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    qa_prompt = ChatPromptTemplate.from_template(qa_template)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        chat_model,
        vectorstore.as_retriever(),
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={
            "prompt": qa_prompt,
        },
        memory=memory,
    )

    return conversation_chain


# Handle user input
def handle_user_input(user_question):
    # Pass the existing chat history along with the new question
    response = st.session_state.conversation({
        "question": user_question, 
        "chat_history": st.session_state.chat_history
    })
    # st.write(response)

    # Update the chat history with the new messages
    st.session_state.chat_history.append((user_question, response["answer"]))


    # Display the conversation
    for user_msg, assistant_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(assistant_msg)



def main():
    load_dotenv()
    st.set_page_config(page_title="ConversationalRetrieval" )

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chat with different document format 🦜🔗 📚")


    # Sidebar
    with st.sidebar:
        st.subheader("Your Documents")

        # Upload the documents
        docs = st.file_uploader("Upload your document here and click on 'Process'", accept_multiple_files=True, type=["pdf" , "docx" , "txt", "csv"])
        if st.button("Process"):
            with st.spinner("Processing documents..."):
                # time.sleep(1)
            # get the PDFs text
                raw_text = extract_text_from_documents(docs)
                # st.write(raw_text)

            # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

            # create vector embeddings and store them in a database
                vectorstore = get_vector_store(text_chunks)
                # st.write(vectorstore)

            # create ConverstionalChain
                st.session_state.conversation = get_conversational_chain(vectorstore)


            st.success("Done processing Documents")

    # Main content
    user_question = st.chat_input("Ask question about your documents")
    if user_question:
        handle_user_input(user_question)



if __name__ == '__main__':
    main()