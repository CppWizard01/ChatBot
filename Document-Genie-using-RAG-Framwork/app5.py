import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import json

# Set up Streamlit page
st.set_page_config(page_title="Document Genie", layout="wide")

api_key = '' 

st.markdown("""## RAG CHATBOT """)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to get text from a web page
def get_web_page_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we notice bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from HTML
        text = soup.get_text(separator='\n')
        return text
    except requests.RequestException as e:
        st.error(f"Error fetching the web page: {e}")
        return ""

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and manage chat history
def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Prepare and save chat entry
    chat_entry = {
        "question": user_question,
        "response": response["output_text"]
    }
    
    # Update chat history in session state
    st.session_state['chat_history'].append(chat_entry)
    save_chat_history(st.session_state['chat_history'])
    
    # Display the response
    st.write("Reply: ", response["output_text"])

# Functions to handle chat history
def load_chat_history():
    """Load chat history from a JSON file."""
    if os.path.exists('chat_history.json'):
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    return []

def save_chat_history(history):
    """Save chat history to a JSON file."""
    with open('chat_history.json', 'w') as f:
        json.dump(history, f, indent=4)

# Function to clear chat history
def clear_chat_history():
    """Clear the chat history."""
    if os.path.exists('chat_history.json'):
        os.remove('chat_history.json')
    st.session_state['chat_history'] = []

# Main function
def main():
    st.header("Ask me Anything....")

    # Input for user question (Moved above chat history)
    user_question = st.text_input("Ask a Question from the Web Page Content", key="user_question")

    # Process user input after displaying chat history
    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)
        # Display the response of the latest question right after the input box
        latest_entry = st.session_state['chat_history'][-1]
        
        st.write("---")

    # Clear chat history button above the chat history
    if st.button("Clear Chat History"):
        clear_chat_history()
        st.success("Chat history cleared!")

    # Load existing chat history at the start
    if not st.session_state['chat_history']:
        st.session_state['chat_history'] = load_chat_history()

    # Display chat history (excluding the latest question and response)
    if st.session_state['chat_history'][:-1]:
        st.subheader("Chat History")
        for entry in reversed(st.session_state['chat_history'][:-1]):
            st.write(f"**Question:** {entry['question']}")
            st.write(f"**Response:** {entry['response']}")
            st.write("---")

    with st.sidebar:
        st.title("Menu:")
        url = st.text_input("Enter the URL of the Web Page", key="url_input")
        if st.button("Submit & Process", key="process_button") and url and api_key:  # Check if URL and API key are provided before processing
            with st.spinner("Processing..."):
                web_text = get_web_page_text(url)
                if web_text:
                    text_chunks = get_text_chunks(web_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Done")

if __name__ == "__main__":
    main()
