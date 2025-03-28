from huggingface_hub import hf_hub_download
import os
import streamlit as st
import pandas as pd
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
import nest_asyncio

# Apply nest_asyncio for async handling
nest_asyncio.apply()

# Enable CUDA for PyTorch if available
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda"
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA not available, using CPU.")

# Streamlit Page Config
st.set_page_config(page_title="Frontlett COVID-19 Chatbot", page_icon="🤖")

# Load Logo
logo_path = "logo.png"
st.image(logo_path, width=200)

st.title("COVID-19 Customer Support Chatbot 🤖")
st.write("Ask me any question about COVID-19!")

# Load FAQ Data
@st.cache_data
def load_faq_data():
    file_path = "COVID19_FAQ.csv"
    df = pd.read_csv(file_path)
    df["text"] = df["questions"] + " " + df["answers"]
    return df

df = load_faq_data()

# Create Vector Store
def create_vector_store(data):
    loader = DataFrameLoader(data, page_content_column="text")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

vectorstore = create_vector_store(df)

# Download Model from Hugging Face
MODEL_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"  # Change to the correct model repo 
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

# Load Llama Model
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=1024,
    n_batch=32,
    verbose=True
)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# User Input
user_input = st.text_input("Your question:")

if st.button("Ask"):
    if user_input:
        with st.spinner("Thinking... 🤔"):
            response = qa_chain.run(user_input)
        st.write(f"🤖 **Bot:** {response}")
