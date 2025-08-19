#step 1
# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("roblexnana/alice-wonderland-dataset")
#
# print("Path to dataset files:", path)

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

#to load the env variables inside this program
load_dotenv(verbose=True)

# Define the path to your data directory and the persistent directory for the vector store
DATA_PATH = "alice_in_wonderland.txt"
DB_PATH = "chroma_db"

def load_data_file(file_path):
    # Use TextLoader for .txt files
    loader = TextLoader(file_path, encoding='utf-8')
    #this line creates document object
    docs = loader.load()
    print("printing the docs below")
    print(type(docs[0]))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Number of documents loaded: {len(docs)}")
    print(f"Number of chunks created: {len(chunks)}")
    return chunks

def get_answer_from_rag(query, chunks):
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")  #hit quota limit for this
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #initializing the vector store with the embedded data from chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        collection_name="alice_collection",
        embedding=embeddings,
        persist_directory=DB_PATH,  # Where to save data locally, remove if not necessary
    )

    # Configure the retriever - you define how should the retriever should return you the results. Here
    #the retriever will return the best top 3 answers

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create a RetrievalQA chain -> here the RetrievalQA is a wrapper that does things below
    #1. it takes the user query and then asks the retriver object to return the answer
    #2. it then stuffs those answer to the llm context space as the chain type is stuff and returns the response
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # Invoke the chain with the user query
    response = qa_chain.invoke({"query": query})
    return response['result']


returned_chunks =  load_data_file(DATA_PATH)
user_query = "Why did Alice went down the rabbit hole?"
answer = get_answer_from_rag(user_query, returned_chunks)
print(f"Question: {user_query}")
print(f"Answer: {answer}")