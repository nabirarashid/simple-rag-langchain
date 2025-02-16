from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

DATA_PATH = "data/books"
CHROMA_PATH = "chroma"

# loading key from .env file
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']


def load_documents():
    loader = DirectoryLoader(DATA_PATH, "*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True )
    
    chunks = text_splitter.split_documents(documents)
    print (f"Split {len(documents)} documents into {len(chunks)} chunks")

    # checking a random chunk
    document = chunks[18]
    print(document.page_content)
    print(document.metadata) # prints source + start_index

    return chunks

def save_to_chroma(chunks: list[Document]):
    # clearing out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # creating a new database from documents
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

    # using persist method to force save
    db.persist()

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

# combining all functions to generate data store

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# only runs when the script is run directly
if __name__ == "__main__":
    generate_data_store()