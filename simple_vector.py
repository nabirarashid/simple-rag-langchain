# just playing around with vectors
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

embedding_function = OpenAIEmbeddings()
vector = embedding_function.embed_query("apple")

print(vector)