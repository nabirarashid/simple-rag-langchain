# just playing around with vectors
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain.evaluation import load_evaluator
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

# loading environment variables from .env file
load_dotenv()

# setting OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# initializing the embedding function
embedding_function = OpenAIEmbeddings()

# generating the embedding for the query
vector = embedding_function.embed_query("apple")

# printing the length of the vector
# print(vector)
print(len(vector))

# more interesting to find distance between vectors
evaluator = load_evaluator("pairwise_embedding_distance")

# to run an evaluation
x = evaluator.evaluate_string_pairs(prediction="apple", prediction_b="orange")

# print the evaluation result
print(x)
