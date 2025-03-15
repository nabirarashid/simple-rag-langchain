from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import argparse

CHROMA_PATH = "chroma"

# loading key from .env file
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

PROMPT_TEMPLATE = """
    Answer the question based on only the following context:

    {context}

    ---
    
    Answer the question based on only the following context: {query}"""

def main(): 
    # creating a command line interface using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # preparing the database
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # searching  the database; best 3 results
    results = db.similarity_search_with_relevance_scores(query_text,k=3)

    # checking some conditions before returning results
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return
    
    # results will be a list of tuples with document and relevance score
    # e.g. [(document1, score1), (document2, score2), ...]

    # printing results using list comprehension; iterating over each tuple
    # extracting page_content from document object & joining 3 chunks with a separator
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # print(context_text)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query_text)
    print(prompt)

    # now getting openai to answer the prompt
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    # to provide sources along with response; access the metadata from document objects
    sources = [doc.meta_data.get("source", None) for doc, _score in results]
    formatted_sources = f"Response:{response_text}\nSources:{sources}"
    print(formatted_sources)

if __name__ == "__main__":
    main()


# Loads the Chroma vector store from CHROMA_PATH
# Uses the OpenAIEmbeddings to perform similarity searches on the vector store
# Prompts the user for a query text