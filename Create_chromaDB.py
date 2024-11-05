import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import openai
import shutil
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = "chroma"
openai.api_key=os.environ['OPENAI_API_KEY']

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap  = 100,
        length_function = len,
    )
    
    chunks = text_splitter.split_documents(documents)
    chunks = chunks[-10:]
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), CHROMA_PATH)
    db.persist()
    print(f"Save {len(chunks)} chunks into {CHROMA_PATH}")

def main():
    loader = PyPDFLoader("human-nutrition-text.pdf")
    documents = loader.load()
    chunks = split_text(documents)
    save_to_chroma(chunks)


if __name__ == "__name__":
    main()






