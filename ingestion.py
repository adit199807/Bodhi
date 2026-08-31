from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
# from transformers import AutoTokenizer
import os
load_dotenv()

def main():
    loader = UnstructuredLoader(file_path="resource/mediumblog1.txt", chunking_strategy='basic', max_characters=1000000)
    documents = loader.load()

    print('splitting........')
    # tokenizer = AutoTokenizer.from_pretrained('text-embedding-3-small' )
    splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)
    chunks = splitter.split_documents(documents)
    print('chunked..........')
    embedding = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(chunks, embedding, index_name=os.environ["INDEX_NAME"])
    print('uploaded')


if __name__ == '__main__':
    main()