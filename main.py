from operator import itemgetter

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os




load_dotenv()
llm = ChatOpenAI()

def format_doc(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

def main():
    embed = OpenAIEmbeddings()
    vectoreStore = PineconeVectorStore(embedding=embed, index_name=os.environ.get('INDEX_NAME'))
    retriver = vectoreStore.as_retriever(kwargs={'k':3})
    prompt = PromptTemplate.from_template(""" 
    Answer the question based only on the following context:
    {context}
    Question: {question}
    Provide a detailed answer
    """)

    chain = (RunnablePassthrough.assign(context=itemgetter('question') | retriver | format_doc)) | prompt | llm | StrOutputParser()
    response = chain.invoke({'question': 'What is Pineconce?'})
    print(response)


if __name__ == '__main__':
    main()