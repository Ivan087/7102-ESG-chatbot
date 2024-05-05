from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

import re
# Set environment variables for API keys
# pinecone_api_key = os.environ.get('PINECONE_API_KEY')

index_name = "esg-chatbot-index"
# embeddings = HuggingFaceEmbeddings()

# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

path = './Esg_data'
# os.chdir(path)
# for file in os.listdir(path):
#     if file.endswith('txt'):
#         file = os.path.join(path,file)
#         loader = TextLoader(file, encoding='utf-8')
#         data = loader.load()
#         ## TODO: specific data preprocess of txt file.
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
#         texts = text_splitter.split_documents(data)
#         ## TODO: [preprocess(text.page_content) for text in texts]
#     else:
#         file = os.path.join(path, file)
#         loader = PyPDFLoader(file)
#         data = loader.load()
#         ## TODO: specific data preprocess of pdf file
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         texts = text_splitter.split_documents(data)
# loader = TextLoader('./Esg_data/0 Governance An Imperative Element of ESG.txt', encoding='utf-8')
loader = PyPDFLoader('./Esg_data/Are ESG Improvements Recognized Perspectives from the Public Sentiments.pdf')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
texts = text_splitter.split_documents(data)
## TODO: preprocess for pdf: remove of references, footers and headers, url, copyright, email,
def txt_preprocess(texts):
    for i, text in enumerate(texts):
        text.page_content = ''
def url_remover(text):
    text = re.sub(r'https\S', '', text)
    text = re.sub(r"\S*https?:\S*", "", text)
    return text
def common_remover(strings):
    common_prefix = ""
    for chars in zip(*strings):
        if len(set(chars)) == 1:
            common_prefix += chars[0]
        else:
            break
    common_suffix = ""
    for chars in zip(*strings[::-1]):
        if len(set(chars)) == 1:
            common_suffix += chars[0]
        else:
            break
    return [string[len(common_prefix):-len(common_suffix)] for string in strings]


print(texts)

# vectorstore.add_documents([texts[0]],namespace = 'txt')

query = "Tell me more about Ketanji Brown Jackson."
# print(vectorstore.similarity_search(query))
# print(vectorstore.similarity_search(query, namespace='txt'))





