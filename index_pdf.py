import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PC

from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME


os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY


PINECONE_INDEX_NAME = 'esg-chatbot-index'

folder_path = 'Esg_data'

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Preprocessing functions
def remove_non_printable(text):
    """Remove non-printable characters from the text."""
    printable = set(chr(i) for i in range(32, 127))
    return ''.join(filter(lambda x: x in printable, text))

def normalize_whitespace(text):
    """Normalize whitespace in the text, reducing multiple spaces to one and stripping trailing spaces."""
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_text(text):
    """Apply all preprocessing steps to the text."""
    text = remove_non_printable(text)
    text = normalize_whitespace(text)
    return text.lower()

def load_text(file_path):
    """Load text content from a file depending on its type."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        content = loader.load()
        return [preprocess_text(page.page_content) for page in content]
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return [preprocess_text(content)]
    return []

all_files = os.listdir(folder_path)
documents = []

for file_name in all_files:
    file_path = os.path.join(folder_path, file_name)
    documents.extend(load_text(file_path))

documents = [Document(text) for text in documents]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)


split_texts = []
for doc in documents:
    result = text_splitter.split_documents([doc])
    split_texts.extend(result)


embeddings = OpenAIEmbeddings(openai_api_base="https://aihubmix.com/v1/", openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")


docsearch = PC.from_texts(
    [text[0] for sublist in split_texts for text in sublist],
    embeddings.embed_query,
    index_name=PINECONE_INDEX_NAME
)
print("Successfully preprocessed, loaded all files, and updated the PINECONE database.")
