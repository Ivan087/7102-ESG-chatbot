import time

from pinecone import Pinecone, ServerlessSpec, PodSpec

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
pc.create_index(
    name=PINECONE_INDEX_NAME, dimension=1536, metric="cosine",
    # spec=ServerlessSpec(
    #     cloud="aws",
    #     region="us-west-2"
    # )
    spec=PodSpec(
    environment="gcp-starter"
  )
)

# pc.describe_index("esg-chatbot-index")
index = pc.Index("esg-chatbot-index")
index.describe_index_stats()

while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
    print(".", end="")
    time.sleep(1)
print("success")

