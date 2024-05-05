import os
import gradio as gr
from langchain_community.vectorstores import Pinecone
from langchain_core.messages import HumanMessage, SystemMessage, FunctionMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from question_judege import judge
BASE_URL = "https://aihubmix.com/v1/"
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# chat = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     openai_api_base=BASE_URL,
#     openai_api_key=OPENAI_API_KEY
# )

def ask_pdf(query, temperature=0.7,model="gpt-3.5-turbo",K=3,APIKEY=OPENAI_API_KEY):
    chat = ChatOpenAI(
        model=model,
        openai_api_base=BASE_URL,
        openai_api_key=APIKEY
    )
    messages = [
        SystemMessage(
            content="You are an ESG consultant, answering only ESG questions, directing users to ESG related questions for non-ESG questions, and your answers are based on the input documents."
        ),
    ]

    embeddings = OpenAIEmbeddings(openai_api_base=BASE_URL, openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    docsearch = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    docs = docsearch.similarity_search(query, k=K, filter=None, namespace=None)
    messages += [FunctionMessage(name="document", content=doc.page_content) for doc in docs]
    messages.append(HumanMessage(content=query))
    answer = chat.invoke(messages, temperature=temperature)
    return answer.content

def chat_model(query, temperature=0.7,model="gpt-3.5-turbo",K=3,APIKEY=OPENAI_API_KEY):
    # if judge([query]):
        return ask_pdf(query, temperature,model,K,APIKEY)
    # else:
    #     return "I am just an ESG consulting assistant and can only answer ESG related questions. Please ask questions about ESG."
if __name__ == "__main__":
    iface = gr.Interface(
        fn=chat_model,
        inputs=[
            gr.Textbox(lines=2, placeholder="Type your question here...",label="QUESTION"),
            gr.Slider(minimum=0, maximum=1, value=0.7, label="Temperature"),
            gr.Dropdown(
                choices=["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4","gemini-pro","gpt-4-1106-preview","gpt-4-32k","gpt-3.5-turbo-16k"],
                label="Select one model...",
                value="gpt-3.5-turbo"
            ),
            gr.Slider(minimum=0, maximum=10, value=3,label="K",step=1),
            gr.Textbox(label="APIKEY",type="password")
        ],
        outputs=gr.Textbox(),
        title="Chat with ESG ChatBot"
    )

    iface.launch()




