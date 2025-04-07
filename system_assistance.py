from fastapi import FastAPI, File, UploadFile
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
import glob
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import json

import uvicorn
#import gradio as gr
"""OPENAI_ENDPOINT = https://hight-m87lalwz-northcentralus.cognitiveservices.azure.com/
API_KEY = 2EqAWpytfgNV4iFEvQ1suZUdAQx142L346bnKLpcGmoyv1tfJOSWJQQJ99BCACHrzpqXJ3w3AAAAACOG3yRc
GPT_API_VERSION = 2025-01-01-preview
EMBED_API_VERSION = 2023-05-15
GPT_MODEL = gpt-35-turbo-16k
EMBED_MODEL = text-embedding-3-large
"""
os.environ["API_KEY"] = "2EqAWpytfgNV4iFEvQ1suZUdAQx142L346bnKLpcGmoyv1tfJOSWJQQJ99BCACHrzpqXJ3w3AAAAACOG3yRc"
os.environ["GPT_API_VERSION"] = "2025-01-01-preview" 
os.environ["EMBED_API_VERSION"] = "2023-05-15"
os.environ["GPT_MODEL"] ="gpt-35-turbo-16k"
os.environ["EMBED_MODEL"] ="text-embedding-3-large"

#load_dotenv()

app = FastAPI()

def get_gpt_instance():
    return AzureChatOpenAI(azure_deployment=os.getenv('GPT_MODEL'), api_key=os.getenv('API_KEY'), api_version=os.getenv('GPT_API_VERSION'), azure_endpoint=os.getenv('OPENAI_ENDPOINT'))

def get_embedding_instance():
    return AzureOpenAIEmbeddings(azure_deployment=os.getenv('EMBED_MODEL'), api_key=os.getenv('API_KEY'), api_version=os.getenv('EMBED_API_VERSION'), azure_endpoint=os.getenv('OPENAI_ENDPOINT'))

def get_text_from_pdf(pdf_path: os.path) -> str:
    texts = ''
    if not os.path.exists(pdf_path): return texts
    pdfs = glob.glob(os.path.join(pdf_path, '*.pdf'))
    if not pdfs: return texts
    for pdf in pdfs:
        with open(pdf, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if len(text) > 0: texts += text
    return texts

def get_chunk_from_texts(texts: str):
    return RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(texts)

def generate_index(pdf_path: os.path, index_path: os.path):
    if not os.path.exists(os.path.join(pdf_path, 'index.faiss')):
        texts = get_text_from_pdf(pdf_path)
        if len(texts) > 0:
            text_chunks = get_chunk_from_texts(texts)
            embedding = get_embedding_instance()
            vectors = FAISS.from_texts(text_chunks, embedding=embedding)
            vectors.save_local(index_path)
    return True

def get_conversation_chain():
    prompt_template = """
     This is a Q&A prompt for TribaLex, a court solution system. Respond concisely. Try to talk in more human-like form. 
     - If you do not find the answer in the context, say  you don't know politely . 
     - Do not generate any information outside of the context.
     - If you're unsure or the context does not have enough information, simply say you are sorry, you don't have enough information to answer that, in a polite manner.
     - Elaborate on the answer with proper information.
     Context: \n {context}? \n
     Question: \n {question}? \n
     Answer:
     """""
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    llm = get_gpt_instance()
    llm.model_rebuild()
    return load_qa_chain(llm=llm, chain_type='stuff', prompt=prompt)

def process_input(question: str, index_path: os.path) -> str:
    pdf_content = FAISS.load_local(index_path, embeddings=get_embedding_instance(), allow_dangerous_deserialization=True)
    if len(question) > 0:
        model_input = pdf_content.similarity_search(question)
    if model_input:
        try:
            chain = get_conversation_chain()
            output = chain({'input_documents': model_input, 'question': question}, return_only_outputs=True)['output_text']
            #return json.dumps({'answer': output})
            return output
        except Exception as e:
            #return json.dumps({'answer': 'Sorry, I am not able to answer your question due to some problem'})
            return e.with_traceback()

@app.post('/upload_pdf')
async def save_file(file: UploadFile = File(...)):
    if not os.path.exists(os.path.join(os.getcwd(), 'pdf')): os.mkdir(f'{os.getcwd()}/pdf')
    if not os.path.exists(os.path.join(os.getcwd(), 'FAISS')): os.mkdir(f'{os.getcwd()}/FAISS')
    file_path = f"{os.path.join(os.getcwd(), 'pdf')}/1.pdf"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    generate_index(pdf_path=os.path.join(os.getcwd(), 'pdf'), index_path=os.path.join(os.getcwd(), 'FAISS'))
    return f"File saved as {file_path}"
    
@app.get('/ask')
async def get_system_response(question: str) -> str:
    return process_input(question, f'{os.getcwd()}/FAISS')

#iface = gr.Interface(
    #fn=get_system_response,
    #inputs=gr.Textbox(placeholder="Type your question here..."),
    #outputs="text",
    #title="Chatbot",
    #description="Enter your question and get a response."
#)

#pdf_uploader = gr.Interface(
    #fn=save_file,
    #inputs=gr.File(type="binary"),
    #outputs="text",
    #title="PDF Uploader",
    #description="Upload a PDF file."
#)

#demo = gr.TabbedInterface([iface, pdf_uploader], ["Chatbot", "PDF Upload"])

#demo.launch()
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)