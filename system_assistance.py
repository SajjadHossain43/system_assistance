from typing import List
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

#load_dotenv()

app = FastAPI()

def auto_config():
    os.environ["API_KEY"] = "2EqAWpytfgNV4iFEvQ1suZUdAQx142L346bnKLpcGmoyv1tfJOSWJQQJ99BCACHrzpqXJ3w3AAAAACOG3yRc"
    os.environ["GPT_API_VERSION"] = "2025-01-01-preview" 
    os.environ["EMBED_API_VERSION"] = "2023-05-15"
    os.environ["GPT_MODEL"] ="gpt-35-turbo-16k"
    os.environ["EMBED_MODEL"] ="text-embedding-3-large"
    os.environ["OPENAI_ENDPOINT"] ="https://hight-m87lalwz-northcentralus.cognitiveservices.azure.com/"
    
#def get_gpt_instance():
    #return AzureChatOpenAI(azure_deployment=os.getenv('GPT_MODEL'), api_key=os.getenv('API_KEY'), api_version=os.getenv('GPT_API_VERSION'), azure_endpoint=os.getenv('OPENAI_ENDPOINT'))
def get_gpt_instance():
    return AzureChatOpenAI(azure_deployment="gpt-35-turbo-16k", api_key="2EqAWpytfgNV4iFEvQ1suZUdAQx142L346bnKLpcGmoyv1tfJOSWJQQJ99BCACHrzpqXJ3w3AAAAACOG3yRc", api_version="2025-01-01-preview", azure_endpoint="https://hight-m87lalwz-northcentralus.cognitiveservices.azure.com/")

#def get_embedding_instance():
    #return AzureOpenAIEmbeddings(azure_deployment=os.getenv('EMBED_MODEL'), api_key=os.getenv('API_KEY'), api_version=os.getenv('EMBED_API_VERSION'), azure_endpoint=os.getenv('OPENAI_ENDPOINT'))
def get_embedding_instance():
    return AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-large", api_key="2EqAWpytfgNV4iFEvQ1suZUdAQx142L346bnKLpcGmoyv1tfJOSWJQQJ99BCACHrzpqXJ3w3AAAAACOG3yRc", api_version="2023-05-15", azure_endpoint="https://hight-m87lalwz-northcentralus.cognitiveservices.azure.com/")

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
    if os.path.exists(os.path.join(index_path, 'index.faiss')):
        os.remove(os.path.join(index_path, 'index.faiss'))
        os.remove(os.path.join(index_path, 'index.pkl'))
    #if not os.path.exists(os.path.join(pdf_path, 'index.faiss')):
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
    auto_config()
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
            return e.with_traceback()

@app.post('/upload_pdf')
async def save_file(file: UploadFile = File(...)):
    if not os.path.exists(os.path.join(os.getcwd(), 'pdf')): os.mkdir(f'{os.getcwd()}/pdf')
    if not os.path.exists(os.path.join(os.getcwd(), 'FAISS')): os.mkdir(f'{os.getcwd()}/FAISS')
    pdf_folder = os.path.join(os.getcwd(), 'pdf')
    existing_pdf_list = glob.glob(f'{pdf_folder}/*.pdf', recursive=True)
    pdf_names = [pdf_name for pdf_name in existing_pdf_list]
    if not file.filename in pdf_names:
        file_path = f"{os.path.join(os.getcwd(), 'pdf')}/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        generate_index(pdf_path=os.path.join(os.getcwd(), 'pdf'), index_path=os.path.join(os.getcwd(), 'FAISS'))
    else:
        return f'File named {file.filename} already exists.'
    return f"File saved as {file_path}"

@app.post('/upload_files')
async def save_files(files: List[UploadFile] = File(...)):
    if not os.path.exists(os.path.join(os.getcwd(), 'pdf')): os.mkdir(f'{os.getcwd()}/pdf')
    if not os.path.exists(os.path.join(os.getcwd(), 'FAISS')): os.mkdir(f'{os.getcwd()}/FAISS')
    pdf_folder = os.path.join(os.getcwd(), 'pdf')
    existing_pdf_list = glob.glob(f'{pdf_folder}/*.pdf', recursive=True)
    pdf_names = [pdf_name for pdf_name in existing_pdf_list]
    for pdf_file in files:
        if not pdf_file in pdf_names:
            file_path = f'{os.path.join(os.getcwd(), 'pdf')}/{pdf_file.filename}'
            with open(file_path, 'wb') as f:
                f.write(await pdf_file.read())
    return f"File saved in {pdf_folder}" if generate_index(pdf_path=os.path.join(os.getcwd(), 'pdf'), index_path=os.path.join(os.getcwd(), 'FAISS')) else 'Files could be saved!'
    
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
#if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 10000))
    #uvicorn.run(app, host="0.0.0.0", port=port)