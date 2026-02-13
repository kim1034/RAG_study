import streamlit as st

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 설정
load_dotenv()
st.set_page_config(page_title="나의 AI 문서 비서", page_icon="🤖")
st.title("📄 무엇이든 물어보세요 (RAG)")

# 2. 로직 (데이터 로드 및 체인 생성)
@st.cache_resource # 앱이 새로고침되어도 데이터를 유지하게 해주는 고마운 기능
def setup_rag():
    loader = PyPDFLoader("study_data.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-miniLm-l6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    
    prompt = ChatPromptTemplate.from_template("""
    문맥을 바탕으로 질문에 답하세요:
    {context}
    
    질문: {input}
    """)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# 3. 화면 구성
chain = setup_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 채팅 입력창
if prompt_input := st.chat_input("PDF 내용에 대해 궁금한 점을 적어주세요"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        response = chain.invoke(prompt_input)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})