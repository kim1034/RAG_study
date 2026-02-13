import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 문서 로드 (기존과 동일)
file_path = "study_data.pdf"
loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
documents = loader.load()

# 2. 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# 3. 임베딩 및 벡터 저장소
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-miniLm-l6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 4. LLM 설정 (본인의 API 키를 입력하세요)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
llm = ChatOpenAI(model="gpt-4o")

# 5. RAG 전용 프롬프트 설정
template = """가져온 문맥(context)을 사용하여 질문에 답하세요. 
답을 모른다면 모른다고 말하고 추측하지 마세요. 
최대한 친절하게 한글로 답변하세요.

#Context:
{context}

#Question:
{question}

#Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 6. RAG 체인 생성 (LCEL 방식 - chains 모듈 불필요)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 실행
print("\n" + "="*50)
query = "지원 내용에는 어떤 것들이 포함되어 있나요?"
response = rag_chain.invoke(query)
print(f"질문: {query}")
print(f"답변: {response}")
print("="*50)