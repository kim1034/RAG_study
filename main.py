import os
from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
# 1. .env 파일 로드 시도 및 결과 확인
if not load_dotenv():
    print("경고: .env 파일을 찾을 수 없습니다. 경로를 확인하세요.")

# 2. 키 가져오기
google_api_key = os.getenv("GOOGLE_API_KEY")

# 3. 키가 제대로 로드되었는지 체크
if google_api_key is None:
    raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 환경변수에 할당 (이제 문자열임이 보장됨)
os.environ["GOOGLE_API_KEY"] = google_api_key

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI # 변경된 부분
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 문서 로드 및 분할 (기존과 동일)
file_path = "study_data.pdf"
loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 2. 벡터 저장소 (기존과 동일)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-miniLm-l6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3. Gemini 설정 (여기에 발급받은 키를 넣으세요)
os.environ["GOOGLE_API_KEY"] = google_api_key  # 본인의 Google API 키로 변경
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.7)  # 모델과 온도 설정  

# 4. RAG 체인 구성
prompt = ChatPromptTemplate.from_template("""
주어진 문맥(context)을 바탕으로 질문에 답하세요.
답을 모른다면 모른다고 하고, 문맥에 없는 내용은 지어내지 마세요.

문맥: {context}
질문: {question}
답변:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 실행
print("\n" + "="*50)
query = "지원 내용에는 어떤 것들이 포함되어 있나요?"
response = rag_chain.invoke(query)
print(f"질문: {query}")
print(f"제미나이 답변: {response}")
print("="*50)