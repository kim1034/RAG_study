import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 1. 문서 로드 경로 설정
# PDF를 사용한다면 PyPDFLoader, 텍스트 파일이면 TextLoader를 사용합니다.
file_path = "study_data.pdf" 

if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.endswith(".txt"):
    loader = TextLoader(file_path)

# 2. 문서 읽기
documents = loader.load()

# 3. 결과 확인 (첫 페이지 내용만 살짝 출력)
print(f"문서를 성공적으로 불러왔습니다. 총 {len(documents)}페이지가 로드되었습니다.")
print("--- 첫 페이지 내용 미리보기 ---")
print(documents[0].page_content[:200]) # 앞 200자만 출력

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 분할기 설정 (한 청크당 500자, 50자씩 겹치게)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 2. 문서 분할 실행
splits = text_splitter.split_documents(documents)

# 3. 결과 확인
print(f"분할 완료! 전체 문서를 {len(splits)}개의 청크로 나누었습니다.")
print("-" * 30)
print(f"첫 번째 청크 내용:\n{splits[0].page_content}")


### 4단계 ###

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 임베딩 모델 설정 (무료 오픈소스 모델 사용)
# 'sentence-transformers/all-MiniLM-L6-v2'는 작고 빠르며 성능이 준수합니다.
model_name = "sentence-transformers/all-miniLm-l6-v2"
model_kwargs = {'device': 'cpu'} # GPU가 있다면 'cuda'로 변경 가능
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 2. 벡터 저장소 생성 및 데이터 저장
# splits: 이전 단계에서 쪼갠 텍스트 뭉치
# embeddings: 방금 설정한 숫자로 바꾸는 도구
# persist_directory: 데이터를 저장할 경로 (폴더가 생성됩니다)
db = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    persist_directory="./chroma_db" 
)

print("벡터 저장소 구축이 완료되었습니다!")

# 3. 간단한 검색 테스트
query = "문서에서 가장 중요한 내용은?" # 문서에 있을 법한 질문을 던져보세요
docs = db.similarity_search(query)

print("\n--- 검색 결과 (가장 유사한 문서 조각) ---")
print(docs[0].page_content)