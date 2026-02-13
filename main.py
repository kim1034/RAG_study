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