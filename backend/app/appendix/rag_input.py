"""
RAG 입력 처리(테스트용 목업)

비개발자 요약:
- RAG 연동 없이도 동작 검증을 할 수 있도록 간단한 더미 입력 처리를 제공합니다.
"""

def process_input(input_text: str) -> str:
    """간단한 더미 입력 처리 함수(테스트용)"""
    return f"Mock processed input: {input_text[:50]}..."
