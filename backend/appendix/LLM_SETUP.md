# LLM Integration Setup Guide

이 문서는 ATM 시스템의 LLM Integration Service 설정 및 배포 가이드입니다.

## 개요

LLM Integration Service는 내부 OpenAI-Compatible API와 외부 Ollama를 지원하는 통합 LLM 인터페이스를 제공합니다. 환경에 따라 자동으로 적절한 LLM 프로바이더로 전환됩니다.

## 지원되는 LLM 프로바이더

### 1. 내부 OpenAI-Compatible API (프로덕션)
- 회사 내부 LLM 서비스
- 엔터프라이즈급 인증 및 보안
- 사용량 추적 및 모니터링

### 2. Ollama (개발/테스트)
- 로컬 개발 환경용
- 외부 네트워크 없이 동작
- 무료 오픈소스 모델

### 3. 폴백 메커니즘
- 주 서비스 실패 시 자동 폴백
- 다중 클라이언트 지원
- 에러 복구 및 재시도

## 환경 변수 설정

### 프로덕션 환경 (내부 LLM)

```bash
# 환경 식별
ENVIRONMENT=production

# 내부 LLM API 설정
INTERNAL_LLM_BASE_URL=https://internal-llm-api.company.com
INTERNAL_LLM_API_KEY=your_api_key_here
INTERNAL_LLM_MODEL=gpt-4
INTERNAL_LLM_TICKET=your_department_ticket

# 사용자 정보
USER_ID=your_user_id
USER_TYPE=AD

# LLM 서비스 설정
LLM_TIMEOUT=60
LLM_MAX_RETRIES=3
LLM_TEMPERATURE=0.7
```

### 개발 환경 (Ollama)

```bash
# 환경 식별
ENVIRONMENT=development

# Ollama 설정
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# LLM 서비스 설정
LLM_TIMEOUT=60
LLM_MAX_RETRIES=3
LLM_TEMPERATURE=0.7
```

## 설치 및 구성

### 1. 프로덕션 환경 설정

#### 1.1 필요한 인증 정보 획득
```bash
# 내부 LLM API 키 발급 요청
# 부서별 티켓 발급
# 사용자 ID 확인
```

#### 1.2 환경 변수 설정
```bash
# Linux/macOS - ~/.bashrc 또는 ~/.zshrc에 추가
export ENVIRONMENT=production
export INTERNAL_LLM_BASE_URL=https://internal-llm-api.company.com
export INTERNAL_LLM_API_KEY=your_api_key_here
export INTERNAL_LLM_MODEL=gpt-4
export INTERNAL_LLM_TICKET=your_department_ticket
export USER_ID=your_user_id
export USER_TYPE=AD

# Windows - 시스템 환경변수 또는 .env 파일
set ENVIRONMENT=production
set INTERNAL_LLM_BASE_URL=https://internal-llm-api.company.com
set INTERNAL_LLM_API_KEY=your_api_key_here
# ... 기타 변수들
```

#### 1.3 연결 테스트
```python
# test_internal_llm.py
import asyncio
import os
from backend.appendix.internal_llm import get_agent_service

async def test_internal_llm():
    try:
        # 환경 변수 설정 확인
        print(f"Environment: {os.getenv('ENVIRONMENT')}")
        print(f"LLM Base URL: {os.getenv('INTERNAL_LLM_BASE_URL')}")
        
        # LLM 서비스 테스트
        agent_service = await get_agent_service()
        response = await agent_service.llm_manager.simple_completion(
            "Hello, this is a test message.",
            "You are a helpful assistant."
        )
        print(f"LLM Response: {response}")
        
        # 리소스 정리
        await agent_service.close()
        print("Internal LLM connection test successful!")
        
    except Exception as e:
        print(f"Internal LLM test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_internal_llm())
```

### 2. 개발 환경 설정 (Ollama)

#### 2.1 Ollama 설치
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# https://ollama.ai/download에서 설치 파일 다운로드
```

#### 2.2 모델 다운로드
```bash
# Llama3.1 모델 설치 (권장)
ollama pull llama3.1

# 다른 모델 옵션
ollama pull codellama
ollama pull mistral
ollama pull gemma

# 설치된 모델 확인
ollama list
```

#### 2.3 Ollama 서비스 시작
```bash
# Ollama 서비스 시작
ollama serve

# 다른 터미널에서 테스트
ollama run llama3.1
```

#### 2.4 환경 변수 설정
```bash
# 개발 환경 설정
export ENVIRONMENT=development
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1
```

#### 2.5 연결 테스트
```python
# test_ollama.py
import asyncio
import os
from backend.appendix.internal_llm import get_agent_service

async def test_ollama():
    try:
        # 환경 설정 확인
        print(f"Environment: {os.getenv('ENVIRONMENT')}")
        print(f"Ollama URL: {os.getenv('OLLAMA_BASE_URL')}")
        print(f"Model: {os.getenv('OLLAMA_MODEL')}")
        
        # LLM 서비스 테스트
        agent_service = await get_agent_service()
        response = await agent_service.llm_manager.simple_completion(
            "Hello, this is a test message.",
            "You are a helpful assistant."
        )
        print(f"LLM Response: {response}")
        
        # 리소스 정리
        await agent_service.close()
        print("Ollama connection test successful!")
        
    except Exception as e:
        print(f"Ollama test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
```

## 폴백 메커니즘 설정

### 자동 폴백 구성
LLM Manager는 다음 순서로 폴백을 시도합니다:

1. **주 서비스** (환경에 따라 내부 API 또는 Ollama)
2. **폴백 서비스** (내부 API 실패 시 Ollama로 폴백)
3. **에러 처리** (모든 서비스 실패 시 에러 로깅 및 예외 발생)

### 폴백 설정 커스터마이징
```python
# 커스텀 폴백 설정 예시
from backend.appendix.internal_llm import LLMManager, LLMConfig, LLMProvider

# 커스텀 설정으로 LLM 매니저 생성
primary_config = LLMConfig(
    provider=LLMProvider.INTERNAL,
    base_url="https://internal-api.company.com",
    api_key="your_api_key",
    model="gpt-4"
)

llm_manager = LLMManager(primary_config)
```

## 에이전트별 LLM 사용

### 문제 분석 에이전트
```python
from backend.appendix.internal_llm import get_agent_service

async def analyze_problem():
    agent_service = await get_agent_service()
    
    analysis = await agent_service.analyze_problem(
        "우리 팀의 월간 보고서 생성을 자동화하고 싶습니다."
    )
    
    return analysis
```

### 컨텍스트 수집 에이전트
```python
async def collect_context():
    agent_service = await get_agent_service()
    
    questions = await agent_service.collect_context_questions(
        problem_analysis,
        existing_context
    )
    
    return questions
```

### 요구사항 생성 에이전트
```python
async def generate_requirements():
    agent_service = await get_agent_service()
    
    requirements = await agent_service.generate_requirements(
        problem_analysis,
        context_data
    )
    
    return requirements
```

## 모니터링 및 로깅

### 로그 설정
```python
import logging

# LLM 서비스 로깅 설정
logging.getLogger('backend.appendix.internal_llm').setLevel(logging.INFO)

# 상세 디버그 로깅 (개발 시)
logging.getLogger('backend.appendix.internal_llm').setLevel(logging.DEBUG)
```

### 성능 모니터링
```python
import time
from backend.appendix.internal_llm import get_agent_service

async def monitor_llm_performance():
    agent_service = await get_agent_service()
    
    start_time = time.time()
    response = await agent_service.llm_manager.simple_completion(
        "Test prompt",
        "Test system message"
    )
    end_time = time.time()
    
    print(f"Response time: {end_time - start_time:.2f} seconds")
    print(f"Response length: {len(response)} characters")
```

### 사용량 추적
```python
# LLM 사용량 추적 예시
class LLMUsageTracker:
    def __init__(self):
        self.request_count = 0
        self.total_tokens = 0
        
    async def track_request(self, response):
        self.request_count += 1
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens += response.usage.get('total_tokens', 0)
```

## 트러블슈팅

### 일반적인 문제들

#### 1. 내부 API 연결 실패
```
Error: Connection failed to internal LLM API
```
**해결방법:**
- API 키가 올바른지 확인
- 부서 티켓이 유효한지 확인
- 네트워크 연결 상태 확인
- 방화벽 설정 확인

#### 2. Ollama 서비스 연결 실패
```
Error: Connection refused to localhost:11434
```
**해결방법:**
- Ollama 서비스가 실행 중인지 확인: `ollama serve`
- 포트 11434가 사용 가능한지 확인
- 모델이 다운로드되어 있는지 확인: `ollama list`

#### 3. 환경 변수 설정 오류
```
Error: Environment variable not found
```
**해결방법:**
- 환경 변수가 올바르게 설정되었는지 확인
- 새 터미널에서 환경 변수 다시 로드
- .env 파일 사용 시 파일 경로 확인

#### 4. 모델 응답 파싱 오류
```
Error: Failed to parse LLM response as JSON
```
**해결방법:**
- 자동 폴백 메커니즘이 동작함
- 로그에서 원본 응답 확인
- 프롬프트 엔지니어링 개선

### 디버그 모드 활성화
```python
import logging
import os

# 디버그 모드 설정
os.environ['LLM_DEBUG'] = 'true'
logging.basicConfig(level=logging.DEBUG)

# 상세 로그 확인
from backend.appendix.internal_llm import get_agent_service
```

## 보안 고려사항

### 1. API 키 보안
- API 키를 코드에 하드코딩하지 않기
- 환경 변수 또는 안전한 키 저장소 사용
- 정기적인 키 로테이션

### 2. 네트워크 보안
- HTTPS 연결 사용
- 방화벽 규칙 설정
- VPN 또는 전용선 사용

### 3. 데이터 보호
- 민감한 데이터 로깅 방지
- 요청/응답 데이터 암호화
- 데이터 보존 정책 준수

## 성능 최적화

### 1. 연결 풀링
- HTTP 클라이언트 재사용
- 연결 상태 모니터링
- 적절한 타임아웃 설정

### 2. 캐싱 전략
- 자주 사용되는 응답 캐싱
- 캐시 무효화 정책
- 메모리 사용량 모니터링

### 3. 배치 처리
- 여러 요청 배치로 처리
- 비동기 처리 활용
- 큐 시스템 도입

## 배포 체크리스트

### 프로덕션 배포 전 확인사항

- [ ] 모든 환경 변수 설정 완료
- [ ] 내부 API 연결 테스트 성공
- [ ] 폴백 메커니즘 테스트 완료
- [ ] 에러 핸들링 검증
- [ ] 로깅 설정 확인
- [ ] 보안 검토 완료
- [ ] 성능 테스트 통과
- [ ] 모니터링 시스템 설정
- [ ] 백업 및 복구 계획 수립
- [ ] 운영 문서 업데이트

### 개발 환경 설정 확인사항

- [ ] Ollama 설치 및 실행
- [ ] 모델 다운로드 완료
- [ ] 환경 변수 설정
- [ ] 연결 테스트 성공
- [ ] 샘플 요청 테스트
- [ ] 디버그 로깅 설정

## 지원 및 문의

### 기술 지원
- 내부 LLM API: IT 헬프데스크
- Ollama 관련: 개발팀
- 시스템 통합: 아키텍처팀

### 추가 리소스
- [LLM Integration Agent 문서](../agents/llm_integration_agent.md)
- [시스템 아키텍처 문서](../sdd.md)
- [개발 가이드](../CLAUDE.md)

---
*LLM Integration Service Setup Guide v1.0*
*Last Updated: 2024-09-23*