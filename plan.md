# AI 기반 문제 해결 코파일럿 시스템 구현 계획

## 프로젝트 개요

Python 초보자들이 업무 문제를 분석하고 포괄적인 솔루션을 생성할 수 있도록 돕는 AI 기반 문제 해결 코파일럿 시스템입니다. FastAPI 백엔드, LangGraph AI 오케스트레이션, SQLite 영속성, Vanilla JavaScript 프론트엔드로 구성된 3-Tier 아키텍처를 사용합니다.

## 프로젝트 구조

```
atm/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 애플리케이션 진입점
│   │   ├── models/              # Pydantic 모델들 (요청/응답)
│   │   │   ├── __init__.py
│   │   │   ├── requests.py      # AnalysisRequest, ResumeRequest
│   │   │   └── responses.py     # StatusResponse, AnalysisResponse
│   │   ├── api/                 # API 엔드포인트들
│   │   │   ├── __init__.py
│   │   │   └── analysis.py      # 분석 관련 엔드포인트
│   │   ├── workflows/           # LangGraph 워크플로우
│   │   │   ├── __init__.py
│   │   │   ├── state.py         # TypedDict 상태 정의
│   │   │   └── graph.py         # 메인 워크플로우 그래프
│   │   ├── agents/              # 각 단계별 에이전트들
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py      # 문제 분석 에이전트
│   │   │   ├── context_collector.py # 컨텍스트 수집 에이전트
│   │   │   ├── requirements_generator.py # 요구사항 생성 에이전트
│   │   │   ├── solution_designer.py # 솔루션 설계 에이전트
│   │   │   └── guide_creator.py # 가이드 생성 에이전트
│   │   └── database/            # SQLite 설정
│   │       ├── __init__.py
│   │       └── checkpointer.py  # SQLite 체크포인터 설정
│   ├── appendix/
│   │   ├── internal_llm.py      # 내부 LLM API 인터페이스
│   │   └── rag_retrieve.py      # RAG 서비스 인터페이스
│   └── requirements.txt         # Python 의존성
├── frontend/
│   ├── index.html               # 메인 HTML
│   ├── js/
│   │   ├── app.js               # 메인 애플리케이션 로직
│   │   ├── components/          # UI 컴포넌트들
│   │   │   ├── ProblemInput.js
│   │   │   ├── ContextCollector.js
│   │   │   ├── DocumentViewer.js
│   │   │   └── ProgressTracker.js
│   │   └── utils/
│   │       ├── api.js           # API 호출 유틸리티
│   │       └── markdown.js      # 마크다운 렌더링
│   ├── css/
│   │   └── styles.css
│   └── libs/                    # 외부 라이브러리
│       ├── marked.min.js        # 마크다운 파서
│       └── highlight.min.js     # 코드 하이라이팅
└── CLAUDE.md                    # 개발 가이드
```

## 구현 단계

### 1단계: 프로젝트 기본 구조 및 의존성 설정
- 디렉토리 구조 생성
- requirements.txt 작성 (fastapi, langgraph, langgraph-checkpoint-sqlite, uvicorn 등)
- 기본 FastAPI 애플리케이션 설정
- 개발 환경 설정

### 2단계: 데이터 모델 및 상태 정의
- Pydantic 모델들 정의 (AnalysisRequest, StatusResponse, ResumeRequest 등)
- LangGraph TypedDict 상태 객체 정의 (problem_description, conversation_history 등)
- SQLite 데이터베이스 및 체크포인터 설정

### 3단계: LangGraph 워크플로우 구현
- 각 노드별 에이전트 구현:
  - analyze_problem: 문제 분석 및 구조화
  - collect_context: HITL을 통한 추가 정보 수집
  - generate_requirements: 요구사항 정의서 생성
  - design_solution: 솔루션 유형 분류 및 라우팅
  - create_guide: 최종 구현 가이드 생성
- 조건부 엣지 로직 구현 (check_context_complete, route_solution)
- interrupt() 기능을 활용한 Human-in-the-loop 구현

### 4단계: FastAPI 엔드포인트 구현
- POST /api/v1/start-analysis: 초기 문제 분석 시작
- GET /api/v1/status/{thread_id}: 진행 상태 조회 (폴링 지원)
- POST /api/v1/resume/{thread_id}: 사용자 입력으로 워크플로우 재개
- BackgroundTasks를 활용한 비동기 LangGraph 실행

### 5단계: LLM 및 RAG 통합
- appendix/internal_llm.py: 내부 OpenAI Compatible API 인터페이스
- appendix/rag_retrieve.py: RAGaaS Portal 데이터 검색 인터페이스
- 환경별 LLM 설정 (내부용/Ollama 외부용) 전환 기능
- LangGraph 에이전트들과 LLM/RAG 서비스 통합

### 6단계: 프론트엔드 구현
- 기본 HTML 구조 및 CSS 스타일링
- JavaScript 컴포넌트들:
  - ProblemInput: 초기 문제 입력
  - ContextCollector: 에이전트 질문 처리
  - DocumentViewer: 마크다운 문서 표시
  - ProgressTracker: 상태 폴링 및 진행 표시
- marked.js를 활용한 마크다운 렌더링
- highlight.js를 활용한 코드 구문 강조

### 7단계: 테스트 및 검증
- 각 컴포넌트별 단위 테스트
- Human-in-the-loop 플로우 통합 테스트
- 체크포인터 복구 테스트
- API 엔드포인트 테스트

### 8단계: 최적화 및 마무리
- 성능 최적화 및 에러 핸들링 강화
- 로깅 및 모니터링 설정
- 사용자 가이드 및 문서화

## 핵심 기술적 포인트

### 1. SQLite 체크포인터
- LangGraph의 SqliteSaver/AsyncSqliteSaver 활용
- 워크플로우 상태 영속성 보장
- 중단 후 재개 기능 지원

### 2. Human-in-the-loop (HITL)
- interrupt() 기능으로 워크플로우 일시 정지
- 에이전트 명확화 질문 처리
- 사용자 입력 대기 및 재개

### 3. 비동기 처리
- FastAPI BackgroundTasks로 장시간 AI 작업 처리
- 클라이언트 응답성 보장
- 논블로킹 사용자 인터페이스

### 4. 폴링 패턴
- 클라이언트의 주기적 상태 확인 방식
- 실시간 진행 상황 업데이트
- 상태 기반 UI 변경

### 5. LLM 추상화
- 내부/외부 환경 간 원활한 전환
- OpenAI Compatible API (내부용)
- Ollama (외부 테스트용)

### 6. RAG 통합
- RAGaaS Portal 데이터 활용
- 컨텍스트 향상을 위한 외부 데이터 검색
- LangGraph 워크플로우 내 통합

## 주요 산출물

시스템은 사용자의 문제 입력을 기반으로 다음과 같은 전문적인 문서를 자동 생성합니다:

1. **요구사항 정의서 (SRS)**: 기능적 및 비기능적 요구사항, 사용자 스토리, 인수 조건 포함
2. **사용자 여정 지도**: 문제점(Pain Points)과 개선된 프로세스(To-Be)를 포함하는 Markdown 형식의 문서
3. **구축 계획서**: 상세한 작업 분해(WBS), 기술 스택, 코드 예시, 테스트 전략 등을 포함한 최종 가이드
4. **기술 스택 추천서**: 추천된 솔루션 유형에 따른 최적의 Python 라이브러리 및 프레임워크 목록

## 솔루션 유형 분류

시스템은 문제를 다음 솔루션 카테고리로 분류합니다:

- `SIMPLE_AUTOMATION`: 기본 스크립팅 솔루션
- `RAG`: 검색 증강 생성 시스템
- `ML_CLASSIFICATION`: 머신러닝 분류 작업
- 문제 분석에 기반한 커스텀 솔루션 유형

## 개발 우선순위

- **보안, 인증 등 실제 동작에 불필요한 기능은 구현하지 않음**
- 핵심 AI 워크플로우와 사용자 상호작용에 집중
- 빠른 프로토타이핑과 검증을 우선시