# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 필요한 가이드를 제공합니다.

## 프로젝트 개요

Python 초보자들이 업무 문제를 분석하고 포괄적인 솔루션을 생성할 수 있도록 돕는 AI 기반 문제 해결 코파일럿 시스템입니다. FastAPI 백엔드, LangGraph AI 오케스트레이션, SQLite 영속성, Vanilla JavaScript 프론트엔드로 구성된 3-Tier 아키텍처를 사용합니다.

## 아키텍처

### 백엔드 (FastAPI)
- **핵심 프레임워크**: async/await 지원하는 FastAPI
- **AI 오케스트레이션**: 다단계 문제 해결 워크플로우를 위한 LangGraph
- **데이터베이스**: 에이전트 상태 영속성과 체크포인팅을 위한 SQLite
- **LLM 통합**: OpenAI Compatible API (내부용) 또는 Ollama (외부 테스트용)
- **RAG 서비스**: 데이터 검색을 위한 RAGaaS Portal 통합

### 프론트엔드 (Vanilla JavaScript SPA)
- **프레임워크 없음**: 최대 제어를 위한 순수 JavaScript
- **클라이언트 사이드 라우팅**: 전체 페이지 새로고침 없는 SPA
- **마크다운 렌더링**: AI 생성 콘텐츠 파싱을 위한 `marked.js`
- **코드 하이라이팅**: 구문 강조를 위한 `highlight.js` 또는 `Prism.js`

### AI 워크플로우 (LangGraph)
시스템은 다음 핵심 노드들로 상태 기반 워크플로우를 구현합니다:
- `analyze_problem`: 비구조화된 입력을 구조화된 JSON으로 변환
- `collect_context`: Human-in-the-loop을 통한 추가 정보 수집
- `generate_requirements`: 사용자 여정 지도 및 요구사항 문서 생성
- `design_solution`: 적절한 솔루션 유형으로 라우팅
- `create_guide`: 구현 계획 및 코드 예시 생성

## 핵심 상태 모델

중앙 상태 객체 (TypedDict)는 다음을 포함합니다:
- `problem_description`: 초기 사용자 입력
- `conversation_history`: 사용자-에이전트 상호작용 기록
- `problem_analysis`: 구조화된 문제 분석
- `user_journey_map`: 생성된 여정 지도 (마크다운)
- `requirements_definition`: 생성된 요구사항 문서 (마크다운)
- `recommended_solution_type`: 솔루션 분류
- `technology_stack`: 추천 라이브러리/프레임워크
- `implementation_plan`: 최종 구현 가이드
- `current_status`: 워크플로우 상태 추적

## API 엔드포인트

### 핵심 엔드포인트
- `POST /api/v1/start-analysis`: 새로운 문제 분석 시작
- `GET /api/v1/status/{thread_id}`: 분석 진행 상황 확인
- `POST /api/v1/resume/{thread_id}`: 사용자 입력으로 재개

### 상태 값
- `PROCESSING`: 분석 진행 중
- `AWAITING_INPUT`: 사용자 명확화 대기 중
- `COMPLETED`: 분석 완료
- `FAILED`: 분석 오류

## 주요 구현 패턴

### Human-in-the-Loop (HITL)
- LangGraph `interrupt()`를 사용하여 실행 일시 정지
- 에이전트 명확화 질문 활성화
- SQLite 체크포인터를 통한 상태 영속성

### 비동기 통신
- 장시간 실행 작업을 위한 폴링 기반 REST API
- 비차단 사용자 인터페이스 업데이트
- 상태 기반 진행 상황 추적

### 출력 생성
시스템은 4가지 주요 결과물을 생성합니다:
1. **요구사항 정의서 (SRS)**: 기능적/비기능적 요구사항
2. **사용자 여정 지도**: 현재 문제점과 개선된 프로세스
3. **구현 계획**: 상세한 WBS, 기술 스택, 코드 예시
4. **기술 스택 추천**: Python 라이브러리 및 프레임워크

## 솔루션 유형
시스템은 문제를 다음 솔루션 카테고리로 분류합니다:
- `SIMPLE_AUTOMATION`: 기본 스크립팅 솔루션
- `RAG`: 검색 증강 생성 시스템
- `ML_CLASSIFICATION`: 머신러닝 분류 작업
- 문제 분석에 기반한 커스텀 솔루션 유형

## 개발 노트

### 구현 우선순위
- **보안, 인증 등 실제 동작에 불필요한 기능은 구현하지 않음**
- 핵심 AI 워크플로우와 사용자 상호작용에 집중

### 데이터베이스 스키마
- SQLite가 내결함성을 위한 LangGraph 체크포인트를 저장
- 상태 영속성으로 중단 후 워크플로우 재개 가능

### LLM 설정
- **내부 사용**: `./appendix/internal_llm.py`를 통한 OpenAI Compatible API
- **외부 테스트**: 로컬 개발 및 테스트를 위한 Ollama
- 두 설정 모두 원활한 전환을 위한 동일한 인터페이스 지원

### RAG 통합
- **RAG 서비스**: RAGaaS Portal 접근을 위한 `./appendix/rag_retrieve.py` 사용
- **데이터 소스**: 컨텍스트 향상을 위해 RAGaaS Portal에서 저장된 데이터 검색
- **통합 지점**: RAG 데이터가 향상된 응답을 위해 LangGraph 워크플로우에 공급

### 프론트엔드 컴포넌트
- `ProblemInput`: 초기 문제 제출
- `ContextCollector`: 명확화 Q&A 처리
- `DocumentViewer`: 생성된 마크다운 문서 표시
- `ProgressTracker`: 상태 폴링 및 UI 업데이트

### 조건부 로직
- `check_context_complete`: 추가 사용자 입력 필요 여부 결정
- `route_solution`: 적절한 솔루션 경로로 분류 및 라우팅