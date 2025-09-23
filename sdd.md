# AI 기반 문제 해결 코파일럿 시스템: 상세 명세서 (Spec Driven Development)

## I. 시스템 개요 및 목적 명세

### 1. 시스템 목표 [1, 2]

본 시스템은 Python 초보 직원들이 업무 문제를 입력하면 [1], AI 에이전트가 이를 분석하고 [1], 요구사항 정의 [1], 솔루션 추천 [1], 및 구현 가이드를 제공하여 [1] 직원 생산성 향상 및 기술적 자립도를 강화하는 것을 목표로 합니다 [2, 3].

### 2. 핵심 산출물 명세 [4, 5]

시스템은 사용자의 문제 입력을 기반으로 다음과 같은 전문적인 문서를 자동 생성합니다:

*   **요구사항 정의서 (SRS)**: 기능적 및 비기능적 요구사항, 사용자 스토리, 인수 조건 포함 [6-10].
*   **예상 사용자 여정 지도 (User Journey Map)**: 문제점(Pain Points)과 개선된 프로세스(To-Be)를 포함하는 Markdown 형식의 문서 [6, 8, 10-13].
*   **구축 계획서 (Implementation Plan)**: 상세한 작업 분해(WBS), 기술 스택, 코드 예시, 테스트 전략 등을 포함한 최종 가이드 [8, 10, 12, 14-16].
*   **기술 스택 추천서**: 추천된 솔루션 유형에 따른 최적의 Python 라이브러리 및 프레임워크 목록 [12, 14, 17].

## II. 기술 아키텍처 명세 (Architecture Specification)

### 1. 아키텍처 프레임워크 [18-20]

시스템은 각 구성 요소의 역할을 명확히 분리하여 확장성과 유지보수성을 극대화한 **3-Tier 아키텍처**를 기반으로 합니다 [18, 21].

| 계층 (Layer) | 핵심 기술 스택 (Core Technology) | 주요 역할 (Primary Role) | 상세 내용 |
| :--- | :--- | :--- | :--- |
| **Frontend** | **Vanilla JavaScript** (SPA) [18, 22, 23] | 사용자 인터페이스 및 상호작용 [20]. | 클라이언트 측 라우팅을 포함하여 전체 페이지 새로고침 없이 화면 전환 처리 [22, 24]. |
| **Backend** | **FastAPI** [18, 22, 25] | API 게이트웨이 및 비동기 작업 관리 [26]. | 고성능 비동기 처리 능력을 제공하며, 장시간 소요되는 AI 작업을 처리하는 데 필수적임 [22]. JWT 기반 인증을 통해 API 엔드포인트를 보호함 [22]. |
| **AI Orchestration** | **LangGraph** [18, 25, 27] | 다단계 문제 해결 워크플로우 실행 및 제어 [27, 28]. | 순환적 추론, 조건부 로직, 상태 지속성을 지원하는 그래프 기반 구조 제공 [27]. |
| **Persistence** | **SQLite** [18, 29, 30] | 에이전트 상태 영속성 관리 및 데이터 저장 [29]. | LangGraph의 "체크포인터(checkpointer)" 역할을 수행하여 내결함성을 확보함 [29]. |
| **LLM** | **OpenAI Compatible or Ollama** | LLM은 Open Comatible한 ./appendix/internal_llm.py를 사내에서 사용하고, 사외에서는 ollama를 통해 동작 test 가능하도록 함
| **RAG** | **RAGaaS Poral** | ./appendix/rag_retrieve.py를 사용하여 RAGaaS Portal에 저장된 Data를 불러와서 활용

### 2. 백엔드 핵심 메커니즘: 비동기 및 상태 기반 통신 [18, 29]

장기 실행 AI 작업의 특성상, 시스템은 **비동기적, 상태 기반 아키텍처**를 필요로 합니다 [18, 27].

| 통신 방식 | 특징 | 역할 |
| :--- | :--- | :--- |
| **폴링 기반 REST API** | 클라이언트가 작업 상태를 주기적으로 확인 [29, 31]. | **LangGraph의 `interrupt()` 기능**과 결합되어, 에이전트가 명확화 질문을 던지고 사용자 입력(`AWAITING_INPUT`)을 기다리며 중단될 때 유용함 [14]. |
| **LangGraph `interrupt()`** | 그래프 실행을 일시 중지시키고, 상태를 데이터베이스에 저장 [14]. | 인간 참여형(Human-in-the-Loop, HITL) 워크플로우를 구현하는 핵심 요소 [14]. |

## III. 데이터 모델 명세 (Data/State Specification)

LangGraph 워크플로우의 중앙 상태 객체는 Python의 `TypedDict`를 사용하여 정의되며, 워크플로우 전반에 걸쳐 수집되고 생성된 모든 정보를 포함합니다 [11, 32].

| 필드명 | 유형 | 설명 | 출처 |
| :--- | :--- | :--- | :--- |
| `problem_description` | `str` | 사용자가 최초로 입력한 문제 설명 [11, 33]. | [11, 33] |
| `conversation_history` | `List[Dict]` | 사용자의 추가 입력과 에이전트의 질문 이력 [11, 12]. | [11, 12] |
| `problem_analysis` | `Dict` | LLM이 구조화한 문제 분석 결과 (핵심 이슈, 제약사항, 성공 기준 포함) [11, 12, 34]. | [11, 12, 34] |
| `missing_information` | `List[str]` | 컨텍스트 수집 단계에서 부족하다고 식별된 정보 목록 [33, 35]. | [33, 35] |
| `user_journey_map` | `Optional[str/Dict]` | 생성된 사용자 여정 지도 (Markdown 또는 Dict) [11, 12]. | [11, 12] |
| `requirements_definition` | `Optional[str]` | 생성된 요구사항 정의서 (Markdown) [11, 12]. | [11, 12] |
| `recommended_solution_type` | `Optional[SolutionType]` | 추천된 솔루션 유형 (예: 'RAG', 'ML_CLASSIFICATION', 'SIMPLE_AUTOMATION') [11, 12]. | [11, 12] |
| `technology_stack` | `Optional[Dict]` | 추천된 기술 스택 및 라이브러리 [12, 14]. | [12, 14] |
| `implementation_plan` | `Optional[str/Dict]` | 생성된 최종 구현 계획 [12, 14]. | [12, 14] |
| `current_status` | `str` | 현재 워크플로우의 상태 (예: 'analyzing', 'awaiting_input', 'complete') [14]. | [14] |

## IV. 핵심 워크플로우 명세 (LangGraph Workflow Specification)

LangGraph는 다음 노드와 조건부 엣지를 포함하는 상태 그래프로 정의됩니다 [36-39]:

### 1. 워크플로우 단계 [26, 39]

| 단계 (Node) | Agent 역할 | 주요 메커니즘 |
| :--- | :--- | :--- |
| `analyze_problem` | 문제 분석 Agent | 비구조적 입력을 구조화된 JSON으로 변환 [34, 40]. 사고의 연쇄(CoT) 및 지식 생성 프롬프팅(Generated Knowledge Prompting) 사용 [40]. |
| `collect_context` | 컨텍스트 수집 Agent | `problem_analysis`의 모호성 필드를 검사하여 추가 정보 필요 여부를 판단 [41]. **LangGraph의 `interrupt()`를 호출**하여 HITL 사이클 시작 [14]. |
| `generate_requirements` | 요구사항 정의 Agent | 구조화된 문제 JSON을 기반으로 사용자 여정 지도 및 요구사항 정의서 생성 [6, 7]. |
| `design_solution` | 솔루션 설계 Agent | 요구사항 분석 및 **조건부 엣지(`add_conditional_edges`)**를 사용해 최적의 솔루션 유형으로 라우팅 [42-44]. |
| `create_guide` / `generate_code` | 구축 계획 Agent | 추천된 기술 스택과 솔루션을 종합하여 상세한 소프트웨어 개발 계획(SDP) 및 코드 예시 생성 [15, 45]. |

### 2. 핵심 조건부 로직 명세 [41, 43, 44]

| 노드 | 조건부 엣지 결정 함수 | 분기 로직 |
| :--- | :--- | :--- |
| **컨텍스트 수집** | `check_context_complete` [37] | 컨텍스트가 불충분할 경우 (`False`), 워크플로우를 **중단(END)**시키고 사용자 입력을 대기함 [37, 46]. 충분할 경우 (`True`), `generate_requirements` 노드로 진행함 [37, 46]. |
| **솔루션 설계** | `route_solution` [42] | 문제 유형(예: 'SIMPLE\_AUTOMATION', 'RAG', 'ML')을 분류한 문자열을 반환하여, 각 솔루션 경로로 분기함 [42, 43]. |

## V. API 엔드포인트 명세 (FastAPI API Specification)

FastAPI 백엔드는 클라이언트-서버 간의 비동기적 상호작용을 위해 다음 세 가지 핵심 엔드포인트를 제공합니다. 모든 엔드포인트는 Pydantic 스키마를 통해 요청/응답 유효성 검증을 수행합니다 [47, 48].

### 1. 작업 시작 (Start Analysis) [29, 48]

| Method | Endpoint | 설명 | 요청 모델 | 응답 모델 |
| :--- | :--- | :--- | :--- | :--- |
| **POST** | `/api/v1/start-analysis` | 사용자의 초기 문제 설명을 제출하고 LangGraph 스레드를 시작함. | `AnalysisRequest` (problem: str) [47] | `AnalysisResponse` (thread\_id: str, status: str) [47, 48] |
| **응답 코드** | `HTTP 202 Accepted` | 작업이 수락되었으며, 최종 결과를 기다리지 않음을 의미함 [29]. |

### 2. 상태 조회 (Get Status) [31, 48]

| Method | Endpoint | 설명 | 요청 모델 | 응답 모델 |
| :--- | :--- | :--- | :--- | :--- |
| **GET** | `/api/v1/status/{thread_id}` | 특정 작업의 현재 진행 상태를 조회함. | (Path Parameter: thread\_id) | `StatusResponse` (status: str, question: Optional[str], output: Optional[Dict]) [48] |
| **상태 값** | `PROCESSING`, `AWAITING_INPUT` (명확화 질문 포함), `COMPLETED`, `FAILED` [31]. |

### 3. 작업 재개 (Resume Analysis) [31, 49]

| Method | Endpoint | 설명 | 요청 모델 | 응답 모델 |
| :--- | :--- | :--- | :--- | :--- |
| **POST** | `/api/v1/resume/{thread_id}` | 사용자로부터 받은 추가 입력(답변)으로 중단된 LangGraph 스레드를 재개함. | `ResumeRequest` (userInput: str) [49] | (status: "PROCESSING") [49] |

## VI. 프론트엔드 구현 명세 (Frontend Implementation Spec)

프론트엔드는 **Vanilla JavaScript SPA**로 구현되며 [22, 23], 경량성과 제어 가능성을 우선시합니다 [22].

### 1. 핵심 UI 컴포넌트 [22, 50]

*   **ProblemInput**: 문제 입력 및 제출 담당 [51].
*   **ContextCollector**: 에이전트의 명확화 질문을 표시하고 사용자 답변을 수집하여 `/resume` 엔드포인트로 전송 [51].
*   **DocumentViewer**: 백엔드에서 생성된 Markdown 형식의 요구사항 정의서, 사용자 여정 지도를 표시 [51].
*   **ProgressTracker**: 작업 상태 엔드포인트(`/status`)를 폴링하여 진행 상황을 추적하고 UI를 동적으로 갱신 [27].

### 2. 출력 렌더링 요구사항 [52]

최종 결과물(계획서, 가이드)의 가독성과 유용성을 극대화하기 위해, 다음과 같은 외부 라이브러리 통합이 명시됩니다:

*   **Markdown 렌더링**: `marked.js`를 사용하여 에이전트가 생성한 Markdown 응답을 HTML로 파싱 [52, 53].
*   **코드 하이라이팅**: `highlight.js` 또는 `Prism.js`를 통합하여 생성된 계획 내의 코드 블록에 구문 강조를 적용 [52, 53].