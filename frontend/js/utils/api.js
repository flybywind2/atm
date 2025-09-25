/**
 * API 클라이언트 (비개발자용 안내)
 *
 * 백엔드(FastAPI)와 통신하는 모듈입니다.
 * - 분석 시작/상태 조회/재개를 위한 HTTP 요청을 보냅니다.
 * - 오류/타임아웃을 처리하고, JSON 응답을 프런트 컴포넌트에서 쓰기 쉽게 반환합니다.
 */

class APIClient {
    constructor(baseURL = 'http://localhost:8080/api/v1') {
        this.baseURL = baseURL;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };
        this.timeout = 600000; // 10 minutes timeout - for long-running workflow operations
    }
    
    /**
     * Make a generic HTTP request with error handling and timeout
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        
        const config = {
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };
        
        // Add timeout support
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        config.signal = controller.signal;
        
        try {
            const response = await fetch(url, config);
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                let errorMessage;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorData.message || errorData.error || `HTTP ${response.status}: ${response.statusText}`;
                    
                    // Handle specific error formats from FastAPI
                    if (Array.isArray(errorData.detail)) {
                        errorMessage = errorData.detail.map(err => err.msg).join(', ');
                    }
                } catch {
                    errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                }
                
                const error = new Error(errorMessage);
                error.status = response.status;
                error.response = response;
                throw error;
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
            
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new Error('Request timeout - please try again');
            }
            
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }
    
    /**
     * 새 분석 시작 (POST /api/v1/start-analysis)
     *
     * 예시 사용:
     *   api.startAnalysis({ description: '월간 보고서 자동화', experience_level: 'beginner' })
     * 예시 응답:
     *   { thread_id: '7c0b...', status: 'started', current_step: 'analyze_problem', ... }
     */
    async startAnalysis(problemData) {
        const payload = {
            problem_description: problemData.description || problemData.problem_description,
            user_context: {
                role: problemData.role || '',
                experience_level: problemData.experience_level || 'beginner',
                preferred_tools: problemData.preferred_tools || [],
                constraints: problemData.constraints || '',
                timeline: problemData.timeline || '',
                additional_context: problemData.additional_context || ''
            }
        };
        
        return await this.request('/start-analysis', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }
    
    /**
     * 상태 조회 (GET /api/v1/status/{thread_id})
     *
     * 예시 사용:
     *   api.getStatus('7c0b...')
     * 예시 응답:
     *   { status: 'collecting_context', progress_percentage: 50, questions: [...], results: {...} }
     */
    async getStatus(threadId) {
        if (!threadId) {
            throw new Error('Thread ID is required');
        }
        
        return await this.request(`/status/${threadId}`, {
            method: 'GET'
        });
    }
    
    /**
     * 재개(사용자 답변 제출) (POST /api/v1/resume/{thread_id})
     *
     * 예시 사용:
     *   api.resumeWorkflow('7c0b...', { input: 'CSV 10분 간격, 3개월 12만행', context_data: {...} })
     * 예시 응답:
     *   { status: 'processing', current_step: 'collecting_context', requires_input: false }
     */
    async resumeWorkflow(threadId, contextData) {
        if (!threadId) {
            throw new Error('Thread ID is required');
        }
        
        const payload = {
            user_input: contextData.input || contextData.user_input,
            context_data: contextData.context_data || contextData.additional_context || {}
        };
        
        return await this.request(`/resume/${threadId}`, {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }
    
    /**
     * Health check endpoint
     */
    async healthCheck() {
        return await this.request('/health', {
            method: 'GET'
        });
    }
    
    /**
     * Get API information and available endpoints
     */
    async getAPIInfo() {
        return await this.request('/', {
            method: 'GET'
        });
    }
    
    /**
     * Cancel a workflow (if supported by backend)
     */
    async cancelWorkflow(threadId) {
        if (!threadId) {
            throw new Error('Thread ID is required');
        }
        
        return await this.request(`/cancel/${threadId}`, {
            method: 'POST'
        });
    }
    
    /**
     * Get workflow history (if supported by backend)
     */
    async getWorkflowHistory(threadId) {
        if (!threadId) {
            throw new Error('Thread ID is required');
        }
        
        return await this.request(`/history/${threadId}`, {
            method: 'GET'
        });
    }
    
    /**
     * Test connection to the backend
     */
    async testConnection() {
        try {
            await this.healthCheck();
            return { success: true, message: 'Backend connection successful' };
        } catch (error) {
            return { 
                success: false, 
                message: `Backend connection failed: ${error.message}`,
                error: error
            };
        }
    }
    
    /**
     * Set timeout for requests
     */
    setTimeout(timeout) {
        this.timeout = timeout;
    }
    
    /**
     * Set base URL for requests
     */
    setBaseURL(baseURL) {
        this.baseURL = baseURL;
    }
    
    /**
     * Add custom headers for all requests
     */
    setHeaders(headers) {
        this.defaultHeaders = { ...this.defaultHeaders, ...headers };
    }
    
    /**
     * Get current configuration
     */
    getConfig() {
        return {
            baseURL: this.baseURL,
            timeout: this.timeout,
            headers: this.defaultHeaders
        };
    }
}

// Export for use in Node.js if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = APIClient;
}

// Make available globally in browser
if (typeof window !== 'undefined') {
    window.APIClient = APIClient;
}
