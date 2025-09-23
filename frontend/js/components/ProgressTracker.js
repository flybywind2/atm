/**
 * Progress Tracker Component
 * 
 * This component displays the real-time progress of the analysis workflow,
 * showing current status, progress percentage, and step information.
 */

class ProgressTrackerComponent {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentStatus = null;
        
        this.render();
        this.setupEventListeners();
    }
    
    /**
     * Render the component HTML
     */
    render() {
        this.container.innerHTML = `
            <div class="progress-tracker">
                <div class="status-header">
                    <div id="status-indicator" class="status-indicator processing">
                        <span class="status-icon">⏳</span>
                        <span id="status-text">분석 준비 중...</span>
                    </div>
                    <div id="thread-info" class="thread-info hidden">
                        작업 ID: <code id="thread-id">-</code>
                    </div>
                </div>
                
                <div class="progress-container">
                    <div class="progress-bar">
                        <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
                    </div>
                    <div id="progress-text" class="progress-text">0% 완료</div>
                </div>
                
                <div class="workflow-steps">
                    <h4>진행 단계</h4>
                    <div class="steps-list">
                        <div class="step-item" data-step="analyze">
                            <div class="step-icon">1</div>
                            <div class="step-content">
                                <div class="step-title">문제 분석</div>
                                <div class="step-description">문제를 구조화하고 카테고리를 분류합니다</div>
                            </div>
                            <div class="step-status">⏸️</div>
                        </div>
                        
                        <div class="step-item" data-step="context">
                            <div class="step-icon">2</div>
                            <div class="step-content">
                                <div class="step-title">컨텍스트 수집</div>
                                <div class="step-description">추가 정보를 수집하고 요구사항을 명확화합니다</div>
                            </div>
                            <div class="step-status">⏸️</div>
                        </div>
                        
                        <div class="step-item" data-step="requirements">
                            <div class="step-icon">3</div>
                            <div class="step-content">
                                <div class="step-title">요구사항 생성</div>
                                <div class="step-description">기능 및 비기능 요구사항을 정의합니다</div>
                            </div>
                            <div class="step-status">⏸️</div>
                        </div>
                        
                        <div class="step-item" data-step="solution">
                            <div class="step-icon">4</div>
                            <div class="step-content">
                                <div class="step-title">솔루션 설계</div>
                                <div class="step-description">적절한 기술 스택과 아키텍처를 선택합니다</div>
                            </div>
                            <div class="step-status">⏸️</div>
                        </div>
                        
                        <div class="step-item" data-step="guide">
                            <div class="step-icon">5</div>
                            <div class="step-content">
                                <div class="step-title">가이드 생성</div>
                                <div class="step-description">구현 가이드와 문서를 생성합니다</div>
                            </div>
                            <div class="step-status">⏸️</div>
                        </div>
                    </div>
                </div>
                
                <div id="current-message" class="current-message hidden">
                    <div class="message-content">
                        <span class="message-icon">💬</span>
                        <span id="message-text">처리 중입니다...</span>
                    </div>
                </div>
                
                <div class="progress-actions">
                    <button type="button" class="btn btn-outline" id="refresh-btn">
                        상태 새로고침
                    </button>
                    <button type="button" class="btn btn-outline" id="cancel-btn" disabled>
                        작업 취소
                    </button>
                </div>
            </div>
        `;
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        const refreshBtn = this.container.querySelector('#refresh-btn');
        const cancelBtn = this.container.querySelector('#cancel-btn');
        
        refreshBtn.addEventListener('click', this.requestStatusUpdate.bind(this));
        cancelBtn.addEventListener('click', this.cancelWorkflow.bind(this));
    }
    
    /**
     * Update status from API response
     */
    updateStatus(status) {\n        this.currentStatus = status;\n        \n        // Update thread ID\n        const threadIdElement = this.container.querySelector('#thread-id');\n        const threadInfoElement = this.container.querySelector('#thread-info');\n        if (status.thread_id) {\n            threadIdElement.textContent = status.thread_id;\n            threadInfoElement.classList.remove('hidden');\n        }\n        \n        // Update progress bar\n        this.updateProgress(status.progress_percentage);\n        \n        // Update status indicator\n        this.updateStatusIndicator(status.status, status.current_step);\n        \n        // Update workflow steps\n        this.updateWorkflowSteps(status.current_step, status.status);\n        \n        // Update message\n        this.updateMessage(status.message, status.requires_input);\n        \n        // Update timestamp\n        this.updateLastUpdated();\n    }\n    \n    /**\n     * Update progress bar\n     */\n    updateProgress(percentage) {\n        const progressFill = this.container.querySelector('#progress-fill');\n        const progressText = this.container.querySelector('#progress-text');\n        \n        const safePercentage = Math.max(0, Math.min(100, percentage || 0));\n        \n        progressFill.style.width = `${safePercentage}%`;\n        progressText.textContent = `${safePercentage}% 완료`;\n        \n        // Add animation\n        progressFill.style.transition = 'width 0.5s ease-in-out';\n    }\n    \n    /**\n     * Update status indicator\n     */\n    updateStatusIndicator(status, currentStep) {\n        const statusIndicator = this.container.querySelector('#status-indicator');\n        const statusText = this.container.querySelector('#status-text');\n        const statusIcon = statusIndicator.querySelector('.status-icon');\n        \n        // Remove all status classes\n        statusIndicator.classList.remove('processing', 'completed', 'error', 'paused');\n        \n        const statusConfig = {\n            'started': { class: 'processing', icon: '🚀', text: '분석 시작됨' },\n            'analyzing': { class: 'processing', icon: '🔍', text: '문제 분석 중' },\n            'collecting_context': { class: 'processing', icon: '📝', text: '컨텍스트 수집 중' },\n            'generating_requirements': { class: 'processing', icon: '📋', text: '요구사항 생성 중' },\n            'designing_solution': { class: 'processing', icon: '🎯', text: '솔루션 설계 중' },\n            'creating_guide': { class: 'processing', icon: '📚', text: '가이드 생성 중' },\n            'completed': { class: 'completed', icon: '✅', text: '분석 완료' },\n            'paused': { class: 'paused', icon: '⏸️', text: '사용자 입력 대기' },\n            'error': { class: 'error', icon: '❌', text: '오류 발생' }\n        };\n        \n        const config = statusConfig[status] || statusConfig['started'];\n        \n        statusIndicator.classList.add(config.class);\n        statusIcon.textContent = config.icon;\n        statusText.textContent = config.text;\n    }\n    \n    /**\n     * Update workflow steps\n     */\n    updateWorkflowSteps(currentStep, status) {\n        const stepItems = this.container.querySelectorAll('.step-item');\n        const stepMapping = {\n            'analyze_problem': 'analyze',\n            'collect_context': 'context', \n            'generate_requirements': 'requirements',\n            'design_solution': 'solution',\n            'create_guide': 'guide'\n        };\n        \n        stepItems.forEach(item => {\n            const stepType = item.dataset.step;\n            const statusElement = item.querySelector('.step-status');\n            \n            // Reset classes\n            item.classList.remove('active', 'completed', 'error');\n            \n            // Determine step status\n            if (stepMapping[currentStep] === stepType) {\n                if (status === 'error') {\n                    item.classList.add('error');\n                    statusElement.textContent = '❌';\n                } else if (status === 'paused') {\n                    item.classList.add('active');\n                    statusElement.textContent = '⏸️';\n                } else {\n                    item.classList.add('active');\n                    statusElement.textContent = '🔄';\n                }\n            } else {\n                // Check if step is completed\n                const stepOrder = ['analyze', 'context', 'requirements', 'solution', 'guide'];\n                const currentIndex = stepOrder.indexOf(stepMapping[currentStep]);\n                const stepIndex = stepOrder.indexOf(stepType);\n                \n                if (stepIndex < currentIndex || status === 'completed') {\n                    item.classList.add('completed');\n                    statusElement.textContent = '✅';\n                } else {\n                    statusElement.textContent = '⏸️';\n                }\n            }\n        });\n    }\n    \n    /**\n     * Update current message\n     */\n    updateMessage(message, requiresInput) {\n        const messageContainer = this.container.querySelector('#current-message');\n        const messageText = this.container.querySelector('#message-text');\n        const messageIcon = messageContainer.querySelector('.message-icon');\n        \n        if (message) {\n            messageText.textContent = message;\n            messageIcon.textContent = requiresInput ? '❓' : '💬';\n            messageContainer.classList.remove('hidden');\n            \n            if (requiresInput) {\n                messageContainer.classList.add('requires-input');\n            } else {\n                messageContainer.classList.remove('requires-input');\n            }\n        } else {\n            messageContainer.classList.add('hidden');\n        }\n    }\n    \n    /**\n     * Update last updated timestamp\n     */\n    updateLastUpdated() {\n        let timestampElement = this.container.querySelector('#last-updated');\n        \n        if (!timestampElement) {\n            timestampElement = document.createElement('div');\n            timestampElement.id = 'last-updated';\n            timestampElement.className = 'last-updated';\n            this.container.appendChild(timestampElement);\n        }\n        \n        const now = new Date();\n        const timeString = now.toLocaleTimeString('ko-KR');\n        timestampElement.textContent = `마지막 업데이트: ${timeString}`;\n    }\n    \n    /**\n     * Request status update (for manual refresh)\n     */\n    requestStatusUpdate() {\n        // This would trigger a manual status check\n        // Implementation depends on parent component\n        console.log('Manual status update requested');\n        \n        const refreshBtn = this.container.querySelector('#refresh-btn');\n        refreshBtn.disabled = true;\n        refreshBtn.textContent = '새로고침 중...';\n        \n        setTimeout(() => {\n            refreshBtn.disabled = false;\n            refreshBtn.textContent = '상태 새로고침';\n        }, 2000);\n    }\n    \n    /**\n     * Cancel workflow\n     */\n    cancelWorkflow() {\n        if (confirm('정말로 작업을 취소하시겠습니까?')) {\n            console.log('Workflow cancellation requested');\n            // Implementation would depend on API support\n        }\n    }\n    \n    /**\n     * Reset component\n     */\n    reset() {\n        this.currentStatus = null;\n        \n        // Reset progress\n        this.updateProgress(0);\n        \n        // Reset status\n        this.updateStatusIndicator('started', '');\n        \n        // Reset steps\n        const stepItems = this.container.querySelectorAll('.step-item');\n        stepItems.forEach(item => {\n            item.classList.remove('active', 'completed', 'error');\n            const statusElement = item.querySelector('.step-status');\n            statusElement.textContent = '⏸️';\n        });\n        \n        // Hide message\n        this.container.querySelector('#current-message').classList.add('hidden');\n        \n        // Hide thread info\n        this.container.querySelector('#thread-info').classList.add('hidden');\n        \n        // Remove timestamp\n        const timestampElement = this.container.querySelector('#last-updated');\n        if (timestampElement) {\n            timestampElement.remove();\n        }\n    }\n    \n    /**\n     * Get current status\n     */\n    getCurrentStatus() {\n        return this.currentStatus;\n    }\n}\n\n// Export for global use\nwindow.ProgressTrackerComponent = ProgressTrackerComponent;\n\n// Export for Node.js if needed\nif (typeof module !== 'undefined' && module.exports) {\n    module.exports = ProgressTrackerComponent;\n}