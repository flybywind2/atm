/**
 * Context Collector Component
 * 
 * This component handles the Human-in-the-Loop context collection phase,
 * displaying agent questions and collecting user responses according to
 * the Vanilla JavaScript Frontend Agent specifications.
 */

class ContextCollector {
    constructor(containerId, onSubmitCallback) {
        this.container = document.getElementById(containerId);
        this.onSubmit = onSubmitCallback;
        this.isSubmitting = false;
        this.questions = [];
        this.currentQuestionIndex = 0;
        this.responses = {};
        
        this.render();
        this.setupEventListeners();
    }
    
    /**
     * Render the component HTML
     */
    render() {
        this.container.innerHTML = `
            <div class="context-collector">
                <div class="collection-header">
                    <h2>추가&nbsp;정보가&nbsp;필요합니다</h2>
                    <p>더&nbsp;나은&nbsp;분석을&nbsp;위해&nbsp;몇&nbsp;가지&nbsp;질문을&nbsp;드립니다.</p>
                    <div class="progress-indicator">
                        <span id="question-progress">질문 1 / 1</span>
                    </div>
                </div>
                
                <div class="agent-question" id="agent-question">
                    <h3>질문</h3>
                    <div id="question-content" class="question-content">
                        질문을 준비하고 있습니다. 잠시만 기다려 주세요...
                    </div>
                </div>
                
                <form id="context-form" class="context-form">
                    <div class="form-group">
                        <label for="user-response">내 답변 *</label>
                        <textarea 
                            id="user-response" 
                            class="form-control" 
                            rows="4"
                            placeholder="구체적인 수치/주기/예시 등을 포함하여 자세히 작성해 주세요."
                            required
                            maxlength="2000"
                        ></textarea>
                        <div class="form-help">💡 <strong>팁:</strong> 가능한 한 구체적으로 작성하고, 모르는 항목은 현재 알고 있는 범위를 설명해 주세요.</div>
                        <div class="character-count"><span id="response-char-count">0</span> / 2000자</div>
                    </div>
                    
                    <div class="response-options" id="response-options" style="display: none;">
                        <!-- Dynamic response options will be inserted here -->
                    </div>
                    
                    <div class="form-actions">
                        <button type="submit" class="btn btn-primary" id="submit-context-btn">
                            <span class="btn-text">답변 제출</span>
                            <span class="btn-loading hidden">처리 중...</span>
                        </button>
                        
                        <button type="button" class="btn btn-secondary" id="skip-btn">
                            건너뛰기
                        </button>
                        
                        <button type="button" class="btn btn-outline" id="back-btn" style="display: none;">
                            이전 질문
                        </button>
                    </div>
                </form>
                
                <div class="context-help">
                    <h4>📝 좋은 답변을 작성하는 방법</h4>
                    <ul>
                        <li><strong>구체적으로:</strong> 수치/기간/예시를 포함해 주세요.</li>
                        <li><strong>현재 상태 설명:</strong> 현재의 업무 흐름/도구를 알려주세요.</li>
                        <li><strong>목표 명확화:</strong> 달성하고 싶은 결과를 적어주세요.</li>
                        <li><strong>제약사항:</strong> 예산/시간/기술적 한계를 알려주세요.</li>
                        <li><strong>모르는 경우:</strong> 모르는 부분은 모른다고 적어주셔도 됩니다.</li>
                    </ul>
                </div>
                
                <div class="conversation-history" id="conversation-history" style="display: none;">
                    <h4>Previous Responses</h4>
                    <div id="history-content"></div>
                </div>
            </div>
        `;
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        const form = this.container.querySelector('#context-form');
        const submitBtn = this.container.querySelector('#submit-context-btn');
        const skipBtn = this.container.querySelector('#skip-btn');
        const backBtn = this.container.querySelector('#back-btn');
        const textarea = this.container.querySelector('#user-response');
        const charCount = this.container.querySelector('#response-char-count');
        
        // Form submission
        form.addEventListener('submit', this.handleSubmit.bind(this));
        
        // Skip button
        skipBtn.addEventListener('click', this.handleSkip.bind(this));
        
        // Back button
        backBtn.addEventListener('click', this.handleBack.bind(this));
        
        // Character count and validation
        textarea.addEventListener('input', () => {
            this.updateCharacterCount();
            this.autoResizeTextarea(textarea);
        });
        
        // Real-time validation
        textarea.addEventListener('blur', this.validateForm.bind(this));
        
        // Initial setup
        this.updateCharacterCount();
    }
    
    /**
     * Set questions from the agent (HITL workflow)
     */
    setQuestions(questions) {
        this.questions = Array.isArray(questions) ? questions : [questions];
        this.currentQuestionIndex = 0;
        this.responses = {};
        
        if (this.questions.length > 0) {
            this.displayCurrentQuestion();
        }
    }
    
    /**
     * Display the current question
     */
    displayCurrentQuestion() {
        if (this.currentQuestionIndex >= this.questions.length) {
            this.handleAllQuestionsCompleted();
            return;
        }
        
        const question = this.questions[this.currentQuestionIndex];
        const questionContent = this.container.querySelector('#question-content');
        const progressIndicator = this.container.querySelector('#question-progress');
        const backBtn = this.container.querySelector('#back-btn');
        const historyContainer = this.container.querySelector('#conversation-history');
        
        // Update question content
        if (typeof question === 'string') {
            questionContent.innerHTML = this.formatPlainQuestion(question);
        } else {
            questionContent.innerHTML = this.formatQuestion(question);
            this.setupQuestionOptions(question);
        }
        
        // Update progress (Korean)
        progressIndicator.textContent = `질문 ${this.currentQuestionIndex + 1} / ${this.questions.length}`;
        
        // Show/hide back button
        backBtn.style.display = this.currentQuestionIndex > 0 ? 'inline-flex' : 'none';
        
        // Show conversation history if we have previous responses
        if (Object.keys(this.responses).length > 0) {
            this.updateConversationHistory();
            historyContainer.style.display = 'block';
        } else {
            historyContainer.style.display = 'none';
        }
        
        // Reset form for new question
        this.resetForm();
        
        // Focus on response input
        setTimeout(() => {
            const textarea = this.container.querySelector('#user-response');
            textarea.focus();
        }, 100);
    }

    /**
     * Format question object into HTML
     */
    formatQuestion(question) {
        let title = question.text || question.question || question;
        title = this.normalizeKoreanHints(String(title));
        let html = `<p>${title}</p>`;
        
        if (question.context) {
            html += `<div class="question-context"><strong>컨텍스트:</strong> ${question.context}</div>`;
        }
        
        if (question.examples) {
            html += '<div class="question-examples"><strong>예시:</strong><ul>';
            question.examples.forEach(example => {
                const ex = this.normalizeKoreanHints(String(example));
                html += `<li>${ex}</li>`;
            });
            html += '</ul></div>';
        }
        
        return html;
    }

    /**
     * Format plain string question: first line as text, following lines starting with '-' as list
     */
    formatPlainQuestion(text) {
        if (!text) return '';
        const lines = String(text).split(/\r?\n/).map(l => l.trim());
        const escape = (s) => String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        let html = '';
        if (lines.length > 0) {
            html += `<p>${escape(this.normalizeKoreanHints(lines[0]))}</p>`;
        }
        const rest = lines.slice(1).filter(l => l.length > 0);
        if (rest.length > 0) {
            html += '<ul>';
            rest.forEach(l => {
                const raw = l.replace(/^[\s]*[−–—\-•]\s*/, '');
                const item = this.normalizeKoreanHints(raw);
                html += `<li>${escape(item)}</li>`;
            });
            html += '</ul>';
        }
        return html || escape(text);
    }

    /**
     * Normalize common Korean hint lines without spaces into readable phrases
     */
    normalizeKoreanHints(s) {
        const map = {
            '오류가발생했습니다. 아래항목을포함하여요구사항을구체적으로작성해주세요': '오류가 발생했습니다. 아래 항목을 포함하여 요구사항을 구체적으로 작성해 주세요.',
            '현재환경서버/OS/네트워크': '현재 환경(서버/OS/네트워크)',
            '데이터출처형식규모': '데이터(출처/형식/규모)',
            '목표원하는결과': '목표(원하는 결과)',
            '제약시간예산보안등)': '제약(시간/예산/보안 등)',
            '이솔루션이해결해야하는핵심사용자시나리오는무엇인가요?': '이 솔루션이 해결해야 하는 핵심 사용자 시나리오는 무엇인가요?'
        };
        const trimmed = String(s).trim();
        if (map[trimmed]) return map[trimmed];
        // Soft fix: common patterns without spaces
        return trimmed
            .replace('오류가발생했습니다.', '오류가 발생했습니다.')
            .replace('아래항목을포함하여', '아래 항목을 포함하여')
            .replace('요구사항을구체적으로작성해주세요', '요구사항을 구체적으로 작성해 주세요')
            .replace('현재환경', '현재 환경')
            .replace('데이터출처형식규모', '데이터(출처/형식/규모)')
            .replace('목표원하는결과', '목표(원하는 결과)')
            .replace('제약시간예산보안등', '제약(시간/예산/보안 등)')
            .replace(/\)+$/, ')');
    }
    
    /**
     * Setup question options (for multiple choice questions)
     */
    setupQuestionOptions(question) {
        const optionsContainer = this.container.querySelector('#response-options');
        const textarea = this.container.querySelector('#user-response');
        
        if (question.options && Array.isArray(question.options)) {
            optionsContainer.style.display = 'block';
            textarea.style.display = 'none';
            
            let optionsHTML = '<div class="options-list">';
            question.options.forEach((option, index) => {
                optionsHTML += `
                    <div class="response-option">
                        <input type="radio" id="option-${index}" name="question-option" value="${option.value || option}">
                        <label for="option-${index}">${option.text || option}</label>
                    </div>
                `;
            });
            optionsHTML += '</div>';
            
            if (question.allowCustom) {
                optionsHTML += `
                    <div class="custom-option">
                        <input type="radio" id="option-custom" name="question-option" value="custom">
                        <label for="option-custom">Other (please specify):</label>
                        <textarea id="custom-response" class="form-control" rows="2" placeholder="Enter your custom response..."></textarea>
                    </div>
                `;
            }
            
            optionsContainer.innerHTML = optionsHTML;
            
            // Setup option change listeners
            const radioButtons = optionsContainer.querySelectorAll('input[type="radio"]');
            radioButtons.forEach(radio => {
                radio.addEventListener('change', this.handleOptionChange.bind(this));
            });
            
        } else {
            optionsContainer.style.display = 'none';
            textarea.style.display = 'block';
        }
    }
    
    /**
     * Handle option change for multiple choice questions
     */
    handleOptionChange(event) {
        const customTextarea = this.container.querySelector('#custom-response');
        
        if (event.target.value === 'custom' && customTextarea) {
            customTextarea.style.display = 'block';
            customTextarea.focus();
        } else if (customTextarea) {
            customTextarea.style.display = 'none';
        }
    }
    
    /**
     * Update character count display
     */
    updateCharacterCount() {
        const textarea = this.container.querySelector('#user-response');
        const charCount = this.container.querySelector('#response-char-count');
        
        if (textarea && charCount) {
            const count = textarea.value.length;
            charCount.textContent = count;
            
            if (count > 2000) {
                charCount.style.color = 'var(--error-color)';
            } else if (count > 1800) {
                charCount.style.color = 'var(--warning-color)';
            } else {
                charCount.style.color = 'var(--text-secondary)';
            }
        }
    }
    
    /**
     * Auto-resize textarea
     */
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
    
    /**
     * Handle form submission
     */
    async handleSubmit(event) {
        event.preventDefault();
        
        if (this.isSubmitting) {
            return;
        }
        
        if (!this.validateForm()) {
            return;
        }
        
        const response = this.getCurrentResponse();
        this.responses[this.currentQuestionIndex] = {
            question: this.questions[this.currentQuestionIndex],
            response: response,
            timestamp: new Date().toISOString()
        };
        
        this.currentQuestionIndex++;
        
        if (this.currentQuestionIndex < this.questions.length) {
            // More questions to show
            this.displayCurrentQuestion();
        } else {
            // All questions completed
            this.handleAllQuestionsCompleted();
        }
    }
    
    /**
     * Get current response from form
     */
    getCurrentResponse() {
        const optionsContainer = this.container.querySelector('#response-options');
        
        if (optionsContainer.style.display !== 'none') {
            // Multiple choice question
            const selectedOption = optionsContainer.querySelector('input[name="question-option"]:checked');
            if (selectedOption) {
                if (selectedOption.value === 'custom') {
                    const customTextarea = this.container.querySelector('#custom-response');
                    return customTextarea ? customTextarea.value.trim() : '';
                }
                return selectedOption.value;
            }
            return '';
        } else {
            // Text response
            const textarea = this.container.querySelector('#user-response');
            return textarea ? textarea.value.trim() : '';
        }
    }
    
    /**
     * Handle skip action
     */
    async handleSkip() {
        if (this.isSubmitting) {
            return;
        }
        
        this.responses[this.currentQuestionIndex] = {
            question: this.questions[this.currentQuestionIndex],
            response: '[SKIPPED]',
            skipped: true,
            timestamp: new Date().toISOString()
        };
        
        this.currentQuestionIndex++;
        
        if (this.currentQuestionIndex < this.questions.length) {
            this.displayCurrentQuestion();
        } else {
            this.handleAllQuestionsCompleted();
        }
    }
    
    /**
     * Handle back button
     */
    handleBack() {
        if (this.currentQuestionIndex > 0) {
            this.currentQuestionIndex--;
            this.displayCurrentQuestion();
            
            // Pre-fill with previous response if available
            const previousResponse = this.responses[this.currentQuestionIndex];
            if (previousResponse && !previousResponse.skipped) {
                const textarea = this.container.querySelector('#user-response');
                if (textarea) {
                    textarea.value = previousResponse.response;
                    this.updateCharacterCount();
                    this.autoResizeTextarea(textarea);
                }
            }
        }
    }
    
    /**
     * Handle all questions completed
     */
    async handleAllQuestionsCompleted() {
        this.setSubmittingState(true);
        
        try {
            const contextData = {
                input: this.formatAllResponses(),
                additional_context: {
                    total_questions: this.questions.length,
                    responses: this.responses,
                    completed_at: new Date().toISOString()
                }
            };
            
            await this.onSubmit(contextData);
        } catch (error) {
            console.error('Context submission error:', error);
            this.setSubmittingState(false);
            this.showError('Failed to submit responses. Please try again.');
        }
    }
    
    /**
     * Format all responses for submission
     */
    formatAllResponses() {
        let formattedResponse = "User responses to agent questions:\n\n";
        
        Object.keys(this.responses).forEach((index) => {
            const item = this.responses[index];
            const questionText = typeof item.question === 'string' ? item.question : item.question.text || item.question.question;
            
            formattedResponse += `Q${parseInt(index) + 1}: ${questionText}\n`;
            formattedResponse += `A${parseInt(index) + 1}: ${item.response}\n\n`;
        });
        
        return formattedResponse;
    }
    
    /**
     * Update conversation history display
     */
    updateConversationHistory() {
        const historyContent = this.container.querySelector('#history-content');
        let historyHTML = '';
        
        Object.keys(this.responses).forEach((index) => {
            const item = this.responses[index];
            const questionText = typeof item.question === 'string' ? item.question : item.question.text || item.question.question;
            
            historyHTML += `
                <div class="history-item">
                    <div class="history-question">Q${parseInt(index) + 1}: ${questionText}</div>
                    <div class="history-response ${item.skipped ? 'skipped' : ''}">
                        A${parseInt(index) + 1}: ${item.response}
                    </div>
                </div>
            `;
        });
        
        historyContent.innerHTML = historyHTML;
    }
    
    /**
     * Validate form
     */
    validateForm() {
        const optionsContainer = this.container.querySelector('#response-options');
        
        if (optionsContainer.style.display !== 'none') {
            // Multiple choice validation
            const selectedOption = optionsContainer.querySelector('input[name="question-option"]:checked');
            if (!selectedOption) {
                this.showError('옵션을 선택해 주세요.');
                return false;
            }
            
            if (selectedOption.value === 'custom') {
                const customTextarea = this.container.querySelector('#custom-response');
                if (!customTextarea || customTextarea.value.trim().length === 0) {
                    this.showError('직접 입력란을 작성해 주세요.');
                    return false;
                }
            }
        } else {
            // Text response validation
            const textarea = this.container.querySelector('#user-response');
            const response = textarea.value.trim();
            
            if (response.length === 0) {
                this.showFieldError(textarea, '답변을 입력해 주세요.');
                return false;
            }
            
            if (response.length > 2000) {
                this.showFieldError(textarea, '답변은 2000자를 초과할 수 없습니다.');
                return false;
            }
            
            this.clearFieldError(textarea);
        }
        
        this.clearError();
        return true;
    }
    
    /**
     * Show field error
     */
    showFieldError(field, message) {
        this.clearFieldError(field);
        
        field.classList.add('error');
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.textContent = message;
        
        field.parentNode.appendChild(errorDiv);
    }
    
    /**
     * Clear field error
     */
    clearFieldError(field) {
        field.classList.remove('error');
        
        const existingError = field.parentNode.querySelector('.field-error');
        if (existingError) {
            existingError.remove();
        }
    }
    
    /**
     * Show general error
     */
    showError(message) {
        this.clearError();
        
        const form = this.container.querySelector('#context-form');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'form-error';
        errorDiv.innerHTML = `
            <div class="error-content">
                <span class="error-icon">⚠️</span>
                <span class="error-message">${message}</span>
            </div>
        `;
        
        form.insertBefore(errorDiv, form.firstChild);
    }
    
    /**
     * Clear general error
     */
    clearError() {
        const existingError = this.container.querySelector('.form-error');
        if (existingError) {
            existingError.remove();
        }
    }
    
    /**
     * Set submitting state
     */
    setSubmittingState(isSubmitting) {
        this.isSubmitting = isSubmitting;
        
        const submitBtn = this.container.querySelector('#submit-context-btn');
        const skipBtn = this.container.querySelector('#skip-btn');
        const backBtn = this.container.querySelector('#back-btn');
        const btnText = submitBtn.querySelector('.btn-text');
        const btnLoading = submitBtn.querySelector('.btn-loading');
        
        if (isSubmitting) {
            submitBtn.disabled = true;
            skipBtn.disabled = true;
            backBtn.disabled = true;
            btnText.classList.add('hidden');
            btnLoading.classList.remove('hidden');
        } else {
            submitBtn.disabled = false;
            skipBtn.disabled = false;
            backBtn.disabled = false;
            btnText.classList.remove('hidden');
            btnLoading.classList.add('hidden');
        }
    }
    
    /**
     * Reset form
     */
    resetForm() {
        const form = this.container.querySelector('#context-form');
        const textarea = this.container.querySelector('#user-response');
        
        // Reset form fields
        form.reset();
        
        // Reset textarea height
        if (textarea) {
            textarea.style.height = 'auto';
        }
        
        // Clear errors
        this.clearError();
        const errors = this.container.querySelectorAll('.field-error');
        errors.forEach(error => error.remove());
        
        const fields = this.container.querySelectorAll('.form-control');
        fields.forEach(field => field.classList.remove('error'));
        
        // Update character count
        this.updateCharacterCount();
        
        this.setSubmittingState(false);
    }
    
    /**
     * Reset entire component
     */
    reset() {
        this.questions = [];
        this.currentQuestionIndex = 0;
        this.responses = {};
        this.resetForm();
        
        // Hide conversation history
        const historyContainer = this.container.querySelector('#conversation-history');
        historyContainer.style.display = 'none';
    }
    
    /**
     * Component lifecycle - called when component is shown
     */
    onShow() {
        if (this.questions.length > 0) {
            const textarea = this.container.querySelector('#user-response');
            if (textarea) {
                textarea.focus();
            }
        }
    }
}

// Export for global use
window.ContextCollector = ContextCollector;

// Export for Node.js if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ContextCollector;
}




