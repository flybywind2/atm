/**
 * Problem Input Component
 * 
 * This component handles the initial problem description input
 * from users, including validation and submission according to
 * the Vanilla JavaScript Frontend Agent specifications.
 */

class ProblemInput {
    constructor(containerId, onSubmitCallback) {
        this.container = document.getElementById(containerId);
        this.onSubmit = onSubmitCallback;
        this.isSubmitting = false;
        
        this.render();
        this.setupEventListeners();
    }
    
    /**
     * Render the component HTML
     */
    render() {
        this.container.innerHTML = `
            <div class="problem-input-container">
                <div class="header">
                    <h1>AI 문제 해결 코파일럿</h1>
                    <p>문제를 설명하면 단계별 가이드와 함께 포괄적인 Python 솔루션을 제공합니다.</p>
                </div>
                
                <form id="problem-input-form" class="problem-input-form">
                    <div class="form-group">
                        <label for="problem-description">
                            문제를 자세히 설명해주세요 *
                        </label>
                        <textarea 
                            id="problem-description" 
                            class="form-control" 
                            rows="8"
                            placeholder="예시: 월간 보고서 생성을 자동화하고 싶습니다. 현재 여러 엑셀 파일에서 수동으로 데이터를 수집하고, 차트를 만들어 PowerPoint에 삽입하고 있습니다. 이 과정은 3-4시간이 걸리고 오류가 발생하기 쉽습니다. 30분 이내에 완료되도록 자동화하고 싶습니다."
                            required
                            maxlength="5000"
                        ></textarea>
                        <div class="form-help">
                            💡 <strong>팁:</strong> 더 나은 분석을 위해 현재 프로세스, 사용하는 도구, 예상 결과 및 제약 사항에 대한 구체적인 세부 정보를 포함해주세요.
                        </div>
                        <div class="character-count">
                            <span id="char-count">0</span> / 5000 글자
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>추가 정보 (선택사항)</label>
                        
                        <div class="context-section">
                            <div class="form-row">
                                <div class="form-col">
                                    <label for="experience-level">프로그래밍 경험</label>
                                    <select id="experience-level" class="form-control">
                                        <option value="">수준을 선택하세요</option>
                                        <option value="beginner">초급 (Python 처음 접함)</option>
                                        <option value="intermediate">중급 (어느 정도 경험 있음)</option>
                                        <option value="advanced">고급 (숙련된 개발자)</option>
                                    </select>
                                </div>
                                
                                <div class="form-col">
                                    <label for="urgency-level">긴급도</label>
                                    <select id="urgency-level" class="form-control">
                                        <option value="">긴급도를 선택하세요</option>
                                        <option value="high">높음 (즉시 필요)</option>
                                        <option value="medium">보통 (1-2주 내)</option>
                                        <option value="low">낮음 (서두르지 않음)</option>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="form-group">
                                <label for="current-tools">현재 사용 중인 도구 및 기술</label>
                                <input
                                    type="text"
                                    id="current-tools"
                                    class="form-control"
                                    placeholder="예: Excel, PowerPoint, SQL Server, API 엔드포인트 등"
                                >
                            </div>
                            
                            <div class="form-group">
                                <label for="constraints">제약사항 또는 요구사항</label>
                                <textarea
                                    id="constraints"
                                    class="form-control"
                                    rows="3"
                                    placeholder="예: 특정 소프트웨어 사용 필수, 보안 요구사항, 예산 제한, 성능 요구사항 등"
                                ></textarea>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-actions">
                        <button type="submit" class="btn btn-primary" id="submit-btn">
                            <span class="btn-text">분석 시작</span>
                            <span class="btn-loading hidden">시작하는 중...</span>
                        </button>
                        
                        <button type="button" class="btn btn-outline" id="reset-btn">
                            폼 초기화
                        </button>
                    </div>
                </form>
                
                <div class="examples-section">
                    <h3>문제 설명 예시</h3>
                    <div class="examples-grid">
                        <div class="example-card" data-example="automation">
                            <h4>🤖 프로세스 자동화</h4>
                            <p>데이터 수집 및 보고서 생성과 같은 반복적인 작업을 자동화</p>
                        </div>
                        
                        <div class="example-card" data-example="data-analysis">
                            <h4>📊 데이터 분석</h4>
                            <p>대용량 데이터셋 분석 및 시각화 대시보드 생성</p>
                        </div>
                        
                        <div class="example-card" data-example="integration">
                            <h4>🔗 시스템 통합</h4>
                            <p>여러 시스템을 연결하고 데이터 동기화 자동화</p>
                        </div>
                        
                        <div class="example-card" data-example="web-scraping">
                            <h4>🕷️ 웹 스크래핑</h4>
                            <p>웹사이트에서 자동으로 정기적으로 데이터 추출</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        const form = this.container.querySelector('#problem-input-form');
        const textarea = this.container.querySelector('#problem-description');
        const submitBtn = this.container.querySelector('#submit-btn');
        const resetBtn = this.container.querySelector('#reset-btn');
        const exampleCards = this.container.querySelectorAll('.example-card');
        const charCount = this.container.querySelector('#char-count');
        
        // Form submission
        form.addEventListener('submit', this.handleSubmit.bind(this));
        
        // Reset button
        resetBtn.addEventListener('click', this.reset.bind(this));
        
        // Character count and validation
        textarea.addEventListener('input', () => {
            this.updateCharacterCount();
            this.validateDescription();
        });
        
        // Example cards
        exampleCards.forEach(card => {
            card.addEventListener('click', () => {
                const exampleType = card.dataset.example;
                this.loadExample(exampleType);
            });
        });
        
        // Auto-resize textarea
        textarea.addEventListener('input', () => {
            this.autoResizeTextarea(textarea);
        });
        
        // Real-time validation
        textarea.addEventListener('blur', this.validateForm.bind(this));
        
        // Initial setup
        this.updateCharacterCount();
    }
    
    /**
     * Update character count display
     */
    updateCharacterCount() {
        const textarea = this.container.querySelector('#problem-description');
        const charCount = this.container.querySelector('#char-count');
        const count = textarea.value.length;
        
        charCount.textContent = count;
        
        if (count > 5000) {
            charCount.style.color = 'var(--error-color)';
            charCount.style.fontWeight = 'bold';
        } else if (count > 4500) {
            charCount.style.color = 'var(--warning-color)';
            charCount.style.fontWeight = 'bold';
        } else {
            charCount.style.color = 'var(--text-secondary)';
            charCount.style.fontWeight = 'normal';
        }
    }
    
    /**
     * Validate problem description
     */
    validateDescription() {
        const textarea = this.container.querySelector('#problem-description');
        const description = textarea.value.trim();
        
        this.clearFieldError(textarea);
        
        if (description.length > 0 && description.length < 20) {
            this.showFieldError(textarea, '더 자세한 설명을 제공해주세요 (최소 20자).');
            return false;
        }

        if (description.length > 5000) {
            this.showFieldError(textarea, '설명은 5000자를 초과할 수 없습니다.');
            return false;
        }
        
        return true;
    }
    
    /**
     * Auto-resize textarea
     */
    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 300) + 'px';
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
        
        const formData = this.getFormData();
        
        this.setSubmittingState(true);
        
        try {
            await this.onSubmit(formData);
        } catch (error) {
            console.error('Form submission error:', error);
            this.setSubmittingState(false);
            this.showFormError('분석 시작에 실패했습니다. 다시 시도해주세요.');
        }
    }
    
    /**
     * Get form data in the format expected by the API
     */
    getFormData() {
        const problemDescription = this.container.querySelector('#problem-description').value.trim();
        const experienceLevel = this.container.querySelector('#experience-level').value;
        const urgencyLevel = this.container.querySelector('#urgency-level').value;
        const currentTools = this.container.querySelector('#current-tools').value.trim();
        const constraints = this.container.querySelector('#constraints').value.trim();
        
        return {
            description: problemDescription,
            experience_level: experienceLevel || 'beginner',
            urgency: urgencyLevel,
            preferred_tools: currentTools ? currentTools.split(',').map(tool => tool.trim()) : [],
            constraints: constraints,
            timeline: urgencyLevel === 'high' ? 'immediate' : urgencyLevel === 'medium' ? '1-2 weeks' : 'flexible',
            additional_context: ''
        };
    }
    
    /**
     * Validate entire form
     */
    validateForm() {
        const textarea = this.container.querySelector('#problem-description');
        const description = textarea.value.trim();
        
        if (description.length === 0) {
            this.showFieldError(textarea, '문제 설명이 필요합니다.');
            textarea.focus();
            return false;
        }
        
        if (description.length < 20) {
            this.showFieldError(textarea, '더 자세한 설명을 제공해주세요 (최소 20자).');
            textarea.focus();
            return false;
        }

        if (description.length > 5000) {
            this.showFieldError(textarea, '설명은 5000자를 초과할 수 없습니다.');
            textarea.focus();
            return false;
        }
        
        this.clearFieldError(textarea);
        this.clearFormError();
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
     * Show form-level error
     */
    showFormError(message) {
        this.clearFormError();
        
        const form = this.container.querySelector('#problem-input-form');
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
     * Clear form-level error
     */
    clearFormError() {
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
        
        const submitBtn = this.container.querySelector('#submit-btn');
        const btnText = submitBtn.querySelector('.btn-text');
        const btnLoading = submitBtn.querySelector('.btn-loading');
        const form = this.container.querySelector('#problem-input-form');
        
        if (isSubmitting) {
            submitBtn.disabled = true;
            btnText.classList.add('hidden');
            btnLoading.classList.remove('hidden');
            form.style.opacity = '0.7';
        } else {
            submitBtn.disabled = false;
            btnText.classList.remove('hidden');
            btnLoading.classList.add('hidden');
            form.style.opacity = '1';
        }
    }
    
    /**
     * Load example text
     */
    loadExample(exampleType) {
        const examples = {
            automation: `I need to automate our monthly sales report generation process. Currently, our team spends 4-5 hours every month collecting data from multiple Excel spreadsheets (customer data, sales figures, inventory levels), manually creating charts and tables, then formatting everything into a PowerPoint presentation for management. 

The process involves:
1. Downloading 5-6 Excel files from different departments
2. Manually copying and consolidating data into a master spreadsheet
3. Creating pivot tables and charts
4. Copying charts into PowerPoint with specific formatting
5. Adding commentary and analysis

This manual process is error-prone and time-consuming. I want to automate this to reduce the time to under 1 hour while ensuring data accuracy and consistent formatting.`,
            
            'data-analysis': `I have daily website log files (CSV format, ~2-5 GB each) that I need to analyze to understand user behavior patterns. Currently, I can't open these files in Excel due to size limitations, and when I try to sample the data, I lose the overall picture.

I need to:
- Process large CSV files efficiently
- Calculate key metrics: page views, session duration, bounce rates, conversion funnels
- Identify peak usage times and popular content
- Track user journey patterns
- Generate automated daily/weekly reports with visualizations
- Create a dashboard that updates automatically

The data includes timestamps, user IDs, page URLs, session data, and user actions. I want to build a Python solution that can handle these large files and provide actionable insights.`,
            
            integration: `Our company uses three separate systems: CRM (customer data), Order Management System (sales data), and Warehouse Management System (inventory data). Currently, every Monday morning, I manually download Excel reports from each system and spend 2-3 hours creating a unified weekly business report.

The challenge is:
- Data formats are different across systems
- Manual data matching is error-prone
- Some data relationships require complex lookups
- The process needs to be completed by 10 AM for management meetings
- Data inconsistencies often require back-and-forth with different departments

I want to automate this integration to:
- Automatically fetch data from each system (they have APIs)
- Standardize and merge the data
- Identify and flag data inconsistencies
- Generate the weekly report automatically
- Send alerts if data is missing or anomalous`,
            
            'web-scraping': `I need to monitor competitor pricing across 3-4 e-commerce websites daily for about 200 products. Currently, this is done manually by our team members who visit each website and record prices in a spreadsheet.

Requirements:
- Extract product names, prices, availability status, and ratings
- Handle dynamic content loading (some sites use JavaScript)
- Respect website terms of service and implement proper delays
- Store data in a structured format for analysis
- Generate daily price change reports
- Alert when prices drop below certain thresholds
- Create weekly competitive analysis summaries

The websites have different layouts and some use anti-bot measures. I need a robust solution that can adapt to minor website changes and handle errors gracefully while maintaining data collection consistency.`
        };
        
        const textarea = this.container.querySelector('#problem-description');
        textarea.value = examples[exampleType] || '';
        
        // Update character count and auto-resize
        this.updateCharacterCount();
        this.autoResizeTextarea(textarea);
        
        // Clear any existing errors
        this.clearFieldError(textarea);
        
        // Focus on textarea and scroll to top
        textarea.focus();
        textarea.setSelectionRange(0, 0);
        textarea.scrollTop = 0;
    }
    
    /**
     * Reset form to initial state
     */
    reset() {
        const form = this.container.querySelector('#problem-input-form');
        form.reset();
        
        // Reset character count
        this.updateCharacterCount();
        
        // Reset textarea height
        const textarea = this.container.querySelector('#problem-description');
        textarea.style.height = 'auto';
        
        // Clear all errors
        this.clearFormError();
        const errors = this.container.querySelectorAll('.field-error');
        errors.forEach(error => error.remove());
        
        const fields = this.container.querySelectorAll('.form-control');
        fields.forEach(field => field.classList.remove('error'));
        
        this.setSubmittingState(false);
        
        // Focus on main textarea
        textarea.focus();
    }
    
    /**
     * Component lifecycle - called when component is shown
     */
    onShow() {
        const textarea = this.container.querySelector('#problem-description');
        if (textarea && !textarea.value) {
            textarea.focus();
        }
    }
}

// Export for global use
window.ProblemInput = ProblemInput;

// Export for Node.js if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProblemInput;
}