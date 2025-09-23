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
                    <h1>AI Problem Solving Copilot</h1>
                    <p>Describe your problem and get comprehensive Python solutions with step-by-step guidance.</p>
                </div>
                
                <form id="problem-input-form" class="problem-input-form">
                    <div class="form-group">
                        <label for="problem-description">
                            Describe your problem in detail *
                        </label>
                        <textarea 
                            id="problem-description" 
                            class="form-control" 
                            rows="8"
                            placeholder="Example: I need to automate monthly report generation. Currently I manually collect data from multiple Excel files, create charts, and insert them into PowerPoint. This process takes 3-4 hours and is prone to errors. I want to automate this to complete in under 30 minutes."
                            required
                            maxlength="5000"
                        ></textarea>
                        <div class="form-help">
                            💡 <strong>Tip:</strong> Include specific details about your current process, tools you use, expected outcomes, and any constraints for better analysis.
                        </div>
                        <div class="character-count">
                            <span id="char-count">0</span> / 5000 characters
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Additional Context (Optional)</label>
                        
                        <div class="context-section">
                            <div class="form-row">
                                <div class="form-col">
                                    <label for="experience-level">Programming Experience</label>
                                    <select id="experience-level" class="form-control">
                                        <option value="">Select your level</option>
                                        <option value="beginner">Beginner (New to Python)</option>
                                        <option value="intermediate">Intermediate (Some experience)</option>
                                        <option value="advanced">Advanced (Experienced developer)</option>
                                    </select>
                                </div>
                                
                                <div class="form-col">
                                    <label for="urgency-level">Urgency</label>
                                    <select id="urgency-level" class="form-control">
                                        <option value="">Select urgency</option>
                                        <option value="high">High (Needed immediately)</option>
                                        <option value="medium">Medium (Within 1-2 weeks)</option>
                                        <option value="low">Low (No rush)</option>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="form-group">
                                <label for="current-tools">Current Tools & Technologies</label>
                                <input 
                                    type="text" 
                                    id="current-tools" 
                                    class="form-control"
                                    placeholder="e.g., Excel, PowerPoint, SQL Server, API endpoints, etc."
                                >
                            </div>
                            
                            <div class="form-group">
                                <label for="constraints">Constraints or Requirements</label>
                                <textarea 
                                    id="constraints" 
                                    class="form-control" 
                                    rows="3"
                                    placeholder="e.g., Must use specific software, security requirements, budget limitations, performance needs, etc."
                                ></textarea>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-actions">
                        <button type="submit" class="btn btn-primary" id="submit-btn">
                            <span class="btn-text">Start Analysis</span>
                            <span class="btn-loading hidden">Starting...</span>
                        </button>
                        
                        <button type="button" class="btn btn-outline" id="reset-btn">
                            Reset Form
                        </button>
                    </div>
                </form>
                
                <div class="examples-section">
                    <h3>Example Problem Descriptions</h3>
                    <div class="examples-grid">
                        <div class="example-card" data-example="automation">
                            <h4>🤖 Process Automation</h4>
                            <p>Automate repetitive tasks like data collection and report generation</p>
                        </div>
                        
                        <div class="example-card" data-example="data-analysis">
                            <h4>📊 Data Analysis</h4>
                            <p>Analyze large datasets and create visualization dashboards</p>
                        </div>
                        
                        <div class="example-card" data-example="integration">
                            <h4>🔗 System Integration</h4>
                            <p>Connect multiple systems and automate data synchronization</p>
                        </div>
                        
                        <div class="example-card" data-example="web-scraping">
                            <h4>🕷️ Web Scraping</h4>
                            <p>Extract data from websites automatically and regularly</p>
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
            this.showFieldError(textarea, 'Please provide a more detailed description (at least 20 characters).');
            return false;
        }
        
        if (description.length > 5000) {
            this.showFieldError(textarea, 'Description cannot exceed 5000 characters.');
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
            this.showFormError('Failed to start analysis. Please try again.');
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
            this.showFieldError(textarea, 'Problem description is required.');
            textarea.focus();
            return false;
        }
        
        if (description.length < 20) {
            this.showFieldError(textarea, 'Please provide a more detailed description (at least 20 characters).');
            textarea.focus();
            return false;
        }
        
        if (description.length > 5000) {
            this.showFieldError(textarea, 'Description cannot exceed 5000 characters.');
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