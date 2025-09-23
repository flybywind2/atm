/**
 * Frontend integration tests
 */

// Mock DOM environment
const { JSDOM } = require('jsdom');
const dom = new JSDOM(`<!DOCTYPE html><html><body></body></html>`);
global.window = dom.window;
global.document = dom.window.document;
global.fetch = require('node-fetch');

describe('Frontend Integration Tests', () => {
    let container;
    let mockApp;

    beforeEach(() => {
        container = document.createElement('div');
        container.id = 'app-container';
        document.body.appendChild(container);

        // Mock application
        mockApp = {
            container: container,
            components: {},
            state: {
                currentStep: 'input',
                threadId: null,
                analysisStatus: 'idle',
                documents: {},
                error: null
            },

            init: function() {
                this.components.problemInput = this.createProblemInput();
                this.components.progressTracker = this.createProgressTracker();
                this.components.documentViewer = this.createDocumentViewer();
                this.components.contextCollector = this.createContextCollector();
                
                this.render();
                this.attachGlobalListeners();
            },

            render: function() {
                this.container.innerHTML = `
                    <div class="app">
                        <header class="app-header">
                            <h1>AI Problem Solving Assistant</h1>
                        </header>
                        <main class="app-main">
                            <div class="step-container ${this.state.currentStep === 'input' ? 'active' : 'hidden'}" id="input-step">
                                <div id="problem-input"></div>
                            </div>
                            <div class="step-container ${this.state.currentStep === 'progress' ? 'active' : 'hidden'}" id="progress-step">
                                <div id="progress-tracker"></div>
                            </div>
                            <div class="step-container ${this.state.currentStep === 'context' ? 'active' : 'hidden'}" id="context-step">
                                <div id="context-collector"></div>
                            </div>
                            <div class="step-container ${this.state.currentStep === 'results' ? 'active' : 'hidden'}" id="results-step">
                                <div id="document-viewer"></div>
                            </div>
                        </main>
                        <div id="error-display" class="error-display ${this.state.error ? 'visible' : 'hidden'}">
                            ${this.state.error || ''}
                        </div>
                    </div>
                `;

                this.renderComponents();
            },

            renderComponents: function() {
                if (this.components.problemInput) {
                    this.components.problemInput.container = document.getElementById('problem-input');
                    this.components.problemInput.render();
                }

                if (this.components.progressTracker) {
                    this.components.progressTracker.container = document.getElementById('progress-tracker');
                    this.components.progressTracker.render();
                }

                if (this.components.documentViewer) {
                    this.components.documentViewer.container = document.getElementById('document-viewer');
                    this.components.documentViewer.render();
                }

                if (this.components.contextCollector) {
                    this.components.contextCollector.container = document.getElementById('context-collector');
                    this.components.contextCollector.render();
                }
            },

            setState: function(newState) {
                this.state = { ...this.state, ...newState };
                this.render();
            },

            showStep: function(step) {
                this.setState({ currentStep: step });
            },

            showError: function(error) {
                this.setState({ error: error });
            },

            clearError: function() {
                this.setState({ error: null });
            },

            createProblemInput: function() {
                return {
                    container: null,
                    render: function() {
                        if (!this.container) return;
                        this.container.innerHTML = `
                            <div class="problem-input-form">
                                <h2>Describe Your Problem</h2>
                                <textarea id="problem-description" placeholder="Describe what you want to automate or solve..."></textarea>
                                <div class="context-inputs">
                                    <select id="technical-level">
                                        <option value="">Technical Level</option>
                                        <option value="beginner">Beginner</option>
                                        <option value="intermediate">Intermediate</option>
                                        <option value="advanced">Advanced</option>
                                    </select>
                                    <input type="text" id="environment" placeholder="Environment (Windows, Linux, etc.)">
                                    <input type="text" id="tools" placeholder="Current tools you use">
                                </div>
                                <button id="start-analysis-btn" type="button">Start Analysis</button>
                            </div>
                        `;
                        this.attachListeners();
                    },
                    attachListeners: function() {
                        const startBtn = this.container.querySelector('#start-analysis-btn');
                        if (startBtn) {
                            startBtn.addEventListener('click', () => this.handleSubmit());
                        }
                    },
                    handleSubmit: async function() {
                        const data = this.collectFormData();
                        if (this.validateData(data)) {
                            try {
                                await mockApp.startAnalysis(data);
                            } catch (error) {
                                mockApp.showError('Failed to start analysis: ' + error.message);
                            }
                        }
                    },
                    collectFormData: function() {
                        return {
                            problem_description: this.container.querySelector('#problem-description').value,
                            user_context: {
                                technical_level: this.container.querySelector('#technical-level').value,
                                environment: this.container.querySelector('#environment').value,
                                tools: this.container.querySelector('#tools').value
                            }
                        };
                    },
                    validateData: function(data) {
                        if (!data.problem_description || data.problem_description.length < 10) {
                            mockApp.showError('Please provide a detailed problem description (at least 10 characters)');
                            return false;
                        }
                        if (!data.user_context.technical_level) {
                            mockApp.showError('Please select your technical level');
                            return false;
                        }
                        return true;
                    }
                };
            },

            createProgressTracker: function() {
                return {
                    container: null,
                    render: function() {
                        if (!this.container) return;
                        this.container.innerHTML = `
                            <div class="progress-tracker">
                                <h2>Analysis Progress</h2>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: 0%"></div>
                                </div>
                                <div class="progress-status">Initializing...</div>
                                <div class="progress-steps"></div>
                            </div>
                        `;
                    },
                    updateProgress: function(data) {
                        const progressFill = this.container.querySelector('.progress-fill');
                        const statusElement = this.container.querySelector('.progress-status');
                        
                        if (progressFill) {
                            progressFill.style.width = `${data.progress_percentage || 0}%`;
                        }
                        if (statusElement) {
                            statusElement.textContent = data.message || 'Processing...';
                        }

                        // Check if user input is required
                        if (data.requires_input) {
                            mockApp.showContextCollection(data.questions);
                        }

                        // Check if completed
                        if (data.status === 'completed') {
                            mockApp.showResults(data.results);
                        }
                    }
                };
            },

            createContextCollector: function() {
                return {
                    container: null,
                    questions: [],
                    render: function() {
                        if (!this.container) return;
                        const questionsHtml = this.questions.map((question, index) => `
                            <div class="question">
                                <label for="answer-${index}">${question}</label>
                                <textarea id="answer-${index}" placeholder="Your answer..."></textarea>
                            </div>
                        `).join('');

                        this.container.innerHTML = `
                            <div class="context-collector">
                                <h2>Additional Information Needed</h2>
                                <p>Please provide more details to help us create a better solution:</p>
                                <div class="questions">
                                    ${questionsHtml}
                                </div>
                                <button id="submit-context-btn" type="button">Submit Answers</button>
                            </div>
                        `;
                        this.attachListeners();
                    },
                    attachListeners: function() {
                        const submitBtn = this.container.querySelector('#submit-context-btn');
                        if (submitBtn) {
                            submitBtn.addEventListener('click', () => this.handleSubmit());
                        }
                    },
                    setQuestions: function(questions) {
                        this.questions = questions || [];
                        this.render();
                    },
                    handleSubmit: async function() {
                        const answers = this.collectAnswers();
                        try {
                            await mockApp.resumeAnalysis(answers);
                        } catch (error) {
                            mockApp.showError('Failed to submit context: ' + error.message);
                        }
                    },
                    collectAnswers: function() {
                        const answers = [];
                        this.questions.forEach((question, index) => {
                            const textarea = this.container.querySelector(`#answer-${index}`);
                            if (textarea) {
                                answers.push(textarea.value);
                            }
                        });
                        return answers;
                    }
                };
            },

            createDocumentViewer: function() {
                return {
                    container: null,
                    documents: {},
                    activeTab: null,
                    render: function() {
                        if (!this.container) return;
                        const tabsHtml = Object.keys(this.documents).map(docType => `
                            <button class="doc-tab ${docType === this.activeTab ? 'active' : ''}" 
                                    data-doc-type="${docType}">
                                ${this.getDocumentTitle(docType)}
                            </button>
                        `).join('');

                        this.container.innerHTML = `
                            <div class="document-viewer">
                                <h2>Generated Documents</h2>
                                <div class="document-tabs">
                                    ${tabsHtml}
                                </div>
                                <div class="document-content">
                                    ${this.renderActiveDocument()}
                                </div>
                            </div>
                        `;
                        this.attachListeners();
                    },
                    loadDocuments: function(documents) {
                        this.documents = documents;
                        this.activeTab = Object.keys(documents)[0] || null;
                        this.render();
                    },
                    renderActiveDocument: function() {
                        if (!this.activeTab || !this.documents[this.activeTab]) {
                            return '<div class="no-document">No document selected</div>';
                        }
                        return `<div class="document-content">${this.documents[this.activeTab]}</div>`;
                    },
                    getDocumentTitle: function(docType) {
                        const titles = {
                            'requirements_doc': 'Requirements',
                            'implementation_guide': 'Implementation Guide',
                            'user_journey': 'User Journey'
                        };
                        return titles[docType] || docType;
                    },
                    attachListeners: function() {
                        const tabs = this.container.querySelectorAll('.doc-tab');
                        tabs.forEach(tab => {
                            tab.addEventListener('click', () => {
                                this.activeTab = tab.dataset.docType;
                                this.render();
                            });
                        });
                    }
                };
            },

            // API interaction methods
            startAnalysis: async function(data) {
                this.clearError();
                this.showStep('progress');
                
                // Mock API call
                const response = await this.mockApiCall('/api/v1/start-analysis', 'POST', data);
                
                if (response.thread_id) {
                    this.state.threadId = response.thread_id;
                    this.startPolling();
                } else {
                    throw new Error('Failed to start analysis');
                }
            },

            resumeAnalysis: async function(answers) {
                if (!this.state.threadId) throw new Error('No active analysis');
                
                const data = {
                    user_input: answers.join('\n'),
                    context_data: { answers: answers }
                };
                
                await this.mockApiCall(`/api/v1/resume/${this.state.threadId}`, 'POST', data);
                
                this.showStep('progress');
                this.startPolling();
            },

            startPolling: function() {
                if (this.pollingInterval) {
                    clearInterval(this.pollingInterval);
                }

                const poll = async () => {
                    try {
                        const status = await this.mockApiCall(`/api/v1/status/${this.state.threadId}`, 'GET');
                        this.handleStatusUpdate(status);
                    } catch (error) {
                        this.showError('Failed to get status: ' + error.message);
                        this.stopPolling();
                    }
                };

                this.pollingInterval = setInterval(poll, 2000);
                poll(); // Initial poll
            },

            stopPolling: function() {
                if (this.pollingInterval) {
                    clearInterval(this.pollingInterval);
                    this.pollingInterval = null;
                }
            },

            handleStatusUpdate: function(status) {
                if (this.components.progressTracker) {
                    this.components.progressTracker.updateProgress(status);
                }

                if (status.status === 'completed' || status.status === 'error') {
                    this.stopPolling();
                }
            },

            showContextCollection: function(questions) {
                this.showStep('context');
                if (this.components.contextCollector) {
                    this.components.contextCollector.setQuestions(questions);
                }
            },

            showResults: function(results) {
                this.showStep('results');
                if (this.components.documentViewer) {
                    this.components.documentViewer.loadDocuments(results);
                }
            },

            // Mock API for testing
            mockApiCall: async function(url, method, data) {
                await new Promise(resolve => setTimeout(resolve, 100)); // Simulate network delay
                
                if (url.includes('/start-analysis')) {
                    return { 
                        status: 'started', 
                        thread_id: 'mock-thread-' + Date.now(),
                        message: 'Analysis started successfully'
                    };
                }
                
                if (url.includes('/status/')) {
                    // Simulate different status responses
                    const responses = [
                        {
                            status: 'running',
                            current_step: 'analyze_problem',
                            progress_percentage: 25,
                            message: 'Analyzing your problem...'
                        },
                        {
                            status: 'awaiting_input',
                            current_step: 'collect_context',
                            progress_percentage: 40,
                            requires_input: true,
                            questions: ['What tools do you currently use?', 'What is your timeline?']
                        },
                        {
                            status: 'completed',
                            current_step: 'create_guide',
                            progress_percentage: 100,
                            results: {
                                requirements_doc: '# Requirements\n## Functional Requirements\n- Feature 1',
                                implementation_guide: '# Implementation Guide\n## Setup\n```python\nprint("hello")\n```',
                                user_journey: '# User Journey\n## Current State\n- Manual process'
                            }
                        }
                    ];
                    
                    // Cycle through responses based on call count
                    this.apiCallCount = (this.apiCallCount || 0) + 1;
                    return responses[(this.apiCallCount - 1) % responses.length];
                }
                
                if (url.includes('/resume/')) {
                    return { status: 'resumed', message: 'Analysis resumed successfully' };
                }
                
                throw new Error('Unknown API endpoint');
            },

            attachGlobalListeners: function() {
                // Handle keyboard shortcuts
                document.addEventListener('keydown', (e) => {
                    if (e.ctrlKey && e.key === 'r') {
                        e.preventDefault();
                        this.restart();
                    }
                });

                // Handle window beforeunload
                window.addEventListener('beforeunload', (e) => {
                    if (this.state.threadId && this.state.analysisStatus !== 'completed') {
                        e.preventDefault();
                        e.returnValue = 'Analysis is in progress. Are you sure you want to leave?';
                    }
                });
            },

            restart: function() {
                this.stopPolling();
                this.setState({
                    currentStep: 'input',
                    threadId: null,
                    analysisStatus: 'idle',
                    documents: {},
                    error: null
                });
                this.apiCallCount = 0;
            }
        };
    });

    afterEach(() => {
        if (mockApp.pollingInterval) {
            clearInterval(mockApp.pollingInterval);
        }
        if (container && container.parentNode) {
            container.parentNode.removeChild(container);
        }
    });

    describe('Application Initialization', () => {
        test('initializes all components correctly', () => {
            mockApp.init();
            
            expect(mockApp.components.problemInput).toBeDefined();
            expect(mockApp.components.progressTracker).toBeDefined();
            expect(mockApp.components.documentViewer).toBeDefined();
            expect(mockApp.components.contextCollector).toBeDefined();
            
            expect(container.querySelector('.app')).not.toBeNull();
            expect(container.querySelector('#input-step')).not.toBeNull();
            expect(container.querySelector('#progress-step')).not.toBeNull();
        });

        test('shows input step by default', () => {
            mockApp.init();
            
            expect(mockApp.state.currentStep).toBe('input');
            expect(container.querySelector('#input-step.active')).not.toBeNull();
            expect(container.querySelector('#progress-step.active')).toBeNull();
        });
    });

    describe('Complete User Flow', () => {
        test('completes full workflow without user input', async () => {
            mockApp.init();
            
            // Start with problem input
            const problemInput = mockApp.components.problemInput;
            problemInput.container.querySelector('#problem-description').value = 'Test automation problem that needs solving';
            problemInput.container.querySelector('#technical-level').value = 'intermediate';
            
            // Submit analysis
            await problemInput.handleSubmit();
            
            expect(mockApp.state.currentStep).toBe('progress');
            expect(mockApp.state.threadId).toBeTruthy();
            
            // Wait for polling to complete (mock will go through states)
            await new Promise(resolve => setTimeout(resolve, 300));
            
            // Should eventually reach results
            expect(mockApp.state.currentStep).toBe('results');
        });

        test('handles workflow with user input required', async () => {
            mockApp.init();
            
            // Modify mock to require user input
            const originalMockCall = mockApp.mockApiCall;
            mockApp.mockApiCall = async function(url, method, data) {
                if (url.includes('/status/')) {
                    return {
                        status: 'awaiting_input',
                        current_step: 'collect_context',
                        progress_percentage: 40,
                        requires_input: true,
                        questions: ['What tools do you use?', 'What is your timeline?']
                    };
                }
                return originalMockCall.call(this, url, method, data);
            };
            
            // Start analysis
            const problemInput = mockApp.components.problemInput;
            problemInput.container.querySelector('#problem-description').value = 'Complex automation problem';
            problemInput.container.querySelector('#technical-level').value = 'beginner';
            
            await problemInput.handleSubmit();
            
            // Wait for status update
            await new Promise(resolve => setTimeout(resolve, 150));
            
            // Should show context collection
            expect(mockApp.state.currentStep).toBe('context');
            
            // Restore original mock
            mockApp.mockApiCall = originalMockCall;
        });
    });

    describe('Error Handling', () => {
        test('displays validation errors', () => {
            mockApp.init();
            
            const problemInput = mockApp.components.problemInput;
            // Leave problem description empty
            problemInput.container.querySelector('#technical-level').value = 'beginner';
            
            problemInput.handleSubmit();
            
            expect(mockApp.state.error).toContain('problem description');
            expect(container.querySelector('.error-display.visible')).not.toBeNull();
        });

        test('handles API errors gracefully', async () => {
            mockApp.init();
            
            // Mock API failure
            mockApp.mockApiCall = async function() {
                throw new Error('Network connection failed');
            };
            
            const problemInput = mockApp.components.problemInput;
            problemInput.container.querySelector('#problem-description').value = 'Test problem that will fail';
            problemInput.container.querySelector('#technical-level').value = 'intermediate';
            
            await problemInput.handleSubmit();
            
            expect(mockApp.state.error).toContain('Network connection failed');
        });

        test('clears errors when restarting', () => {
            mockApp.init();
            mockApp.showError('Test error message');
            
            expect(mockApp.state.error).toBeTruthy();
            
            mockApp.restart();
            
            expect(mockApp.state.error).toBeNull();
            expect(mockApp.state.currentStep).toBe('input');
        });
    });

    describe('Component Integration', () => {
        test('progress tracker updates from status polling', async () => {
            mockApp.init();
            
            // Start analysis to begin polling
            const problemInput = mockApp.components.problemInput;
            problemInput.container.querySelector('#problem-description').value = 'Test integration';
            problemInput.container.querySelector('#technical-level').value = 'intermediate';
            
            await problemInput.handleSubmit();
            
            // Wait for initial status update
            await new Promise(resolve => setTimeout(resolve, 150));
            
            const progressFill = container.querySelector('.progress-fill');
            expect(progressFill.style.width).toBe('25%');
        });

        test('context collector integrates with progress tracker', async () => {
            mockApp.init();
            
            // Force context collection state
            mockApp.showContextCollection(['Test question 1', 'Test question 2']);
            
            expect(mockApp.state.currentStep).toBe('context');
            
            const questions = container.querySelectorAll('.question');
            expect(questions.length).toBe(2);
            
            // Fill answers
            container.querySelector('#answer-0').value = 'Answer 1';
            container.querySelector('#answer-1').value = 'Answer 2';
            
            // Submit context
            await mockApp.components.contextCollector.handleSubmit();
            
            expect(mockApp.state.currentStep).toBe('progress');
        });

        test('document viewer displays results from completed analysis', () => {
            mockApp.init();
            
            const mockResults = {
                requirements_doc: '# Test Requirements',
                implementation_guide: '# Test Guide',
                user_journey: '# Test Journey'
            };
            
            mockApp.showResults(mockResults);
            
            expect(mockApp.state.currentStep).toBe('results');
            
            const tabs = container.querySelectorAll('.doc-tab');
            expect(tabs.length).toBe(3);
            
            const activeContent = container.querySelector('.document-content');
            expect(activeContent.textContent).toContain('Test Requirements');
        });
    });

    describe('State Management', () => {
        test('maintains state consistency across step transitions', () => {
            mockApp.init();
            
            // Initial state
            expect(mockApp.state.currentStep).toBe('input');
            expect(mockApp.state.threadId).toBeNull();
            
            // Transition to progress
            mockApp.setState({ currentStep: 'progress', threadId: 'test-123' });
            expect(mockApp.state.currentStep).toBe('progress');
            expect(mockApp.state.threadId).toBe('test-123');
            
            // Transition to results
            mockApp.setState({ currentStep: 'results' });
            expect(mockApp.state.currentStep).toBe('results');
            expect(mockApp.state.threadId).toBe('test-123'); // Should preserve
        });

        test('handles concurrent state updates correctly', () => {
            mockApp.init();
            
            // Simulate rapid state updates
            mockApp.setState({ threadId: 'thread-1' });
            mockApp.setState({ currentStep: 'progress' });
            mockApp.setState({ error: 'test error' });
            
            expect(mockApp.state.threadId).toBe('thread-1');
            expect(mockApp.state.currentStep).toBe('progress');
            expect(mockApp.state.error).toBe('test error');
        });
    });

    describe('Accessibility and Usability', () => {
        test('provides keyboard navigation support', () => {
            mockApp.init();
            
            // Test restart shortcut
            const event = new dom.window.KeyboardEvent('keydown', {
                key: 'r',
                ctrlKey: true,
                bubbles: true
            });
            
            const initialStep = mockApp.state.currentStep;
            mockApp.setState({ currentStep: 'progress' });
            
            document.dispatchEvent(event);
            
            expect(mockApp.state.currentStep).toBe('input');
        });

        test('warns before page unload during active analysis', () => {
            mockApp.init();
            mockApp.setState({ threadId: 'active-123', analysisStatus: 'running' });
            
            const event = new dom.window.Event('beforeunload');
            let defaultPrevented = false;
            event.preventDefault = () => { defaultPrevented = true; };
            
            window.dispatchEvent(event);
            
            expect(defaultPrevented).toBe(true);
        });
    });
});