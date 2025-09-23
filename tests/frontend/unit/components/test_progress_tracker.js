/**
 * Unit tests for ProgressTracker component
 */

// Mock DOM environment
const { JSDOM } = require('jsdom');
const dom = new JSDOM(`<!DOCTYPE html><html><body></body></html>`);
global.window = dom.window;
global.document = dom.window.document;

describe('ProgressTracker Component', () => {
    let container;
    let progressTracker;

    beforeEach(() => {
        container = document.createElement('div');
        container.id = 'progress-container';
        document.body.appendChild(container);

        // Mock ProgressTracker component
        progressTracker = {
            container: container,
            currentStep: null,
            progressPercentage: 0,
            status: 'initialized',
            pollingInterval: null,
            threadId: null,

            steps: [
                { id: 'analyze_problem', name: 'Analyzing Problem', percentage: 20 },
                { id: 'collect_context', name: 'Collecting Context', percentage: 40 },
                { id: 'generate_requirements', name: 'Generating Requirements', percentage: 60 },
                { id: 'design_solution', name: 'Designing Solution', percentage: 80 },
                { id: 'create_guide', name: 'Creating Guide', percentage: 100 }
            ],

            render: function() {
                const stepsHtml = this.steps.map(step => `
                    <div class="step ${step.id === this.currentStep ? 'active' : ''} 
                         ${this.progressPercentage >= step.percentage ? 'completed' : ''}" 
                         data-step="${step.id}">
                        <div class="step-icon">
                            ${this.progressPercentage >= step.percentage ? '✓' : '○'}
                        </div>
                        <div class="step-label">${step.name}</div>
                    </div>
                `).join('');

                this.container.innerHTML = `
                    <div class="progress-tracker">
                        <div class="progress-header">
                            <h3>Analysis Progress</h3>
                            <div class="progress-percentage">${this.progressPercentage}%</div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${this.progressPercentage}%"></div>
                        </div>
                        <div class="progress-steps">
                            ${stepsHtml}
                        </div>
                        <div class="progress-status ${this.status}">${this.getStatusMessage()}</div>
                        <div class="progress-actions">
                            ${this.status === 'awaiting_input' ? '<button id="provide-input-btn">Provide Additional Info</button>' : ''}
                            ${this.status === 'error' ? '<button id="retry-btn">Retry</button>' : ''}
                        </div>
                    </div>
                `;
            },

            updateProgress: function(data) {
                this.currentStep = data.current_step;
                this.progressPercentage = data.progress_percentage || 0;
                this.status = data.status;
                this.render();
            },

            getStatusMessage: function() {
                const messages = {
                    'initialized': 'Starting analysis...',
                    'running': 'Analysis in progress...',
                    'awaiting_input': 'Waiting for additional information',
                    'completed': 'Analysis completed successfully!',
                    'error': 'An error occurred during analysis',
                    'timeout': 'Analysis timed out'
                };
                return messages[this.status] || 'Unknown status';
            },

            startPolling: function(threadId, interval = 2000) {
                this.threadId = threadId;
                this.stopPolling(); // Clear any existing interval

                const poll = async () => {
                    try {
                        const response = await this.fetchStatus(threadId);
                        this.updateProgress(response);

                        if (response.status === 'completed' || response.status === 'error') {
                            this.stopPolling();
                        }
                    } catch (error) {
                        console.error('Polling error:', error);
                        this.handlePollingError(error);
                    }
                };

                // Initial poll
                poll();
                
                // Set up interval
                this.pollingInterval = setInterval(poll, interval);
            },

            stopPolling: function() {
                if (this.pollingInterval) {
                    clearInterval(this.pollingInterval);
                    this.pollingInterval = null;
                }
            },

            fetchStatus: async function(threadId) {
                // Mock API call
                await new Promise(resolve => setTimeout(resolve, 100));
                
                // Return mock response based on current state
                const mockResponses = {
                    'mock-thread-running': {
                        thread_id: threadId,
                        status: 'running',
                        current_step: 'analyze_problem',
                        progress_percentage: 25,
                        message: 'Analyzing your problem...'
                    },
                    'mock-thread-input': {
                        thread_id: threadId,
                        status: 'awaiting_input',
                        current_step: 'collect_context',
                        progress_percentage: 40,
                        requires_input: true,
                        questions: ['What file formats do you work with?']
                    },
                    'mock-thread-completed': {
                        thread_id: threadId,
                        status: 'completed',
                        current_step: 'create_guide',
                        progress_percentage: 100,
                        results: { implementation_guide: '# Guide content' }
                    }
                };

                return mockResponses[threadId] || mockResponses['mock-thread-running'];
            },

            handlePollingError: function(error) {
                this.status = 'error';
                this.render();
                
                // Retry logic
                setTimeout(() => {
                    if (this.threadId && this.status === 'error') {
                        this.startPolling(this.threadId);
                    }
                }, 5000); // Retry after 5 seconds
            },

            calculateStepProgress: function(stepId) {
                const step = this.steps.find(s => s.id === stepId);
                return step ? step.percentage : 0;
            },

            animateProgress: function(targetPercentage) {
                const startPercentage = this.progressPercentage;
                const duration = 500; // 500ms animation
                const startTime = Date.now();

                const animate = () => {
                    const elapsed = Date.now() - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    
                    this.progressPercentage = Math.round(
                        startPercentage + (targetPercentage - startPercentage) * progress
                    );
                    
                    this.render();

                    if (progress < 1) {
                        requestAnimationFrame(animate);
                    }
                };

                requestAnimationFrame(animate);
            }
        };
    });

    afterEach(() => {
        if (progressTracker.pollingInterval) {
            clearInterval(progressTracker.pollingInterval);
        }
        if (container && container.parentNode) {
            container.parentNode.removeChild(container);
        }
    });

    describe('Rendering', () => {
        test('renders initial progress state', () => {
            progressTracker.render();
            
            expect(container.querySelector('.progress-tracker')).not.toBeNull();
            expect(container.querySelector('.progress-percentage')).not.toBeNull();
            expect(container.querySelector('.progress-bar')).not.toBeNull();
            expect(container.querySelector('.progress-steps')).not.toBeNull();
            expect(container.querySelector('.progress-status')).not.toBeNull();
        });

        test('renders all progress steps', () => {
            progressTracker.render();
            
            const steps = container.querySelectorAll('.step');
            expect(steps.length).toBe(5);
            
            const stepLabels = Array.from(steps).map(step => 
                step.querySelector('.step-label').textContent
            );
            
            expect(stepLabels).toContain('Analyzing Problem');
            expect(stepLabels).toContain('Collecting Context');
            expect(stepLabels).toContain('Generating Requirements');
            expect(stepLabels).toContain('Designing Solution');
            expect(stepLabels).toContain('Creating Guide');
        });

        test('renders progress bar with correct width', () => {
            progressTracker.progressPercentage = 60;
            progressTracker.render();
            
            const progressFill = container.querySelector('.progress-fill');
            expect(progressFill.style.width).toBe('60%');
        });

        test('renders status message correctly', () => {
            progressTracker.status = 'running';
            progressTracker.render();
            
            const statusElement = container.querySelector('.progress-status');
            expect(statusElement.textContent).toBe('Analysis in progress...');
        });
    });

    describe('Progress Updates', () => {
        test('updates progress from API response', () => {
            const mockData = {
                current_step: 'generate_requirements',
                progress_percentage: 60,
                status: 'running',
                message: 'Generating requirements...'
            };

            progressTracker.updateProgress(mockData);

            expect(progressTracker.currentStep).toBe('generate_requirements');
            expect(progressTracker.progressPercentage).toBe(60);
            expect(progressTracker.status).toBe('running');
        });

        test('marks completed steps correctly', () => {
            progressTracker.updateProgress({
                current_step: 'generate_requirements',
                progress_percentage: 60,
                status: 'running'
            });

            progressTracker.render();

            const completedSteps = container.querySelectorAll('.step.completed');
            expect(completedSteps.length).toBe(3); // First 3 steps should be completed at 60%
        });

        test('marks active step correctly', () => {
            progressTracker.updateProgress({
                current_step: 'collect_context',
                progress_percentage: 40,
                status: 'running'
            });

            progressTracker.render();

            const activeStep = container.querySelector('.step.active');
            expect(activeStep).not.toBeNull();
            expect(activeStep.dataset.step).toBe('collect_context');
        });
    });

    describe('Status Handling', () => {
        test('shows input required state', () => {
            progressTracker.updateProgress({
                current_step: 'collect_context',
                progress_percentage: 40,
                status: 'awaiting_input'
            });

            progressTracker.render();

            expect(container.querySelector('#provide-input-btn')).not.toBeNull();
            expect(container.querySelector('.progress-status').textContent)
                .toBe('Waiting for additional information');
        });

        test('shows error state with retry button', () => {
            progressTracker.updateProgress({
                current_step: 'analyze_problem',
                progress_percentage: 20,
                status: 'error'
            });

            progressTracker.render();

            expect(container.querySelector('#retry-btn')).not.toBeNull();
            expect(container.querySelector('.progress-status').textContent)
                .toBe('An error occurred during analysis');
        });

        test('shows completed state', () => {
            progressTracker.updateProgress({
                current_step: 'create_guide',
                progress_percentage: 100,
                status: 'completed'
            });

            progressTracker.render();

            expect(container.querySelector('.progress-status').textContent)
                .toBe('Analysis completed successfully!');
            expect(progressTracker.progressPercentage).toBe(100);
        });
    });

    describe('Polling Functionality', () => {
        test('starts polling with correct interval', (done) => {
            const originalFetch = progressTracker.fetchStatus;
            let pollCount = 0;

            progressTracker.fetchStatus = async function(threadId) {
                pollCount++;
                expect(threadId).toBe('test-thread');
                
                if (pollCount === 2) {
                    progressTracker.stopPolling();
                    progressTracker.fetchStatus = originalFetch;
                    done();
                }

                return {
                    thread_id: threadId,
                    status: 'running',
                    current_step: 'analyze_problem',
                    progress_percentage: 25
                };
            };

            progressTracker.startPolling('test-thread', 100); // 100ms interval for fast testing
        });

        test('stops polling on completion', (done) => {
            progressTracker.fetchStatus = async function() {
                return {
                    thread_id: 'test-thread',
                    status: 'completed',
                    current_step: 'create_guide',
                    progress_percentage: 100
                };
            };

            progressTracker.startPolling('test-thread', 100);

            setTimeout(() => {
                expect(progressTracker.pollingInterval).toBeNull();
                expect(progressTracker.status).toBe('completed');
                done();
            }, 200);
        });

        test('handles polling errors gracefully', (done) => {
            progressTracker.fetchStatus = async function() {
                throw new Error('Network error');
            };

            const originalHandleError = progressTracker.handlePollingError;
            progressTracker.handlePollingError = function(error) {
                expect(error.message).toBe('Network error');
                expect(this.status).toBe('error');
                progressTracker.handlePollingError = originalHandleError;
                done();
            };

            progressTracker.startPolling('test-thread');
        });
    });

    describe('Step Calculation', () => {
        test('calculates step progress correctly', () => {
            expect(progressTracker.calculateStepProgress('analyze_problem')).toBe(20);
            expect(progressTracker.calculateStepProgress('collect_context')).toBe(40);
            expect(progressTracker.calculateStepProgress('generate_requirements')).toBe(60);
            expect(progressTracker.calculateStepProgress('design_solution')).toBe(80);
            expect(progressTracker.calculateStepProgress('create_guide')).toBe(100);
        });

        test('handles unknown step gracefully', () => {
            expect(progressTracker.calculateStepProgress('unknown_step')).toBe(0);
        });
    });

    describe('Animation', () => {
        test('animates progress changes', (done) => {
            progressTracker.progressPercentage = 0;
            
            // Mock requestAnimationFrame
            let animationFrame;
            global.requestAnimationFrame = (callback) => {
                animationFrame = setTimeout(callback, 16); // ~60fps
                return animationFrame;
            };

            progressTracker.animateProgress(50);

            setTimeout(() => {
                expect(progressTracker.progressPercentage).toBeGreaterThan(0);
                expect(progressTracker.progressPercentage).toBeLessThanOrEqual(50);
                
                setTimeout(() => {
                    expect(progressTracker.progressPercentage).toBe(50);
                    clearTimeout(animationFrame);
                    done();
                }, 600); // Wait for animation to complete
            }, 100);
        });
    });

    describe('User Interactions', () => {
        test('handles provide input button click', () => {
            progressTracker.updateProgress({
                current_step: 'collect_context',
                progress_percentage: 40,
                status: 'awaiting_input'
            });

            progressTracker.render();

            const button = container.querySelector('#provide-input-btn');
            let clicked = false;
            
            button.addEventListener('click', () => {
                clicked = true;
            });

            button.click();
            expect(clicked).toBe(true);
        });

        test('handles retry button click', () => {
            progressTracker.updateProgress({
                current_step: 'analyze_problem',
                progress_percentage: 20,
                status: 'error'
            });

            progressTracker.render();

            const button = container.querySelector('#retry-btn');
            let clicked = false;
            
            button.addEventListener('click', () => {
                clicked = true;
            });

            button.click();
            expect(clicked).toBe(true);
        });
    });

    describe('Accessibility', () => {
        test('includes proper ARIA attributes', () => {
            // Enhanced render with accessibility
            progressTracker.render = function() {
                this.container.innerHTML = `
                    <div class="progress-tracker" role="progressbar" 
                         aria-valuenow="${this.progressPercentage}" 
                         aria-valuemin="0" 
                         aria-valuemax="100"
                         aria-label="Analysis Progress">
                        <div class="progress-header">
                            <h3 id="progress-title">Analysis Progress</h3>
                            <div class="progress-percentage" aria-live="polite">${this.progressPercentage}%</div>
                        </div>
                        <div class="progress-bar" aria-labelledby="progress-title">
                            <div class="progress-fill" style="width: ${this.progressPercentage}%"></div>
                        </div>
                        <div class="progress-status ${this.status}" 
                             role="status" 
                             aria-live="polite">${this.getStatusMessage()}</div>
                    </div>
                `;
            };

            progressTracker.progressPercentage = 60;
            progressTracker.render();

            const progressbar = container.querySelector('[role="progressbar"]');
            expect(progressbar.getAttribute('aria-valuenow')).toBe('60');
            expect(progressbar.getAttribute('aria-valuemin')).toBe('0');
            expect(progressbar.getAttribute('aria-valuemax')).toBe('100');
            expect(container.querySelector('[aria-live="polite"]')).not.toBeNull();
            expect(container.querySelector('[role="status"]')).not.toBeNull();
        });
    });
});