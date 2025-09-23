/**
 * Main Application Logic
 * 
 * This file orchestrates the overall application flow, managing transitions
 * between different phases of the problem-solving workflow according to
 * the Vanilla JavaScript Frontend Agent specifications.
 */

class ProblemSolvingApp {
    constructor() {
        this.currentStep = 'problem-input';
        this.threadId = null;
        this.pollingInterval = null;
        this.components = {};
        this.apiClient = new APIClient('http://localhost:8000/api/v1');
        
        this.init();
    }
    
    /**
     * Initialize the application
     */
    init() {
        this.initializeComponents();
        this.setupEventListeners();
        this.showComponent('problem-input-container');
        
        console.log('AI Problem Solving Copilot initialized');
    }
    
    /**
     * Initialize all UI components
     */
    initializeComponents() {
        // Initialize problem input component
        this.components.problemInput = new ProblemInput(
            'problem-input-container',
            this.onProblemSubmit.bind(this)
        );
        
        // Initialize context collector component
        this.components.contextCollector = new ContextCollector(
            'context-collector',
            this.onContextSubmit.bind(this)
        );
        
        // Initialize progress tracker component
        this.components.progressTracker = new ProgressTracker(
            'progress-tracker'
        );
        
        // Initialize document viewer component
        this.components.documentViewer = new DocumentViewer(
            'document-viewer'
        );
    }
    
    /**
     * Setup global event listeners
     */
    setupEventListeners() {
        // Handle browser back/forward buttons
        window.addEventListener('popstate', (e) => {
            if (e.state && e.state.step) {
                this.showComponent(e.state.step);
            }
        });
        
        // Handle page visibility changes for polling optimization
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.pollingInterval) {
                // Reduce polling frequency when tab is hidden
                this.adjustPollingFrequency(5000);
            } else if (!document.hidden && this.pollingInterval) {
                // Restore normal polling frequency when tab is visible
                this.adjustPollingFrequency(2000);
            }
        });
    }
    
    /**
     * Handle problem submission from the input component
     */
    async onProblemSubmit(problemData) {
        try {
            this.showLoadingOverlay('Starting problem analysis...');
            
            // Start analysis via API
            const response = await this.apiClient.startAnalysis(problemData);
            
            this.threadId = response.thread_id;
            this.hideLoadingOverlay();
            
            // Transition to progress tracking
            this.showComponent('progress-tracker');
            this.components.progressTracker.setThreadId(this.threadId);
            this.startProgressPolling();
            
            // Update browser history
            history.pushState({ step: 'progress-tracker' }, '', '#progress');
            
        } catch (error) {
            this.hideLoadingOverlay();
            this.showErrorModal('Failed to start problem analysis: ' + error.message);
        }
    }
    
    /**
     * Handle context submission from the context collector
     */
    async onContextSubmit(contextData) {
        try {
            this.showLoadingOverlay('Processing additional information...');
            
            // Resume workflow with user input
            await this.apiClient.resumeWorkflow(this.threadId, contextData);
            
            this.hideLoadingOverlay();
            
            // Return to progress tracking
            this.showComponent('progress-tracker');
            this.startProgressPolling();
            
        } catch (error) {
            this.hideLoadingOverlay();
            this.showErrorModal('Failed to process information: ' + error.message);
        }
    }
    
    /**
     * Start polling for workflow progress
     */
    startProgressPolling() {
        this.stopProgressPolling(); // Clear any existing polling
        
        this.pollingInterval = setInterval(async () => {
            await this.pollStatus();
        }, 2000); // Poll every 2 seconds
        
        // Initial status check
        this.pollStatus();
    }
    
    /**
     * Poll status implementation following agent specifications
     */
    async pollStatus() {
        try {
            const status = await this.apiClient.getStatus(this.threadId);
            this.handleStatusUpdate(status);
            
        } catch (error) {
            console.error('Status polling error:', error);
            // Continue polling despite errors, but show user feedback
            this.showToast('Connection issue - retrying...', 'warning');
        }
    }
    
    /**
     * Stop progress polling
     */
    stopProgressPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
    
    /**
     * Adjust polling frequency
     */
    adjustPollingFrequency(interval) {
        if (this.pollingInterval) {
            this.stopProgressPolling();
            this.pollingInterval = setInterval(async () => {
                await this.pollStatus();
            }, interval);
        }
    }
    
    /**
     * Handle status updates from the API following agent polling pattern
     */
    handleStatusUpdate(status) {
        // Update progress tracker
        this.components.progressTracker.updateStatus(status);
        
        // Handle different workflow states
        switch (status.status) {
            case 'waiting_for_input':
                this.stopProgressPolling();
                this.handleUserInputRequired(status);
                break;
                
            case 'completed':
                this.stopProgressPolling();
                this.handleWorkflowCompleted(status);
                break;
                
            case 'error':
                this.stopProgressPolling();
                this.showErrorModal('Workflow execution error: ' + (status.message || 'Unknown error'));
                break;
                
            case 'processing':
                // Continue polling
                break;
                
            default:
                // Continue polling for other statuses
                break;
        }
    }
    
    /**
     * Handle when user input is required (HITL workflow)
     */
    handleUserInputRequired(status) {
        // Configure context collector with the agent's questions
        this.components.contextCollector.setQuestions(status.questions || []);
        
        // Transition to context collection
        this.showComponent('context-collector');
        
        // Update browser history
        history.pushState({ step: 'context-collector' }, '', '#context');
    }
    
    /**
     * Handle workflow completion
     */
    async handleWorkflowCompleted(status) {
        try {
            // Display results in document viewer
            this.components.documentViewer.displayResults(status.results);
            
            // Transition to results section
            this.showComponent('document-viewer');
            
            // Update browser history
            history.pushState({ step: 'document-viewer' }, '', '#results');
            
            this.showToast('Analysis completed successfully!', 'success');
            
        } catch (error) {
            this.showErrorModal('Failed to display results: ' + error.message);
        }
    }
    
    /**
     * Show a specific component and hide others
     */
    showComponent(componentId) {
        const components = ['problem-input-container', 'progress-tracker', 'context-collector', 'document-viewer'];
        
        components.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.classList.remove('active');
            }
        });
        
        const targetComponent = document.getElementById(componentId);
        if (targetComponent) {
            targetComponent.classList.add('active');
        }
        
        this.currentStep = componentId;
        
        // Initialize component if needed
        if (this.components[componentId.replace('-container', '').replace('-', '')]) {
            const component = this.components[componentId.replace('-container', '').replace('-', '')];
            if (component.onShow) {
                component.onShow();
            }
        }
    }
    
    /**
     * Show loading overlay
     */
    showLoadingOverlay(message = 'Processing...') {
        let overlay = document.getElementById('loading-overlay');
        
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loading-overlay';
            overlay.className = 'loading-overlay';
            overlay.innerHTML = `
                <div class="loading-spinner">
                    <div class="spinner"></div>
                    <p>${message}</p>
                </div>
            `;
            document.body.appendChild(overlay);
        } else {
            const messageElement = overlay.querySelector('p');
            if (messageElement) {
                messageElement.textContent = message;
            }
            overlay.classList.remove('hidden');
        }
    }
    
    /**
     * Hide loading overlay
     */
    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }
    
    /**
     * Show error modal
     */
    showErrorModal(message) {
        let modal = document.getElementById('error-modal');
        
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'error-modal';
            modal.className = 'error-modal';
            modal.innerHTML = `
                <div class="error-modal-content">
                    <button class="error-modal-close">&times;</button>
                    <h3>Error</h3>
                    <p id="error-message">${message}</p>
                    <button id="error-ok-btn" class="btn btn-primary">OK</button>
                </div>
            `;
            document.body.appendChild(modal);
            
            // Setup event listeners
            modal.querySelector('.error-modal-close').addEventListener('click', () => this.hideErrorModal());
            modal.querySelector('#error-ok-btn').addEventListener('click', () => this.hideErrorModal());
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideErrorModal();
                }
            });
        } else {
            const messageElement = modal.querySelector('#error-message');
            if (messageElement) {
                messageElement.textContent = message;
            }
            modal.classList.remove('hidden');
        }
    }
    
    /**
     * Hide error modal
     */
    hideErrorModal() {
        const modal = document.getElementById('error-modal');
        if (modal) {
            modal.classList.add('hidden');
        }
    }
    
    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 3000);
    }
    
    /**
     * Reset the application to initial state
     */
    reset() {
        this.stopProgressPolling();
        this.threadId = null;
        this.currentStep = 'problem-input';
        
        // Reset all components
        Object.values(this.components).forEach(component => {
            if (component.reset) {
                component.reset();
            }
        });
        
        // Show initial component
        this.showComponent('problem-input-container');
        
        // Clear browser history
        history.replaceState({ step: 'problem-input-container' }, '', '#');
    }
    
    /**
     * Get current application state
     */
    getState() {
        return {
            currentStep: this.currentStep,
            threadId: this.threadId,
            isPolling: !!this.pollingInterval
        };
    }
}

// Global application instance
let app;

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    app = new ProblemSolvingApp();
    
    // Expose app globally for debugging
    window.ProblemSolvingApp = app;
    
    console.log('AI Problem Solving Copilot started successfully');
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (app) {
        app.stopProgressPolling();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProblemSolvingApp;
}