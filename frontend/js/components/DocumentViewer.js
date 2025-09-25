/**
 * 문서 뷰어 컴포넌트 (비개발자용 안내)
 *
 * 생성된 문서(요구사항, 사용자 여정, 구현 가이드, 기술 추천서)를 탭으로 보여줍니다.
 * - 미리보기/복사/인쇄/다운로드/전체 다운로드 기능을 제공합니다.
 * - 진행 중에는 상단 박스가 단계별 상태(분석/수집/생성 중)를 안내합니다.
 */

class DocumentViewerComponent {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.documents = {};
        this.currentDocument = null;
        this.solutionInfo = {};
        
        if (!this.container) {
            throw new Error(`Container with ID '${containerId}' not found`);
        }
        
        this.render();
        this.setupEventListeners();
        // Default header: processing state until completion
        if (typeof this.setHeaderState === 'function') {
            this.setHeaderState(false);
        }
    }
    
    /**
     * Render the component HTML
     */
    render() {
        this.container.innerHTML = `
            <div class="document-viewer">
                <div class="results-header">
                    <div class="completion-status">
                        <span class="status-icon">✅</span>
                        <h3>분석 완료!</h3>
                        <p>문제 분석이 성공적으로 완료되었습니다. 아래에서 결과를 확인하세요.</p>
                    </div>
                    
                    <div id="solution-summary" class="solution-summary hidden">
                        <div class="summary-item">
                            <label>솔루션 유형:</label>
                            <span id="solution-type">-</span>
                        </div>
                        <div class="summary-item">
                            <label>추천 기술 스택:</label>
                            <div id="tech-stack" class="tech-stack"></div>
                        </div>
                    </div>
                </div>
                
                <div class="document-tabs">
                    <button class="tab-button active" data-doc="requirements">
                        📋 요구사항 명세서
                    </button>
                    <button class="tab-button" data-doc="journey">
                        🗺️ 사용자 여정 지도
                    </button>
                    <button class="tab-button" data-doc="guide">
                        📚 구현 가이드
                    </button>
                    <button class="tab-button" data-doc="tech">
                        🔧 기술 추천서
                    </button>
                </div>
                
                <div class="document-content-container">
                    <div id="document-content" class="document-content">
                        <div class="loading-state">
                            <p>문서를 불러오는 중...</p>
                        </div>
                    </div>
                    
                    <div class="document-actions">
                        <button type="button" class="btn btn-outline" id="back-to-progress-btn">
                            ↩️ 진행 화면으로 돌아가기
                        </button>
                        <button type="button" class="btn btn-primary" id="download-btn">
                            📥 다운로드
                        </button>
                        
                        <button type="button" class="btn btn-outline" id="copy-btn">
                            📋 복사
                        </button>
                        
                        <button type="button" class="btn btn-outline" id="print-btn">
                            🖨️ 인쇄
                        </button>
                        
                        <div class="action-group">
                            <button type="button" class="btn btn-outline" id="download-all-btn">
                                📦 전체 다운로드
                            </button>
                            
                            <button type="button" class="btn btn-outline" id="new-analysis-btn">
                                🔄 새 분석 시작
                            </button>
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
        const tabButtons = this.container.querySelectorAll('.tab-button');
        const downloadBtn = this.container.querySelector('#download-btn');
        const copyBtn = this.container.querySelector('#copy-btn');
        const printBtn = this.container.querySelector('#print-btn');
        const downloadAllBtn = this.container.querySelector('#download-all-btn');
        const newAnalysisBtn = this.container.querySelector('#new-analysis-btn');
        const backBtn = this.container.querySelector('#back-to-progress-btn');
        
        // Tab switching
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const docType = button.dataset.doc;
                this.switchDocument(docType);
                
                // Update active tab
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
            });
        });
        
        // Document actions
        downloadBtn.addEventListener('click', () => this.downloadCurrentDocument());
        copyBtn.addEventListener('click', () => this.copyCurrentDocument());
        printBtn.addEventListener('click', () => this.printCurrentDocument());
        downloadAllBtn.addEventListener('click', () => this.downloadAllDocuments());
        newAnalysisBtn.addEventListener('click', () => this.startNewAnalysis());
        if (backBtn) {
            backBtn.addEventListener('click', () => this.backToProgress());
        }
    }

    /**
     * Update header state (processing vs completed)
     */
    setHeaderState(isComplete = false) {
        const box = this.container.querySelector('.completion-status');
        if (!box) return;
        const title = box.querySelector('h3');
        const desc = box.querySelector('p');
        const icon = box.querySelector('.status-icon');

        if (isComplete) {
            box.setAttribute('data-state', 'completed');
            if (icon) icon.textContent = '✔️';
            if (title) title.textContent = '분석 완료!';
            if (desc) desc.textContent = '분석이 완료되었습니다. 아래에서 결과를 확인하세요.';
        } else {
            box.setAttribute('data-state', 'processing');
            if (icon) icon.textContent = '⏳';
            if (title) title.textContent = '분석 진행 중...';
            if (desc) desc.textContent = '문서가 순차적으로 생성됩니다. 준비되는 대로 표시합니다.';
        }
    }

    /**
     * Update header with specific stage messages
     */
    setHeaderByStatus(status = '') {
        const s = String(status || '').toLowerCase();
        if (s === 'completed') {
            this.setHeaderState(true);
            return;
        }
        // Map status to Korean messages
        const stageMap = {
            'analyzing': '문제 분석 중...',
            'collecting_context': '컨텍스트 수집 중...',
            'awaiting_input': '사용자 입력 대기 중...',
            'generating_requirements': '요구사항 생성 중...',
            'designing_solution': '솔루션 설계 중...',
            'creating_guide': '가이드 생성 중...'
        };
        const msg = stageMap[s] || '분석 진행 중...';
        const box = this.container.querySelector('.completion-status');
        if (!box) return;
        box.setAttribute('data-state', 'processing');
        const title = box.querySelector('h3');
        const desc = box.querySelector('p');
        const icon = box.querySelector('.status-icon');
        if (icon) icon.textContent = '⏳';
        if (title) title.textContent = msg;
        if (desc) desc.textContent = '문서가 순차적으로 생성됩니다. 준비되는 대로 표시합니다.';
    }
    
    /**
     * Display analysis results
     */
    displayResults(results) {
        this.documents = results || {};
        this.solutionInfo = {
            type: results.solution_type,
            techStack: results.tech_stack || []
        };

        // Mark header as completed
        this.setHeaderState(true);

        // Update solution summary
        this.updateSolutionSummary();

        // Update tab indicators
        this.updateTabIndicators();

        // Display first document
        this.switchDocument('requirements');
    }

    /**
     * CRITICAL FIX: Update partial results in real-time
     */
    updatePartialResults(partialResults, isComplete = false) {
        if (!partialResults) return;

        // Merge partial results with existing documents
        this.documents = { ...this.documents, ...partialResults };

        // Only update header when completed; otherwise keep any stage-specific header
        if (isComplete && typeof this.setHeaderState === 'function') {
            this.setHeaderState(true);
        }

        // Update solution info if available
        if (partialResults.solution_type) {
            this.solutionInfo.type = partialResults.solution_type;
        }
        if (partialResults.tech_stack) {
            this.solutionInfo.techStack = partialResults.tech_stack;
        }

        // Update solution summary
        this.updateSolutionSummary();

        // Update tab indicators to show which documents are available
        this.updateTabIndicators();

        // If we're currently viewing a document that was just updated, refresh it
        if (this.currentDocument) {
            const documentMap = {
                'requirements': 'requirements_document',
                'journey': 'user_journey_map',
                'guide': 'implementation_guide',
                'tech': 'tech_recommendations'
            };

            const documentKey = documentMap[this.currentDocument];
            if (partialResults[documentKey] && this.currentDocument) {
                this.switchDocument(this.currentDocument);
            }
        } else {
            // If no document is currently selected, show the first available one
            const availableDocs = ['requirements', 'journey', 'guide', 'tech'];
            const documentMap = {
                'requirements': 'requirements_document',
                'journey': 'user_journey_map',
                'guide': 'implementation_guide',
                'tech': 'tech_recommendations'
            };

            for (const docType of availableDocs) {
                if (this.documents[documentMap[docType]]) {
                    this.switchDocument(docType);
                    break;
                }
            }
        }

        console.log('[DOCUMENT VIEWER] Updated with partial results:', Object.keys(partialResults));
    }

    /**
     * Update tab indicators to show document availability status
     */
    updateTabIndicators() {
        const documentMap = {
            'requirements': 'requirements_document',
            'journey': 'user_journey_map',
            'guide': 'implementation_guide',
            'tech': 'tech_recommendations'
        };

        const tabButtons = this.container.querySelectorAll('.tab-button');

        tabButtons.forEach(button => {
            const docType = button.dataset.doc;
            const documentKey = documentMap[docType];

            // Remove existing status classes
            button.classList.remove('available', 'generating', 'pending');

            if (this.documents[documentKey]) {
                // Document is available
                button.classList.add('available');
                button.style.opacity = '1';

                // Add visual indicator
                if (!button.querySelector('.status-indicator')) {
                    const indicator = document.createElement('span');
                    indicator.className = 'status-indicator';
                    indicator.textContent = '✅';
                    indicator.style.cssText = 'margin-left: 5px; font-size: 12px;';
                    button.appendChild(indicator);
                }
            } else {
                // Document not yet available
                button.classList.add('pending');
                button.style.opacity = '0.6';

                // Add pending indicator
                if (!button.querySelector('.status-indicator')) {
                    const indicator = document.createElement('span');
                    indicator.className = 'status-indicator';
                    indicator.textContent = '⏳';
                    indicator.style.cssText = 'margin-left: 5px; font-size: 12px;';
                    button.appendChild(indicator);
                } else {
                    const indicator = button.querySelector('.status-indicator');
                    indicator.textContent = '⏳';
                }
            }
        });
    }
    
    /**
     * Update solution summary section
     */
    updateSolutionSummary() {
        const summaryElement = this.container.querySelector('#solution-summary');
        const solutionTypeElement = this.container.querySelector('#solution-type');
        const techStackElement = this.container.querySelector('#tech-stack');
        
        if (this.solutionInfo.type) {
            solutionTypeElement.textContent = this.getSolutionTypeLabel(this.solutionInfo.type);
            
            if (this.solutionInfo.techStack && this.solutionInfo.techStack.length > 0) {
                techStackElement.innerHTML = this.solutionInfo.techStack
                    .map(tech => {
                        if (typeof tech === 'string') return `<span class="tech-tag">${tech}</span>`;
                        if (tech && typeof tech === 'object') {
                            const label = tech.technology || tech.name || tech.layer || '기술';
                            return `<span class="tech-tag">${label}</span>`;
                        }
                        return '';
                    })
                    .join('');
            } else {
                techStackElement.textContent = '정보 없음';
            }
            
            summaryElement.classList.remove('hidden');
        }
    }
    
    /**
     * Get human-readable solution type label
     */
    getSolutionTypeLabel(type) {
        const labels = {
            'SIMPLE_AUTOMATION': '기본 자동화',
            'COMPLEX_AUTOMATION': '고급 자동화',
            'RAG': '검색 증강 생성',
            'ADVANCED_RAG': '고급 RAG',
            'ML_CLASSIFICATION': '머신러닝 분류',
            'DASHBOARD': '대시보드',
            'API_INTEGRATION': 'API 통합',
            'RAG_AUTOMATION': 'RAG + 자동화',
            'AUTOMATION_DASHBOARD': '자동화 + 대시보드',
            'ML_RAG': '머신러닝 + RAG'
        };
        
        return labels[type] || type;
    }
    
    /**
     * Switch to a different document
     */
    switchDocument(docType) {
        this.currentDocument = docType;
        
        const contentElement = this.container.querySelector('#document-content');
        
        const documentMap = {
            'requirements': 'requirements_document',
            'journey': 'user_journey_map', 
            'guide': 'implementation_guide',
            'tech': 'tech_recommendations'
        };
        
        const documentKey = documentMap[docType];
        let documentContent = this.documents[documentKey];
        
        // Normalize non-string content (e.g., tech_recommendations object)
        if (documentKey === 'tech_recommendations') {
            documentContent = this.formatTechRecommendations(documentContent);
        } else if (typeof documentContent !== 'string') {
            documentContent = this.stringifyIfObject(documentContent);
        }
        
        if (documentContent) {
            // Render markdown content with proper handling
            let htmlContent;
            if (window.markdownRenderer && typeof window.markdownRenderer.render === 'function') {
                try {
                    htmlContent = `<div class="markdown-content">${window.markdownRenderer.render(documentContent)}</div>`;
                } catch (error) {
                    console.warn('MarkdownRenderer failed, using fallback:', error);
                    htmlContent = `<div class="markdown-content"><pre style="white-space: pre-wrap; font-family: 'Courier New', monospace; background: #f8f9fa; padding: 20px; border-radius: 8px; line-height: 1.5;">${documentContent}</pre></div>`;
                }
            } else {
                // Fallback to pre-formatted text
                htmlContent = `<div class="markdown-content"><pre style="white-space: pre-wrap; font-family: 'Courier New', monospace; background: #f8f9fa; padding: 20px; border-radius: 8px; line-height: 1.5;">${documentContent}</pre></div>`;
            }
                
            contentElement.innerHTML = htmlContent;
            
            // Initialize syntax highlighting if available
            if (window.hljs) {
                contentElement.querySelectorAll('pre code').forEach(block => {
                    window.hljs.highlightElement(block);
                });
            }
        } else {
            contentElement.innerHTML = `
                <div class="no-content">
                    <p>⚠️ 이 문서는 아직 생성되지 않았습니다.</p>
                    <p>분석이 완료되지 않았거나 오류가 발생했을 수 있습니다.</p>
                </div>
            `;
        }
        
        // Scroll to top
        contentElement.scrollTop = 0;
    }
    
    /**
     * Download current document
     */
    downloadCurrentDocument() {
        if (!this.currentDocument) return;
        
        const documentMap = {
            'requirements': { key: 'requirements_document', filename: '요구사항_명세서.md' },
            'journey': { key: 'user_journey_map', filename: '사용자_여정_지도.md' },
            'guide': { key: 'implementation_guide', filename: '구현_가이드.md' },
            'tech': { key: 'tech_recommendations', filename: '기술_추천서.md' }
        };
        
        const docInfo = documentMap[this.currentDocument];
        let content = this.documents[docInfo.key];
        if (docInfo.key === 'tech_recommendations') {
            content = this.formatTechRecommendations(content);
        } else if (typeof content !== 'string') {
            content = this.stringifyIfObject(content);
        }
        
        if (content) {
            this.downloadBlob(new Blob([content], { type: 'text/markdown' }), docInfo.filename);
        }
    }
    
    /**
     * Copy current document to clipboard
     */
    async copyCurrentDocument() {
        if (!this.currentDocument) return;
        
        const documentMap = {
            'requirements': 'requirements_document',
            'journey': 'user_journey_map',
            'guide': 'implementation_guide', 
            'tech': 'tech_recommendations'
        };
        
        const documentKey = documentMap[this.currentDocument];
        let content = this.documents[documentKey];
        if (documentKey === 'tech_recommendations') {
            content = this.formatTechRecommendations(content);
        } else if (typeof content !== 'string') {
            content = this.stringifyIfObject(content);
        }
        
        if (content) {
            try {
                await navigator.clipboard.writeText(content);
                this.showToast('문서가 클립보드에 복사되었습니다!', 'success');
            } catch (err) {
                this.showToast('복사에 실패했습니다. 브라우저가 지원하지 않을 수 있습니다.', 'error');
            }
        }
    }

    /**
     * Convert tech recommendations object to readable Markdown
     */
    formatTechRecommendations(data) {
        try {
            if (!data) return '';
            if (typeof data === 'string') return data;

            const lines = [];
            lines.push('# 기술추천서');

            // 요약 섹션
            const summary = [];
            if (this.solutionInfo && this.solutionInfo.type) {
                summary.push(`- 추천 솔루션 유형: ${this.solutionInfo.type}`);
            }
            if (data.complexity) summary.push(`- 복잡도: ${data.complexity}`);
            if (data.domain) summary.push(`- 도메인: ${data.domain}`);
            if (data.constraints && Array.isArray(data.constraints) && data.constraints.length) {
                summary.push(`- 주요 제약: ${data.constraints.slice(0,3).map(c => typeof c === 'string' ? c : (c.description || c.type)).join(', ')}`);
            }
            if (summary.length) {
                lines.push('', '## 요약', ...summary);
            }

            // 권장 아키텍처 구성 (테이블)
            const stack = Array.isArray(data.recommended_stack) ? data.recommended_stack : null;
            if (stack && stack.length) {
                lines.push('', '## 권장 아키텍처 구성', '', '| 계층/구성요소 | 기술 | 근거 |', '|---|---|---|');
                stack.forEach(s => {
                    if (typeof s === 'string') {
                        lines.push(`| - | ${s} | - |`);
                    } else if (s && typeof s === 'object') {
                        const layer = s.layer || s.category || s.component || '-';
                        const tech = s.technology || s.name || '-';
                        const reason = s.reason || s.rationale || s.justification || '';
                        lines.push(`| ${layer} | ${tech} | ${reason} |`);
                    }
                });
            }

            // 추천 기술 (주요/보조)
            const primary = Array.isArray(data.primary_technologies) ? data.primary_technologies : null;
            const secondary = Array.isArray(data.secondary_technologies) ? data.secondary_technologies : null;
            const asList = Array.isArray(data) ? data : null;

            const renderTechList = (title, list) => {
                if (!list || !list.length) return;
                lines.push('', `## ${title}`);
                list.forEach((t, idx) => {
                    if (typeof t === 'string') {
                        lines.push(`- **${t}**`);
                    } else if (t && typeof t === 'object') {
                        const name = t.technology || t.name || `항목 ${idx+1}`;
                        const version = t.version ? ` (권장 버전: ${t.version})` : '';
                        const why = t.reason || t.rationale || t.justification || '';
                        lines.push(`- **${name}**${version}${why ? ` — ${why}` : ''}`);
                        if (Array.isArray(t.alternatives) && t.alternatives.length) {
                            lines.push(`  - 대안: ${t.alternatives.join(', ')}`);
                        }
                        if (t.license) lines.push(`  - 라이선스: ${t.license}`);
                    }
                });
            };

            if (primary || secondary || asList) {
                renderTechList('주요기술', primary || asList);
                if (secondary) renderTechList('보조기술', secondary);
            }

            // 통합/아키텍처/접점
            if (data.integration_points || data.architecture_overview || data.implementation_approach) {
                lines.push('', '## 통합 및 운영 고려사항');
                if (Array.isArray(data.integration_points)) {
                    data.integration_points.forEach(p => lines.push(`- 통합: ${typeof p === 'string' ? p : JSON.stringify(p)}`));
                }
                if (data.architecture_overview) {
                    lines.push('', '### 아키텍처 개요', '', typeof data.architecture_overview === 'string' ? data.architecture_overview : '');
                }
                if (data.implementation_approach) {
                    lines.push('', '### 구현 접근 방식', '', typeof data.implementation_approach === 'string' ? data.implementation_approach : '');
                }
            }

            // 선정 근거/비고
            if (data.rationale || data.notes) {
                if (data.rationale) lines.push('', '## 선정 근거', '', typeof data.rationale === 'string' ? data.rationale : '');
                if (data.notes) lines.push('', '## 비고', '', typeof data.notes === 'string' ? data.notes : '');
            }

            // 정보가 너무 적을 경우 JSON 표시
            if (lines.length <= 4) {
                return '```json\n' + JSON.stringify(data, null, 2) + '\n```';
            }
            return lines.join('\n');
        } catch (e) {
            return String(data ?? '');
        }
    }

    /**
     * Stringify generic objects safely to Markdown code block
     */
    stringifyIfObject(value) {
        if (value == null) return '';
        if (typeof value === 'string') return value;
        try {
            return '```json\n' + JSON.stringify(value, null, 2) + '\n```';
        } catch {
            return String(value);
        }
    }
    
    /**
     * Print current document
     */
    printCurrentDocument() {
        const contentElement = this.container.querySelector('#document-content');
        const content = contentElement.innerHTML;
        
        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>분석 결과 - ${this.getDocumentTitle()}</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
                    .markdown-content { max-width: none; }
                    pre { background: #f5f5f5; padding: 1rem; border-radius: 4px; }
                    code { background: #f0f0f0; padding: 0.2rem 0.4rem; border-radius: 2px; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                ${content}
            </body>
            </html>
        `);
        printWindow.document.close();
        printWindow.print();
    }
    
    /**
     * Download all documents
     */
    downloadAllDocuments() {
        Object.keys(this.documents).forEach((key, index) => {
            setTimeout(() => {
                const docType = this.getDocTypeFromKey(key);
                if (docType) {
                    this.currentDocument = docType;
                    this.downloadCurrentDocument();
                }
            }, index * 500);
        });
        
        this.showToast('모든 문서 다운로드를 시작합니다.', 'info');
    }
    
    /**
     * Get document type from key
     */
    getDocTypeFromKey(key) {
        const mapping = {
            'requirements_document': 'requirements',
            'user_journey_map': 'journey',
            'implementation_guide': 'guide',
            'tech_recommendations': 'tech'
        };
        return mapping[key];
    }
    
    /**
     * Download blob as file
     */
    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.style.display = 'none';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
    }
    
    /**
     * Get current document title
     */
    getDocumentTitle() {
        const titles = {
            'requirements': '요구사항_명세서',
            'journey': '사용자_여정_지도',
            'guide': '구현_가이드',
            'tech': '기술_추천서'
        };
        
        return titles[this.currentDocument] || '문서';
    }
    
    /**
     * Start new analysis
     */
    startNewAnalysis() {
        if (confirm('새로운 분석을 시작하면 현재 결과가 사라집니다. 계속하시겠습니까?')) {
            if (window.ProblemSolvingApp) {
                window.ProblemSolvingApp.reset();
            }
        }
    }

    /**
     * Return to the progress tracker view
     */
    backToProgress() {
        const app = window.ProblemSolvingApp;
        if (!app) return;
        app.showComponent('progress-tracker');
        try { history.pushState({ step: 'progress-tracker' }, '', '#progress'); } catch (e) {}
    }
    
    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
            color: white;
            padding: 12px 16px;
            border-radius: 8px;
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            opacity: 0;
            transition: opacity 0.3s ease;
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => { toast.style.opacity = '1'; }, 100);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => {
                if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                }
            }, 300);
        }, 3000);
    }
    
    /**
     * Reset component
     */
    reset() {
        this.documents = {};
        this.currentDocument = null;
        this.solutionInfo = {};
        
        this.render();
        this.setupEventListeners();
    }
}

// Export for global use
window.DocumentViewerComponent = DocumentViewerComponent;
