// ═══════════════════════════════════════════════════════════
//  CONFIGURATION
// ═══════════════════════════════════════════════════════════
const API_BASE = 'http://localhost:5000';

// ═══════════════════════════════════════════════════════════
//  GLOBAL STATE
// ═══════════════════════════════════════════════════════════
let operationsData = null;       // Raw uploaded JSON
let optimizedData = null;       // Baseline solution from optimizer
let rescheduledData = null;      // Latest reschedule result
let currentWorkflow = null;
let isOptimized = false;         // Whether optimizer was run
let lastChartFolder = null;      // folder for reschedule charts

function getOperationsList(data) {
    if (!data) return [];
    if (Array.isArray(data)) return data;
    return data.assignments || data.workOrderOperationDtoList || data.operations || data.items || data.data || [];
}

// ═══════════════════════════════════════════════════════════
//  TAILWIND HELPERS  (replaces style.display toggling)
// ═══════════════════════════════════════════════════════════
function show(el) {
    if (!el) return;
    el.classList.remove('hidden');
    // For flex elements that were hidden, restore flex display
    if (el.dataset.flex === 'true') {
        el.style.display = 'flex';
    }
}
function hide(el) {
    if (!el) return;
    el.classList.add('hidden');
    el.style.display = '';
}
function showFlex(el) {
    if (!el) return;
    el.classList.remove('hidden');
    el.style.display = 'flex';
}
function showInlineFlex(el) {
    if (!el) return;
    el.classList.remove('hidden');
    el.style.display = 'inline-flex';
}

// ═══════════════════════════════════════════════════════════
//  DOM ELEMENTS
// ═══════════════════════════════════════════════════════════
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileStats = document.getElementById('fileStats');
const viewDataBtn = document.getElementById('viewDataBtn');
const viewStationsBtn = document.getElementById('viewStationsBtn');

const actionsSection = document.getElementById('actionsSection');
const downloadBtn = document.getElementById('downloadBtn');
const rescheduleBtn = document.getElementById('rescheduleBtn');
const viewGanttBtn = document.getElementById('viewGanttBtn');

const rescheduleTypeSection = document.getElementById('rescheduleTypeSection');
const rescheduleCards = document.querySelectorAll('.reschedule-card');
const workflowSection = document.getElementById('workflowSection');
const workflowTitle = document.getElementById('workflowTitle');
const processingSection = document.getElementById('processingSection');
const processingText = document.getElementById('processingText');
const resultsSection = document.getElementById('resultsSection');
const resultsMessage = document.getElementById('resultsSubtitle');
const downloadResultBtn = document.getElementById('downloadJsonBtn');
const viewRescheduleGanttBtn = document.getElementById('viewRescheduleGanttBtn');
// resetBtn uses onclick in HTML, no variable needed

const dataModal = document.getElementById('dataModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const modalClose = document.getElementById('modalClose');

// Optimization modals
const optimizeModal = document.getElementById('optimizeModal');
const optimizingModal = document.getElementById('optimizingModal');
const optimizingStatus = document.getElementById('optimizingStatus');
const optimizerProgressFill = document.getElementById('optimizerProgressFill');
const optimizeYesBtn = document.getElementById('optimizeYesBtn');
const optimizeNoBtn = document.getElementById('optimizeNoBtn');

// Gantt modal
const ganttModal = document.getElementById('ganttModal');
const ganttModalTitle = document.getElementById('ganttModalTitle');
const ganttModalBody = document.getElementById('ganttModalBody');
const ganttModalClose = document.getElementById('ganttModalClose');

// Urgent job action modal
const urgentJobActionModal = document.getElementById('urgentJobActionModal');
const navReschedule = document.getElementById('navReschedule');
const navActions = document.getElementById('navActions');

// ═══════════════════════════════════════════════════════════
//  INITIALIZE
// ═══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    setupUploadHandlers();
    setupWorkflowHandlers();
    setupModalHandlers();
    setupOptimizeModalHandlers();
    setupGanttModalHandlers();
    setupUrgentJobActionModal();
    setupDownloadResultHandler();
    checkServerHealth();
    if (navReschedule) {
        navReschedule.addEventListener('click', (e) => {
            e.preventDefault();
            showRescheduleOptions();
        });
    }
    if (navActions) {
        navActions.addEventListener('click', (e) => {
            e.preventDefault();
            if (!operationsData) {
                alert('Önce bir veri dosyası yükleyin.');
                return;
            }
            showActionsSection();
        });
    }
});

// ═══════════════════════════════════════════════════════════
//  SERVER HEALTH CHECK
// ═══════════════════════════════════════════════════════════
async function checkServerHealth() {
    const serverDot = document.getElementById('serverDot');
    const serverLabel = document.getElementById('serverLabel');
    if (!serverLabel) return;
    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
            serverLabel.textContent = 'Sunucu: çevrimiçi';
            serverDot.className = 'size-2 rounded-full bg-green-500 shadow shadow-green-500/60';
        } else {
            throw new Error('not ok');
        }
    } catch {
        serverLabel.textContent = 'Sunucu: çevrimdışı';
        serverDot.className = 'size-2 rounded-full bg-red-500 shadow shadow-red-500/40';
    }
}

// ═══════════════════════════════════════════════════════════
//  UPLOAD HANDLERS
// ═══════════════════════════════════════════════════════════
function setupUploadHandlers() {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover-active');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover-active'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover-active');
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/json') handleFile(file);
    });

    viewDataBtn.addEventListener('click', () => showDataModal('Makineler', extractMachines()));
    viewStationsBtn.addEventListener('click', () => showDataModal('İstasyonlar', extractStations()));
    if (downloadBtn) downloadBtn.addEventListener('click', downloadCurrentData);
    if (rescheduleBtn) {
        rescheduleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            showRescheduleOptions();
        });
    }
    if (viewGanttBtn) viewGanttBtn.addEventListener('click', () => showGanttCharts('baseline'));
    // resetBtn uses onclick="resetApplication()" in HTML — no addEventListener needed
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) handleFile(file);
}

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);
            if (validateOperationsData(data)) {
                operationsData = data;
                displayFileInfo(file, data);
                showOptimizePopup(data);
            } else {
                alert('Geçersiz JSON yapısı. Lütfen dosyanın geçerli operasyon verileri içerdiğinden emin olun.');
            }
        } catch (error) {
            alert('JSON dosyası ayrıştırma hatası: ' + error.message);
        }
    };
    reader.readAsText(file);
}

function validateOperationsData(data) {
    if (data.hasOwnProperty('assignments')) {
        const assignments = data.assignments;
        if (!Array.isArray(assignments)) return false;
        return assignments.length > 0;
    }
    if (data.hasOwnProperty('workOrderOperationDtoList')) {
        const operations = data.workOrderOperationDtoList;
        if (!Array.isArray(operations)) return false;
        return operations.length > 0;
    }
    const operations = getOperationsList(data);
    if (!operations || !Array.isArray(operations)) return false;
    if (operations.length === 0) return false;
    const sample = operations[0];
    return sample.hasOwnProperty('id') && sample.hasOwnProperty('objectType');
}

function displayFileInfo(file, data) {
    const ops = data.assignments || data.workOrderOperationDtoList || (Array.isArray(data) ? data : data.operations) || data;
    const opCount = Array.isArray(ops) ? ops.length : 0;

    // Update header chip
    const chip = document.getElementById('headerFileChip');
    const chipName = document.getElementById('headerFileName');
    if (chip && chipName) {
        chipName.textContent = file.name;
        showFlex(chip);
    }

    fileName.textContent = file.name;
    fileStats.textContent = `${opCount} operasyon · ${(file.size / 1024).toFixed(2)} KB`;
    showFlex(fileInfo);
    showInlineFlex(viewDataBtn);
    showInlineFlex(viewStationsBtn);
}

// ═══════════════════════════════════════════════════════════
//  OPTIMIZATION POPUP
// ═══════════════════════════════════════════════════════════
function showOptimizePopup(data) {
    const ops = getOperationsList(data);
    const statsEl = document.getElementById('optimizeStats');
    if (statsEl) {
        statsEl.textContent = `📦 ${Array.isArray(ops) ? ops.length : '?'} operasyon yüklendi`;
    }
    showFlex(optimizeModal);
}

function setupOptimizeModalHandlers() {
    optimizeYesBtn.addEventListener('click', async () => {
        hide(optimizeModal);
        await runOptimization();
    });

    optimizeNoBtn.addEventListener('click', () => {
        hide(optimizeModal);
        isOptimized = false;
        hide(viewGanttBtn);
        showActionsSection();
    });
}

async function runOptimization() {
    if (!operationsData) { alert('Lütfen önce bir JSON dosyası yükleyin.'); return; }
    hide(optimizeModal);
    showFlex(optimizingModal);
    // Kick off visual progress bar — no callback so it doesn't interfere with fetch
    animateProgress(90, 'Optimizasyon çalışıyor...', null);

    try {
        const response = await fetch(`${API_BASE}/optimize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(operationsData),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ error: 'Server error' }));
            throw new Error(err.error || `Sunucu hatası: ${response.status}`);
        }

        const result = await response.json();
        optimizedData = result.baseline;
        isOptimized = true;
        lastChartFolder = result.chart_folder || null;

        if (optimizerProgressFill) optimizerProgressFill.style.width = '100%';
        if (optimizingStatus) optimizingStatus.textContent = `✅ Tamamlandı! ${result.chart_count} Gantt grafiği oluşturuldu.`;
        await sleep(1200);

        hide(optimizingModal);
        if (lastChartFolder) {
            showInlineFlex(viewGanttBtn);
            viewGanttBtn.style.display = 'flex';
        }
        showActionsSection();
        // Show KPI for baseline right away in results section
        show(resultsSection);
        document.getElementById('resultsSubtitle').textContent =
            `Optimizasyon tamamlandı — ${result.chart_count} Gantt grafiği oluşturuldu.`;
        renderKpiDashboard(optimizedData);
        checkServerHealth();

    } catch (err) {
        hide(optimizingModal);
        const isNetworkError = (err instanceof TypeError) ||
            err.message.includes('fetch') ||
            err.message.includes('Failed') ||
            err.message.includes('NetworkError') ||
            err.message.includes('ERR_CONNECTION');

        if (isNetworkError) {
            const proceed = confirm(
                `⚠️ Optimizasyon sunucusuna (${API_BASE}) bağlanılamadı.\n\n` +
                `server.py'nin çalıştığından emin olun:\n  python server.py\n\n` +
                `Optimizasyonsuz devam edilsin mi?`
            );
            if (proceed) {
                isOptimized = false;
                showActionsSection();
            }
        } else {
            alert('Optimizasyon hatası: ' + err.message);
            isOptimized = false;
            showActionsSection();
        }
        checkServerHealth();
    }
}

let progressInterval = null;
function animateProgress(targetPct, statusText, onComplete) {
    const steps = [
        { pct: 20, text: 'Veri doğrulanıyor...' },
        { pct: 45, text: 'GT sezgiseli çalıştırılıyor...' },
        { pct: 70, text: 'ATC sıralaması uygulanıyor...' },
        { pct: 90, text: 'Gantt grafikleri oluşturuluyor...' },
        { pct: targetPct, text: statusText },
    ];
    let i = 0;
    const tick = () => {
        if (i >= steps.length) { if (onComplete) onComplete(); return; }
        const s = steps[i++];
        if (optimizerProgressFill) optimizerProgressFill.style.width = s.pct + '%';
        if (optimizingStatus) optimizingStatus.textContent = s.text;
        setTimeout(tick, 900);
    };
    tick();
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════════════════════
//  GANTT CHARTS GALLERY
// ═══════════════════════════════════════════════════════════
function setupGanttModalHandlers() {
    ganttModalClose.addEventListener('click', () => hide(ganttModal));
    ganttModal.addEventListener('click', (e) => { if (e.target === ganttModal) hide(ganttModal); });
    viewRescheduleGanttBtn.addEventListener('click', () => showGanttCharts('reschedule'));
}

async function showGanttCharts(type = 'baseline') {
    ganttModalTitle.textContent = type === 'reschedule' ? '📊 Yeniden Planlama Karşılaştırma Grafikleri' : '📊 Temel Gantt Grafikleri';
    ganttModalBody.innerHTML = `
        <div class="flex flex-col items-center gap-4 py-12">
            <div class="spinner"></div>
            <p class="text-sm text-slate-500">Grafikler yükleniyor...</p>
        </div>`;
    showFlex(ganttModal);

    try {
        const response = await fetch(`${API_BASE}/charts`);
        if (!response.ok) throw new Error('Grafik listesi yüklenemedi');
        const { charts } = await response.json();

        let filtered = charts;
        if (type === 'baseline') {
            filtered = charts.filter(c => c.type === 'baseline');
        } else if (type === 'reschedule') {
            filtered = charts.filter(c => c.type === 'reschedule');
        }

        if (!filtered || filtered.length === 0) {
            ganttModalBody.innerHTML = `
                <div class="flex flex-col items-center gap-3 py-12 text-center">
                    <span class="material-symbols-outlined text-slate-600 text-[48px]">bar_chart</span>
                    <p class="text-sm text-slate-500">Henüz grafik yok. Önce optimize ediciyi çalıştırın.</p>
                </div>`;
            return;
        }

        let html = '<div class="gantt-gallery">';
        for (const chart of filtered) {
            const imgUrl = `${API_BASE}/chart/${encodeURIComponent(chart.filename)}`;
            const label = chart.filename.replace(/_/g, ' ').replace('.png', '');
            html += `
                <div class="bg-surface2 rounded-xl overflow-hidden border border-border-col p-3">
                    <p class="text-xs text-slate-400 mb-2 font-medium truncate">${label}</p>
                    <img class="gantt-chart-img" src="${imgUrl}" alt="${label}"
                         onclick="openImageFullscreen('${imgUrl}')"
                         onerror="this.parentElement.innerHTML='<p class=\\'text-xs text-red-400 p-4\\'>Grafik bulunamadı</p>'">
                </div>`;
        }
        html += '</div>';
        ganttModalBody.innerHTML = html;

    } catch (err) {
        ganttModalBody.innerHTML = `
            <div class="flex flex-col items-center gap-3 py-12 text-center">
                <span class="material-symbols-outlined text-red-500 text-[48px]">warning</span>
                <p class="text-sm text-red-400 font-semibold">Grafikler yüklenemedi</p>
                <p class="text-xs text-slate-500">${err.message}</p>
                <p class="text-xs text-slate-600 mt-1"><strong>server.py</strong>'nin çalıştığından emin olun</p>
            </div>`;
    }
}

function openImageFullscreen(url) {
    const w = window.open('', '_blank');
    w.document.write(`<html><body style="margin:0;background:#0d1117;">
        <img src="${url}" style="max-width:100%;max-height:100vh;display:block;margin:auto;">
    </body></html>`);
    w.document.close();
}

// ═══════════════════════════════════════════════════════════
//  WORKFLOW NAVIGATION
// ═══════════════════════════════════════════════════════════
function showActionsSection() {
    show(actionsSection);
    show(rescheduleBtn);   // ensure reschedule card is always visible
    show(downloadBtn);     // ensure download card is always visible
    actionsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showRescheduleOptions() {
    if (!operationsData) {
        alert('Önce bir veri dosyası yükleyin.');
        return;
    }
    if (!isOptimized) {
        alert('Yeniden planlama için önce optimizasyon çalıştırılmalı.');
        showActionsSection();
        return;
    }
    hide(workflowSection);
    show(rescheduleTypeSection);
    rescheduleTypeSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setupWorkflowHandlers() {
    rescheduleCards.forEach(card => {
        card.addEventListener('click', () => {
            rescheduleCards.forEach(c => c.classList.remove('border-primary', 'bg-primary/10'));
            card.classList.add('border-primary', 'bg-primary/10');
            showWorkflowForm(card.dataset.type);
        });
    });

    const urgentJobUpload = document.getElementById('urgentJobUpload');
    const urgentJobFileInput = document.getElementById('urgentJobFileInput');
    urgentJobUpload.addEventListener('click', () => urgentJobFileInput.click());
    urgentJobUpload.addEventListener('dragover', (e) => {
        e.preventDefault();
        urgentJobUpload.classList.add('dragover-active');
    });
    urgentJobUpload.addEventListener('dragleave', () => urgentJobUpload.classList.remove('dragover-active'));
    urgentJobUpload.addEventListener('drop', (e) => {
        e.preventDefault();
        urgentJobUpload.classList.remove('dragover-active');
        const file = e.dataTransfer.files[0];
        if (!file) return;
        urgentJobFileInput.files = e.dataTransfer.files;
        const infoEl = document.getElementById('urgentJobFileInfo');
        if (infoEl) {
            infoEl.textContent = `✓ ${file.name} seçildi`;
            showFlex(infoEl);
        }
    });
    urgentJobFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const infoEl = document.getElementById('urgentJobFileInfo');
        if (infoEl) {
            infoEl.textContent = `✓ ${file.name} seçildi`;
            showFlex(infoEl);
        }
    });
    document.getElementById('urgentJobRescheduleBtn').addEventListener('click', processUrgentJob);
    document.getElementById('stationChangesRescheduleBtn').addEventListener('click', processStationChanges);
    document.getElementById('machineChangesRescheduleBtn').addEventListener('click', processMachineChanges);
    document.getElementById('dueDateRescheduleBtn').addEventListener('click', processDueDateChanges);
}

function showWorkflowForm(type) {
    document.querySelectorAll('.workflow-form').forEach(f => hide(f));
    show(workflowSection);
    currentWorkflow = type;

    const titles = {
        'urgent-job': '🚨 Acil İş Girişi',
        'station-changes': '🏭 Makineyi Devre Dışı Bırak',
        'machine-changes': '⚙️ İstasyonu Devre Dışı Bırak',
        'due-date': '📅 Termin Tarihi Değişiklikleri',
    };
    workflowTitle.textContent = titles[type] || type;

    const formIds = {
        'urgent-job': 'urgentJobForm',
        'station-changes': 'stationChangesForm',
        'machine-changes': 'machineChangesForm',
        'due-date': 'dueDateForm',
    };
    const formEl = document.getElementById(formIds[type]);
    if (formEl) show(formEl);

    if (type === 'due-date') populateDueDateOperationDropdown();

    workflowSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function populateDueDateOperationDropdown() {
    const sel = document.getElementById('dueDateOperationId');
    if (!sel) return;
    sel.innerHTML = '<option value="">— Operasyon seçin —</option>';
    if (!operationsData) return;
    const ops = getOperationsList(operationsData);
    if (!Array.isArray(ops)) return;
    ops.forEach(op => {
        const id = op.id ?? op.Id ?? op.ID ?? '';
        const label = op.name ?? op.Name ?? op.workOrderNumber ?? '';
        if (id === '') return;
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = label ? `${id} – ${label}` : `${id}`;
        sel.appendChild(opt);
    });
}

// ═══════════════════════════════════════════════════════════
//  RESCHEDULE PROCESSING
// ═══════════════════════════════════════════════════════════
async function processUrgentJob() {
    // If manual tab is active, the Save button handles it separately.
    // This handler is only for the upload tab.
    const fi = document.getElementById('urgentJobFileInput');
    if (!fi.files[0]) { alert('Lütfen acil iş içeren güncellenmiş JSON dosyasını yükleyin.'); return; }
    const reader = new FileReader();
    reader.onload = async (e) => {
        try {
            const updatedData = JSON.parse(e.target.result);
            const newOps = getOperationsList(updatedData);
            if (!Array.isArray(newOps) && !(updatedData && updatedData.id)) {
                throw new Error('Yüklenen JSON, acil iş için kullanılabilir bir operasyon verisi içermiyor.');
            }
            await performRescheduling({ type: 'urgent-job', urgentJobData: updatedData });
        } catch (error) { alert('Dosya işleme hatası: ' + error.message); }
    };
    reader.readAsText(fi.files[0]);
}

// ═══════════════════════════════════════════════════════════
//  URGENT JOB MANUAL FORM
// ═══════════════════════════════════════════════════════════
function switchUrgentJobTab(tab) {
    const uploadPanel = document.getElementById('ujPanelUpload');
    const manualPanel = document.getElementById('ujPanelManual');
    const tabUpload = document.getElementById('ujTabUpload');
    const tabManual = document.getElementById('ujTabManual');

    const activeClass = ['bg-primary', 'text-white'];
    const inactiveClass = ['text-slate-400'];

    if (tab === 'upload') {
        show(uploadPanel); hide(manualPanel);
        tabUpload.classList.add(...activeClass); tabUpload.classList.remove(...inactiveClass);
        tabManual.classList.remove(...activeClass); tabManual.classList.add(...inactiveClass);
    } else {
        hide(uploadPanel); show(manualPanel);
        tabManual.classList.add(...activeClass); tabManual.classList.remove(...inactiveClass);
        tabUpload.classList.remove(...activeClass); tabUpload.classList.add(...inactiveClass);
    }
}

function collectUrgentJobFormData() {
    const v = (id) => document.getElementById(id)?.value?.trim() ?? '';
    const n = (id) => { const val = v(id); return val === '' ? null : Number(val); };
    const dt = (id) => { const val = v(id); return val ? new Date(val).toISOString() : null; };

    return {
        id: v('ujId'),
        objectType: v('ujObjectType') || null,
        name: v('ujName'),
        status: v('ujStatus') || null,
        plannedStartDateTime: dt('ujPlannedStart'),
        plannedEndDateTime: dt('ujPlannedEnd'),
        realPlannedStartDateTime: dt('ujRealStart'),
        realPlannedEndDateTime: dt('ujRealEnd'),
        endDate: dt('ujEndDate'),
        plannedPartCount: n('ujPlannedPartCount'),
        producedPartCount: n('ujProducedPartCount'),
        cycleTime: n('ujCycleTime'),
        plannedSetupDuration: n('ujSetupDuration'),
        completionRate: n('ujCompletionRate'),
        erpOperationItemNumber: v('ujErpOperationItemNumber') || null,
        parentId: v('ujParentId') || null,
        parentName: v('ujParentName') || null,
        partNumber: v('ujPartNumber') || null,
        workOrderNumber: v('ujWorkOrderNumber') || null,
        erpCustomerOrderNumber: v('ujErpCustomerOrderNumber') || null,
        erpCustomerOrderItem: v('ujErpCustomerOrderItem') || null,
        priority: n('ujPriority'),
        workCenterId: v('ujWorkCenterId') || null,
        workCenterName: v('ujWorkCenterName') || null,
        workCenterMachineId: v('ujWorkCenterMachineId') || null,
        workCenterMachineCode: v('ujWorkCenterMachineCode') || null,
        color: v('ujColor') || null,
        isPlannable: v('ujIsPlannable') === '' ? null : v('ujIsPlannable') === 'true',
        hourlyCost: n('ujHourlyCost'),
    };
}

function saveUrgentJobOperation() {
    const op = collectUrgentJobFormData();
    if (!op.id) { alert('ID gerekli.'); return; }
    if (!op.name) { alert('Ad gerekli.'); return; }
    if (!op.workCenterId) { alert('İş Merkezi ID\'si gerekli.'); return; }

    // Store temporarily for the modal handlers to use
    window._pendingUrgentOp = op;

    const summary = document.getElementById('ujActionOpSummary');
    if (summary) summary.textContent = `📦 Op #${op.id} · ${op.name} · WC: ${op.workCenterId}`;

    showFlex(urgentJobActionModal);
}

function setupUrgentJobActionModal() {
    document.getElementById('ujActionAddAsIs').addEventListener('click', async () => {
        hide(urgentJobActionModal);
        const op = window._pendingUrgentOp;
        if (!op) return;
        appendOperationToData(op);
        await performRescheduling({ type: 'urgent-job', urgentJobData: op });
    });

    document.getElementById('ujActionRestart').addEventListener('click', async () => {
        hide(urgentJobActionModal);
        const op = window._pendingUrgentOp;
        if (!op) return;
        appendOperationToData(op);
        hide(workflowSection);
        hide(rescheduleTypeSection);
        await runOptimization();
    });

    document.getElementById('ujActionCancel').addEventListener('click', () => {
        hide(urgentJobActionModal);
    });

    // Close on backdrop click
    urgentJobActionModal.addEventListener('click', (e) => {
        if (e.target === urgentJobActionModal) hide(urgentJobActionModal);
    });
}

function appendOperationToData(op) {
    if (!operationsData) return;
    // Support assignments, operations array, or bare array formats
    if (Array.isArray(operationsData.assignments)) {
        operationsData.assignments.push(op);
    } else if (Array.isArray(operationsData.workOrderOperationDtoList)) {
        operationsData.workOrderOperationDtoList.push(op);
    } else if (Array.isArray(operationsData.operations)) {
        operationsData.operations.push(op);
    } else if (Array.isArray(operationsData)) {
        operationsData.push(op);
    }
}

async function processStationChanges() {
    const machineId = document.getElementById('stationMachineId').value;
    const machineCount = document.getElementById('stationMachineCount').value;
    if (!machineId || !machineCount) { alert('Lütfen tüm alanları doldurun.'); return; }
    await performRescheduling({ type: 'station-changes', machineId, newCount: parseInt(machineCount) });
}

async function processMachineChanges() {
    const stationId = document.getElementById('machineStationId').value;
    const stationCount = document.getElementById('machineStationCount').value;
    if (!stationId || !stationCount) { alert('Lütfen tüm alanları doldurun.'); return; }
    await performRescheduling({ type: 'machine-changes', stationId, newCount: parseInt(stationCount) });
}

async function processDueDateChanges() {
    const operationId = document.getElementById('dueDateOperationId').value;
    const newDate = document.getElementById('dueDateNewDate').value;
    if (!operationsData) {
        alert('Devam etmek için lütfen önce bir JSON dosyası yükleyin.');
        return;
    }
    if (!operationId || !newDate) { alert('Lütfen tüm alanları doldurun.'); return; }
    await performRescheduling({ type: 'due-date', operationId, newDueDate: newDate });
}

async function performRescheduling(params) {
    hide(workflowSection);
    hide(rescheduleTypeSection);
    show(processingSection);
    processingText.textContent = 'Yeniden planlama isteği işleniyor...';
    processingSection.scrollIntoView({ behavior: 'smooth', block: 'center' });

    if (isOptimized) {
        try {
            processingText.textContent = 'Optimize edici arka uca gönderiliyor...';
            const response = await fetch(`${API_BASE}/reschedule`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                const detail = err.error || err.message || `HTTP ${response.status}`;
                throw new Error(`HTTP ${response.status}: ${detail}`);
            }
            const result = await response.json();
            rescheduledData = result.result;
            lastChartFolder = result.folder || null;

            hide(processingSection);
            show(resultsSection);
            resultsMessage.textContent =
                `Yeniden planlama tamamlandı. Hedef — T_max: ${rescheduledData.objective?.T_max?.toFixed(2) ?? '?'}s, C_max: ${rescheduledData.objective?.C_max?.toFixed(2) ?? '?'}s`;

            if (result.chart_count > 0) {
                showInlineFlex(viewRescheduleGanttBtn);
            }
            renderKpiDashboard(rescheduledData);
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            return;
        } catch (err) {
            hide(processingSection);
            alert('Yeniden planlama hatası: ' + err.message);
            return;
        }
    }

    // Fallback: local simulation (no backend)
    await simulateRescheduling(params);
    hide(processingSection);
    show(resultsSection);
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

async function simulateRescheduling(params) {
    return new Promise((resolve) => {
        setTimeout(() => {
            const operations = getOperationsList(operationsData);

            rescheduledData = operations.map(op => {
                const newOp = { ...op };
                switch (params.type) {
                    case 'urgent-job':
                        // BUG FIX 5: was params.data, corrected to params.urgentJobData
                        if (params.urgentJobData) {
                            const newOps = getOperationsList(params.urgentJobData);
                            rescheduledData = (Array.isArray(newOps) && newOps.length) ? newOps : [params.urgentJobData];
                        }
                        break;
                    case 'station-changes':
                        if (newOp.workCenterMachineId == params.machineId)
                            newOp.reschedulingReason = `Machine count updated to ${params.newCount}`;
                        break;
                    case 'machine-changes':
                        if (newOp.workCenterId == params.stationId)
                            newOp.reschedulingReason = `Station count updated to ${params.newCount}`;
                        break;
                    case 'due-date':
                        if (newOp.id == params.operationId) {
                            newOp.realPlannedEndDateTime = params.newDueDate;
                            newOp.reschedulingReason = 'Due date updated';
                        }
                        break;
                }
                newOp.rescheduledAt = new Date().toISOString();
                newOp.reschedulingType = params.type;
                return newOp;
            });

            resultsMessage.textContent =
                `${Array.isArray(rescheduledData) ? rescheduledData.length : '?'} operasyon başarıyla yeniden planlandı.`;
            resolve();
        }, 2000);
    });
}

// ═══════════════════════════════════════════════════════════
//  DATA EXTRACTION (for view modals)
// ═══════════════════════════════════════════════════════════
function extractMachines() {
    const data = optimizedData || operationsData;
    const operations = optimizedData?.schedule || getOperationsList(data) || [data];
    const machines = new Map();
    (Array.isArray(operations) ? operations : []).forEach(op => {
        const machineKey = op.machine ?? op.workCenterMachineId;
        const machineLabel = op.workCenterMachineCode ?? op.machine_label ?? machineKey;
        if (machineKey !== undefined && machineKey !== null) {
            machines.set(machineKey, { id: machineKey, code: machineLabel, workCenter: op.workCenterName || 'N/A' });
        }
    });
    return Array.from(machines.values());
}

function extractStations() {
    const data = optimizedData || operationsData;
    const operations = optimizedData?.schedule || getOperationsList(data) || [data];
    const stations = new Map();
    (Array.isArray(operations) ? operations : []).forEach(op => {
        const stationKey = op.station ?? op.workCenterId;
        const stationLabel = op.workCenterName ?? op.station_label ?? stationKey;
        if (stationKey !== undefined && stationKey !== null) {
            stations.set(stationKey, { id: stationKey, name: stationLabel });
        }
    });
    return Array.from(stations.values());
}

// ═══════════════════════════════════════════════════════════
//  MODAL HANDLERS (data view)
// ═══════════════════════════════════════════════════════════
function setupModalHandlers() {
    modalClose.addEventListener('click', closeModal);
    dataModal.addEventListener('click', (e) => { if (e.target === dataModal) closeModal(); });
}

function showDataModal(title, data) {
    modalTitle.textContent = title;
    modalBody.innerHTML = createDataTable(data);
    showFlex(dataModal);
}

function closeModal() { hide(dataModal); }

function createDataTable(data) {
    if (!data || data.length === 0) return '<p class="p-6 text-sm text-slate-500">No data available.</p>';
    const headers = Object.keys(data[0]);
    let html = '<table class="data-table"><thead><tr>';
    headers.forEach(h => { html += `<th>${h.charAt(0).toUpperCase() + h.slice(1)}</th>`; });
    html += '</tr></thead><tbody>';
    data.forEach(row => {
        html += '<tr>';
        headers.forEach(h => { html += `<td>${row[h] ?? 'N/A'}</td>`; });
        html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
}

// ═══════════════════════════════════════════════════════════
//  DOWNLOAD
// ═══════════════════════════════════════════════════════════
function downloadCurrentData() {
    const data = optimizedData || operationsData;
    downloadJSON(data, 'operations-data.json');
}

function setupDownloadResultHandler() {
    if (downloadResultBtn) {
        downloadResultBtn.addEventListener('click', () => {
            downloadJSON(rescheduledData, `rescheduled-${new Date().toISOString().split('T')[0]}.json`);
        });
    }
}

function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
}

// ═══════════════════════════════════════════════════════════
//  RESET
// ═══════════════════════════════════════════════════════════
function resetApplication() {
    operationsData = null; optimizedData = null;
    rescheduledData = null; currentWorkflow = null;
    isOptimized = false; lastChartFolder = null;
    window._pendingUrgentOp = null;

    fileInput.value = '';

    // Header chip
    const chip = document.getElementById('headerFileChip');
    if (chip) hide(chip);

    hide(fileInfo);
    hide(viewDataBtn);
    hide(viewStationsBtn);
    hide(actionsSection);
    hide(viewGanttBtn);
    hide(viewRescheduleGanttBtn);
    hide(rescheduleTypeSection);
    hide(workflowSection);
    hide(processingSection);
    hide(resultsSection);
    hide(optimizeModal);
    hide(optimizingModal);
    hide(ganttModal);
    hide(urgentJobActionModal);

    // Reset KPI dashboard
    ['kpiOnTime', 'kpiLate', 'kpiCmax', 'kpiTmax'].forEach(id => {
        const el = document.getElementById(id); if (el) el.textContent = '—';
    });
    ['kpiMachineUtil', 'kpiStationUtil'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = '<p class="text-xs text-slate-600 italic">Veri yok</p>';
    });
    const kpiTable = document.getElementById('kpiDelayTable');
    if (kpiTable) kpiTable.innerHTML = '<p class="text-xs text-slate-600 italic">Veri yok</p>';
    const kpiBadge = document.getElementById('kpiDelayBadge');
    if (kpiBadge) kpiBadge.textContent = '';

    // Reset manual form
    const manualInputIds = [
        'ujId', 'ujObjectType', 'ujName', 'ujStatus',
        'ujPlannedStart', 'ujPlannedEnd', 'ujRealStart', 'ujRealEnd', 'ujEndDate',
        'ujPlannedPartCount', 'ujProducedPartCount', 'ujCycleTime', 'ujSetupDuration', 'ujCompletionRate',
        'ujErpOperationItemNumber', 'ujParentId', 'ujParentName', 'ujPartNumber',
        'ujWorkOrderNumber', 'ujErpCustomerOrderNumber', 'ujErpCustomerOrderItem', 'ujPriority',
        'ujWorkCenterId', 'ujWorkCenterName', 'ujWorkCenterMachineId', 'ujWorkCenterMachineCode',
        'ujColor', 'ujIsPlannable', 'ujHourlyCost'
    ];
    manualInputIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
    });
    // Reset tabs back to upload
    switchUrgentJobTab('upload');

    rescheduleCards.forEach(c => c.classList.remove('border-primary', 'bg-primary/10'));
    window.scrollTo({ top: 0, behavior: 'smooth' });
    checkServerHealth();
}

// ═══════════════════════════════════════════════════════════
//  KPI DASHBOARD
// ═══════════════════════════════════════════════════════════
function renderKpiDashboard(result) {
    if (!result) return;

    // ── 1. Headline stat cards ──────────────────────────
    const delays = result.job_delays || [];
    const onTimeJobs = delays.filter(d => !d.is_late).length;
    const lateJobs = delays.filter(d => d.is_late).length;
    const obj = result.objective || {};
    const cmax = obj.C_max != null ? parseFloat(obj.C_max).toFixed(1) : '—';
    const tmax = obj.T_max != null ? parseFloat(obj.T_max).toFixed(1) : '—';

    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('kpiOnTime', delays.length ? onTimeJobs : '—');
    set('kpiLate', delays.length ? lateJobs : '—');
    set('kpiCmax', cmax);
    set('kpiTmax', tmax);

    // ── 2. Utilization bars ─────────────────────────────
    function renderUtilBars(containerId, utilBlock) {
        const el = document.getElementById(containerId);
        if (!el) return;
        const rows = (utilBlock && utilBlock.rows) ? utilBlock.rows : [];
        if (!rows.length) { el.innerHTML = '<p class="text-xs text-slate-600 italic">Veri yok</p>'; return; }

        el.innerHTML = rows.map(r => {
            const pct = Math.min(100, Math.round(r.utilization_pct || 0));
            const activePct = Math.min(100, Math.round(r.active_utilization_pct || 0));
            const barColor = pct >= 80 ? '#ef4444' : pct >= 50 ? '#f59e0b' : '#22c55e';
            return `
            <div>
              <div class="flex items-center justify-between mb-0.5">
                <span class="text-[11px] text-slate-300 font-medium">${r.label || r.resource_id}</span>
                <span class="text-[10px] text-slate-500">${pct}% genel · ${activePct}% aktif</span>
              </div>
              <div class="w-full h-1.5 bg-surface rounded-full overflow-hidden">
                <div class="h-full rounded-full transition-all duration-700"
                     style="width:${pct}%; background:${barColor}"></div>
              </div>
            </div>`;
        }).join('');
    }
    renderUtilBars('kpiMachineUtil', result.machine_utilization);
    renderUtilBars('kpiStationUtil', result.station_utilization);

    // ── 3. Job delay table ──────────────────────────────
    const badge = document.getElementById('kpiDelayBadge');
    const tableEl = document.getElementById('kpiDelayTable');
    if (badge) badge.textContent = delays.length ? `${delays.length} iş` : '';
    if (!tableEl) return;

    if (!delays.length) {
        tableEl.innerHTML = '<p class="text-xs text-slate-600 italic">Gecikme verisi yok</p>';
        return;
    }

    // Sort: late jobs first, then by delay descending
    const sorted = [...delays].sort((a, b) => {
        if (a.is_late !== b.is_late) return a.is_late ? -1 : 1;
        return (b.delay_hours || 0) - (a.delay_hours || 0);
    });

    const rows = sorted.map(d => {
        const late = d.is_late;
        const delay = d.delay_hours != null ? parseFloat(d.delay_hours).toFixed(2) : '0.00';
        const due = d.due != null ? parseFloat(d.due).toFixed(1) : '—';
        const comp = d.completion != null ? parseFloat(d.completion).toFixed(1) : '—';
        const rowBg = late ? 'bg-red-500/5' : 'bg-green-500/5';
        const badge = late
            ? '<span class="px-1.5 py-0.5 bg-red-500/20 text-red-400 text-[9px] font-bold rounded">GEÇ</span>'
            : '<span class="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[9px] font-bold rounded">OK</span>';
        return `<tr class="${rowBg}">
            <td class="px-3 py-1.5 text-slate-300 font-mono text-xs">#${d.job_id}</td>
            <td class="px-3 py-1.5 text-xs text-slate-400">${due} sa</td>
            <td class="px-3 py-1.5 text-xs text-slate-400">${comp} sa</td>
            <td class="px-3 py-1.5 text-xs font-semibold ${late ? 'text-red-400' : 'text-green-400'}">${delay} sa</td>
            <td class="px-3 py-1.5">${badge}</td>
        </tr>`;
    }).join('');

    tableEl.innerHTML = `
        <table class="data-table w-full text-left">
            <thead>
                <tr>
                    <th class="px-3 py-2">İş</th>
                    <th class="px-3 py-2">Termin</th>
                    <th class="px-3 py-2">Tamamlanma</th>
                    <th class="px-3 py-2">Gecikme</th>
                    <th class="px-3 py-2">Durum</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>`;
}

