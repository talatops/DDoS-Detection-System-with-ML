const gpuChart = document.getElementById('gpu-chart');
const ppsChart = document.getElementById('pps-chart');

const chartLayout = {
    margin: { t: 10, r: 20, l: 40, b: 30 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#e2e8f0' },
    xaxis: { type: 'date', tickformat: '%H:%M:%S' }
};

Plotly.newPlot(gpuChart, [{
    x: [], y: [], mode: 'lines', line: { color: '#38bdf8' },
    name: 'GPU %'
}], { ...chartLayout, yaxis: { title: 'GPU %', range: [0, 100] } }, { responsive: true });

Plotly.newPlot(ppsChart, [{
    x: [], y: [], mode: 'lines', line: { color: '#a78bfa' },
    name: 'Throughput (pps)'
}], { ...chartLayout, yaxis: { title: 'Packets/sec' } }, { responsive: true });

function renderSummary(summary) {
    const cards = [
        { label: 'GPU Utilization', value: `${summary.gpu_percent.toFixed(1)}%` },
        { label: 'CPU Utilization', value: `${summary.cpu_percent.toFixed(1)}%` },
        { label: 'Packets / sec', value: summary.pps_in.toLocaleString() },
        { label: 'Windows Processed', value: summary.windows_processed.toLocaleString() },
        { label: 'Active Model', value: summary.model || 'N/A' },
        { label: 'Memory (MB)', value: summary.memory_mb.toFixed(0) },
    ];
    document.getElementById('summary-cards').innerHTML = cards.map(card => `
        <div class="summary-card">
            <div class="summary-label">${card.label}</div>
            <div class="summary-value">${card.value}</div>
        </div>
    `).join('');
}

function renderAlerts(alerts) {
    const table = document.getElementById('alerts-table');
    if (!alerts.length) {
        table.innerHTML = '<tr><td>No alerts yet</td></tr>';
        return;
    }
    table.innerHTML = `
        <tr><th>Time</th><th>Window</th><th>Source IP</th><th>Detector</th><th>Entropy</th><th>ML</th></tr>
        ${alerts.map(alert => `
            <tr>
                <td>${alert.timestamp}</td>
                <td>${alert.window_index}</td>
                <td>${alert.src_ip}</td>
                <td><span class="badge">${alert.detector}</span></td>
                <td>${alert.entropy_score.toFixed(3)}</td>
                <td>${alert.ml_score.toFixed(3)}</td>
            </tr>
        `).join('')}
    `;
}

function renderKernelTimes(kernelTimes) {
    const table = document.getElementById('kernel-table');
    if (!kernelTimes.length) {
        table.innerHTML = '<tr><td>No GPU kernel runs logged</td></tr>';
        return;
    }
    table.innerHTML = `
        <tr><th>Time</th><th>Kernel</th><th>Duration (ms)</th></tr>
        ${kernelTimes.map(entry => `
            <tr>
                <td>${entry.timestamp}</td>
                <td>${entry.kernel}</td>
                <td>${entry.duration_ms.toFixed(3)}</td>
            </tr>
        `).join('')}
    `;
}

function renderBlocking(blockingRows) {
    const table = document.getElementById('blocking-table');
    if (!blockingRows.length) {
        table.innerHTML = '<tr><td>No blocking events</td></tr>';
        return;
    }
    table.innerHTML = `
        <tr><th>Time</th><th>IP</th><th>Impacted Packets</th><th>Dropped Packets</th></tr>
        ${blockingRows.map(row => `
            <tr>
                <td>${row.timestamp}</td>
                <td>${row.ip}</td>
                <td>${row.impacted_packets.toLocaleString()}</td>
                <td>${row.dropped_packets.toLocaleString()}</td>
            </tr>
        `).join('')}
    `;
}

function renderManifest(manifest) {
    const table = document.getElementById('manifest-table');
    if (!manifest.models) {
        table.innerHTML = '<tr><td>No model manifest found</td></tr>';
        return;
    }
    table.innerHTML = `
        <tr><th>Name</th><th>Type</th><th>Recall</th><th>FPR</th></tr>
        ${manifest.models.map(model => `
            <tr>
                <td>${model.name}${manifest.selected_model === model.name ? ' ⭐' : ''}</td>
                <td>${model.type}</td>
                <td>${(model.recall * 100).toFixed(2)}%</td>
                <td>${(model.false_positive_rate * 100).toFixed(2)}%</td>
            </tr>
        `).join('')}
    `;
}

function renderTraining(training) {
    const table = document.getElementById('model-table');
    const metrics = training.model_metrics || {};
    table.innerHTML = `
        <tr><th>Accuracy</th><td>${((metrics.test_accuracy || 0) * 100).toFixed(2)}%</td></tr>
        <tr><th>Recall</th><td>${((metrics.recall || 0) * 100).toFixed(2)}%</td></tr>
        <tr><th>Precision</th><td>${((metrics.precision || 0) * 100).toFixed(2)}%</td></tr>
        <tr><th>ROC AUC</th><td>${(metrics.roc_auc || 0).toFixed(3)}</td></tr>
        <tr><th>Selected Model</th><td>${training.selected_model || 'N/A'}</td></tr>
        <tr><th>Train Rows</th><td>${training.dataset?.train_rows ?? 'N/A'}</td></tr>
        <tr><th>Test Rows</th><td>${training.dataset?.test_rows ?? 'N/A'}</td></tr>
    `;
}

function updateCharts(history) {
    if (!history.timestamp || !history.timestamp.length) {
        return;
    }
    const timestamps = history.timestamp.map(ts => new Date(Number(ts)));
    Plotly.update(gpuChart, { x: [timestamps], y: [history.gpu || []] });
    Plotly.update(ppsChart, { x: [timestamps], y: [history.pps || []] });
}

function downloadTrainingReport() {
    window.open('/api/training-report', '_blank');
}

async function refresh() {
    try {
        const response = await fetch('/api/dashboard');
        const data = await response.json();
        renderSummary(data.summary || {});
        renderAlerts(data.alerts || []);
        renderKernelTimes(data.kernel_times || []);
        renderBlocking(data.blocking || []);
        renderManifest(data.model_manifest || {});
        renderTraining(data.training || {});
        updateCharts(data.metrics_history || {});
    } catch (error) {
        console.error('Dashboard refresh failed', error);
    }
}

refresh();
setInterval(refresh, 3000);

