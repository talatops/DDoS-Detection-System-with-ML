// Dashboard JavaScript for real-time updates

const socket = io();

// Chart data
let gpuData = {
    x: [],
    y: [],
    type: 'scatter',
    mode: 'lines',
    name: 'GPU Utilization (%)'
};

let throughputData = {
    x: [],
    y: [],
    type: 'scatter',
    mode: 'lines',
    name: 'Throughput (pps)'
};

// Initialize charts
Plotly.newPlot('gpu-chart', [gpuData], {
    title: 'GPU Utilization',
    xaxis: { title: 'Time' },
    yaxis: { title: 'Utilization (%)', range: [0, 100] }
});

Plotly.newPlot('throughput-chart', [throughputData], {
    title: 'Throughput',
    xaxis: { title: 'Time' },
    yaxis: { title: 'Packets per Second' }
});

// WebSocket event handlers
socket.on('connect', function() {
    console.log('Connected to server');
});

socket.on('metrics_update', function(data) {
    updateMetrics(data);
    updateCharts(data);
});

socket.on('alerts_update', function(data) {
    updateAlerts(data.alerts);
});

socket.on('blackhole_update', function(data) {
    updateBlackholeList(data.blackhole_ips);
});

// Update metrics display
function updateMetrics(data) {
    if (data.gpu_percent !== undefined) {
        document.getElementById('gpu-util').textContent = data.gpu_percent.toFixed(1) + '%';
    }
    if (data.pps_in !== undefined) {
        document.getElementById('throughput-pps').textContent = Math.round(data.pps_in).toLocaleString();
    }
    if (data.pps_in !== undefined) {
        const gbps = (data.pps_in * 1500 * 8) / 1e9; // Approximate
        document.getElementById('throughput-gbps').textContent = gbps.toFixed(2);
    }
}

// Update charts
function updateCharts(data) {
    const now = new Date().toISOString();
    
    // GPU chart
    gpuData.x.push(now);
    gpuData.y.push(data.gpu_percent || 0);
    if (gpuData.x.length > 100) {
        gpuData.x.shift();
        gpuData.y.shift();
    }
    Plotly.redraw('gpu-chart');
    
    // Throughput chart
    throughputData.x.push(now);
    throughputData.y.push(data.pps_in || 0);
    if (throughputData.x.length > 100) {
        throughputData.x.shift();
        throughputData.y.shift();
    }
    Plotly.redraw('throughput-chart');
}

// Update alerts list
function updateAlerts(alerts) {
    const alertsList = document.getElementById('alerts-list');
    alertsList.innerHTML = '';
    
    if (alerts.length === 0) {
        alertsList.innerHTML = '<p>No alerts</p>';
        return;
    }
    
    alerts.forEach(alert => {
        const div = document.createElement('div');
        div.className = 'alert-item';
        div.innerHTML = `
            <strong>${alert.detector}</strong> - ${alert.src_ip}<br>
            Score: ${alert.score.toFixed(3)} | Time: ${alert.timestamp}
        `;
        alertsList.appendChild(div);
    });
    
    document.getElementById('attacks-detected').textContent = alerts.length;
}

// Update blackhole list
function updateBlackholeList(ips) {
    const blackholeList = document.getElementById('blackhole-list');
    blackholeList.innerHTML = '';
    
    if (ips.length === 0) {
        blackholeList.innerHTML = '<p>No blackholed IPs</p>';
        return;
    }
    
    ips.forEach(ip => {
        const div = document.createElement('div');
        div.className = 'ip-item';
        div.textContent = ip;
        blackholeList.appendChild(div);
    });
    
    document.getElementById('blackhole-count').textContent = ips.length;
}

// Toggle RTBH
function toggleRTBH() {
    fetch('/api/toggle-rtbh', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            alert('RTBH toggled: ' + (data.enabled ? 'ON' : 'OFF'));
        })
        .catch(error => {
            console.error('Error toggling RTBH:', error);
        });
}

// Clear blackhole list
function clearBlackhole() {
    if (confirm('Clear all blackholed IPs?')) {
        fetch('/api/clear-blackhole', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                updateBlackholeList([]);
            })
            .catch(error => {
                console.error('Error clearing blackhole:', error);
            });
    }
}

// Load training metrics
function loadTrainingMetrics() {
    fetch('/api/training-metrics')
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Training metrics not available');
        })
        .then(data => {
            // Update metrics display
            if (data.metrics) {
                document.getElementById('model-accuracy').textContent = 
                    (data.metrics.test_accuracy * 100).toFixed(1) + '%';
                document.getElementById('roc-auc').textContent = 
                    data.metrics.roc_auc.toFixed(3);
                document.getElementById('f1-score').textContent = 
                    data.metrics.f1_score.toFixed(3);
                document.getElementById('precision').textContent = 
                    data.metrics.precision.toFixed(3);
            }
            
            // Update model info
            if (data.model_info) {
                document.getElementById('model-algo').textContent = data.model_info.algorithm;
                document.getElementById('model-features').textContent = data.model_info.n_features;
            }
            
            if (data.dataset) {
                document.getElementById('train-samples').textContent = 
                    data.dataset.train_samples.toLocaleString();
                document.getElementById('test-samples').textContent = 
                    data.dataset.test_samples.toLocaleString();
            }
        })
        .catch(error => {
            console.log('Training metrics not loaded:', error);
            // Hide training section if no data
            document.getElementById('training-metrics').innerHTML = 
                '<p style="color: #999;">Training metrics not available. Train the model first.</p>';
        });
}

// Load training report in new window
function loadTrainingReport() {
    window.open('/api/training-report', '_blank');
}

// Load initial data
fetch('/api/metrics')
    .then(response => response.json())
    .then(data => updateMetrics(data));

fetch('/api/alerts')
    .then(response => response.json())
    .then(data => updateAlerts(data.alerts));

fetch('/api/blackhole')
    .then(response => response.json())
    .then(data => updateBlackholeList(data.blackhole_ips));

// Load training metrics on page load
loadTrainingMetrics();

