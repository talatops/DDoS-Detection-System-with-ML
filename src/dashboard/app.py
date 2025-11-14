#!/usr/bin/env python3
"""
Flask dashboard for real-time DDoS detection monitoring.
Provides REST API and WebSocket for live updates.
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import json
import os
import time
from datetime import datetime
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ddos-detection-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Shared state (in production, use Redis or database)
metrics_data = {
    'gpu_utilization': [],
    'throughput_pps': [],
    'throughput_gbps': [],
    'alerts': [],
    'blackhole_list': [],
    'detection_stats': {
        'total_windows': 0,
        'attacks_detected': 0,
        'false_positives': 0
    }
}

def load_metrics_from_file():
    """Load metrics from CSV log files."""
    metrics_file = 'logs/metrics.csv'
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    # Parse last line
                    parts = lines[-1].strip().split(',')
                    if len(parts) >= 5:
                        return {
                            'timestamp': parts[0],
                            'cpu_percent': float(parts[1]),
                            'gpu_percent': float(parts[2]),
                            'memory_mb': float(parts[3]),
                            'pps_in': float(parts[4]),
                            'pps_processed': float(parts[5]) if len(parts) > 5 else 0
                        }
        except Exception as e:
            print(f"Error loading metrics: {e}")
    return None

def load_alerts_from_file():
    """Load recent alerts from log file."""
    alerts_file = 'logs/alerts.csv'
    alerts = []
    if os.path.exists(alerts_file):
        try:
            with open(alerts_file, 'r') as f:
                lines = f.readlines()
                # Get last 50 alerts
                for line in lines[-50:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        alerts.append({
                            'timestamp': parts[0],
                            'window_start': parts[1],
                            'src_ip': parts[2],
                            'score': float(parts[3]),
                            'detector': parts[4]
                        })
        except Exception as e:
            print(f"Error loading alerts: {e}")
    return alerts

def load_blackhole_list():
    """Load blackhole list from JSON file."""
    blackhole_file = 'blackhole.json'
    if os.path.exists(blackhole_file):
        try:
            with open(blackhole_file, 'r') as f:
                data = json.load(f)
                return data.get('blackhole_ips', [])
        except Exception as e:
            print(f"Error loading blackhole list: {e}")
    return []

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/metrics')
def get_metrics():
    """Get current system metrics."""
    metrics = load_metrics_from_file()
    if metrics:
        return jsonify(metrics)
    return jsonify({'error': 'No metrics available'}), 404

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts."""
    alerts = load_alerts_from_file()
    return jsonify({'alerts': alerts})

@app.route('/api/blackhole')
def get_blackhole():
    """Get current blackhole list."""
    blackhole_list = load_blackhole_list()
    return jsonify({'blackhole_ips': blackhole_list})

@app.route('/api/stats')
def get_stats():
    """Get detection statistics."""
    return jsonify(metrics_data['detection_stats'])

@app.route('/api/training-metrics')
def get_training_metrics():
    """Get ML model training metrics."""
    metrics_file = 'reports/training_metrics.json'
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({'error': f'Error loading training metrics: {e}'}), 500
    return jsonify({'error': 'Training metrics not found'}), 404

@app.route('/api/training-report')
def get_training_report():
    """Get training report text."""
    report_file = 'reports/training_report.txt'
    if os.path.exists(report_file):
        try:
            with open(report_file, 'r') as f:
                return f.read(), 200, {'Content-Type': 'text/plain'}
        except Exception as e:
            return jsonify({'error': f'Error loading report: {e}'}), 500
    return jsonify({'error': 'Training report not found'}), 404

@app.route('/api/roc-curve')
def get_roc_curve():
    """Serve ROC curve image."""
    from flask import send_file
    roc_file = 'results/ml_roc_curve.png'
    if os.path.exists(roc_file):
        return send_file(roc_file, mimetype='image/png')
    return jsonify({'error': 'ROC curve not found'}), 404

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    print('Client connected')
    emit('connected', {'data': 'Connected to DDoS Detection Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print('Client disconnected')

def background_task():
    """Background task to emit metrics updates."""
    while True:
        time.sleep(1)  # Update every second
        
        metrics = load_metrics_from_file()
        if metrics:
            socketio.emit('metrics_update', metrics)
        
        alerts = load_alerts_from_file()
        if alerts:
            socketio.emit('alerts_update', {'alerts': alerts[-10:]})  # Last 10 alerts
        
        blackhole_list = load_blackhole_list()
        if blackhole_list:
            socketio.emit('blackhole_update', {'blackhole_ips': blackhole_list})

if __name__ == '__main__':
    # Start background task
    thread = threading.Thread(target=background_task, daemon=True)
    thread.start()
    
    # Run Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

