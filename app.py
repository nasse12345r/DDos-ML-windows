from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import threading
import time
import os
import queue
import subprocess
import tempfile
import os

# Fix OpenBLAS memory allocation issue on Windows
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import joblib
import sklearn
import xgboost
import logging
import warnings

# ─── FLASK CONFIGURATION ───────────────────────────────
app = Flask(__name__)
# Prefer environment variable for secret key, fallback for local dev
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fyp-ddos-secret!')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── GLOBAL VARIABLES ──────────────────────────────────
# Default settings (can be modified via web UI later if needed)
INTERFACE = "Wi-Fi" # Default Windows interface name; usually "Wi-Fi" or "Ethernet"
CAPTURE_TIME = 10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models")

# Bounded queue to prevent memory leaks/processing lag
analysis_queue = queue.Queue(maxsize=2)
is_running = False  # Controls the background threads

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── LOAD MODELS ──────────────────────────────────────
rf = None
xgb = None
iso = None
scaler = None
features = None

def load_models():
    global rf, xgb, iso, scaler, features
    if rf is not None:
        return
    try:
        logger.info("Loading models...")
        rf       = joblib.load(os.path.join(MODEL_PATH, 'random_forest.pkl'))
        xgb      = joblib.load(os.path.join(MODEL_PATH, 'xgboost.pkl'))
        iso      = joblib.load(os.path.join(MODEL_PATH, 'isolation_forest.pkl'))
        scaler   = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        features = joblib.load(os.path.join(MODEL_PATH, 'feature_names.pkl'))
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Debugging aid for corrupted files
        for model_file in ['random_forest.pkl', 'xgboost.pkl', 'isolation_forest.pkl', 'scaler.pkl', 'feature_names.pkl']:
            path = os.path.join(MODEL_PATH, model_file)
            if os.path.exists(path):
                size = os.path.getsize(path)
                logger.info(f"File {model_file}: Size={size} bytes")
                if size < 100:
                    with open(path, 'r', errors='ignore') as mf:
                        logger.info(f"Content preview of {model_file}: {mf.read()[:50]}")
            else:
                logger.info(f"File {model_file}: NOT FOUND")

# ─── IDS LOGIC ────────────────────────────────────────

def capture_traffic(cycle):
    """Capture live traffic for CAPTURE_TIME seconds using tshark (Windows compatible)"""
    # Use tempfile to get the OS's temporary directory (works on Windows/Linux)
    temp_dir = tempfile.gettempdir()
    pcap_file = os.path.join(temp_dir, f"capture_{cycle}.pcap")
    
    msg = f"Capturing traffic on {INTERFACE} for {CAPTURE_TIME}s (Cycle {cycle})..."
    logger.info(msg)
    socketio.emit('log', {'timestamp': time.strftime('%H:%M:%S'), 'level': 'INFO', 'message': msg})
    
    try:
        if os.path.exists(pcap_file):
            os.remove(pcap_file)

        # dumpcap is the underlying capture engine for Wireshark. It is safer on Windows as it doesn't invoke external tools like etwdump
        # -i: interface, -a: duration:10 (auto-stop after 10 seconds), -w: write to file, -q: quiet mode
        proc = subprocess.Popen([
            'dumpcap', '-i', INTERFACE, '-a', f'duration:{CAPTURE_TIME}', '-w', pcap_file, '-q'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        _, stderr = proc.communicate() # dumpcap auto-terminates due to '-a duration'
        if proc.returncode != 0:
            logger.error(f"dumpcap error: {stderr.decode()}")
            socketio.emit('log', {'timestamp': time.strftime('%H:%M:%S'), 'level': 'ERROR', 'message': f"Capture error: {stderr.decode()}"})
            time.sleep(2)
            return None

        return pcap_file
    except Exception as e:
        logger.error(f"Capture error: {e}")
        socketio.emit('log', {'timestamp': time.strftime('%H:%M:%S'), 'level': 'ERROR', 'message': f"Capture error: Make sure Wireshark/tshark is installed and in PATH. ({e})"})
        time.sleep(2) # Prevent rapid failure looping
        return None

def pcap_to_flows(pcap_path):
    """Convert pcap to flow features using NFStream"""
    try:
        if not os.path.exists(pcap_path):
            return None
            
        # Lazy-load NFStreamer to prevent memory exhaustion (Page File errors) on Windows startup
        from nfstream import NFStreamer

        # n_meters=1 disables multiprocessing in NFStream, which prevents memory spikes and process locks on Windows
        streamer = NFStreamer(
            source=pcap_path,
            statistical_analysis=True,
            splt_analysis=0,
            n_meters=1
        )
        df = streamer.to_pandas()

        if df is None or len(df) == 0:
            return None
        return df

    except Exception as e:
        logger.error(f"Flow extraction error: {e}")
        return None

def preprocess_flows(df):
    """Map NFStream columns to match training features and approximate missing ones"""
    col_map = {
        'protocol':                      'Protocol',
        'bidirectional_duration_ms':     'Flow Duration',
        'src2dst_packets':               'Total Fwd Packets',
        'dst2src_packets':               'Total Backward Packets',
        'src2dst_bytes':                 'Total Length of Fwd Packets',
        'dst2src_bytes':                 'Total Length of Bwd Packets',
        'src2dst_max_ps':                'Fwd Packet Length Max',
        'src2dst_min_ps':                'Fwd Packet Length Min',
        'src2dst_mean_ps':               'Fwd Packet Length Mean',
        'src2dst_stddev_ps':             'Fwd Packet Length Std',
        'dst2src_max_ps':                'Bwd Packet Length Max',
        'dst2src_min_ps':                'Bwd Packet Length Min',
        'dst2src_mean_ps':               'Bwd Packet Length Mean',
        'dst2src_stddev_ps':             'Bwd Packet Length Std',
        'bidirectional_mean_ps':         'Packet Length Mean',
        'bidirectional_stddev_ps':       'Packet Length Std',
        'bidirectional_max_ps':          'Max Packet Length',
        'bidirectional_min_ps':          'Min Packet Length',
        'src2dst_mean_piat_ms':          'Fwd IAT Mean',
        'src2dst_stddev_piat_ms':        'Fwd IAT Std',
        'src2dst_max_piat_ms':           'Fwd IAT Max',
        'src2dst_min_piat_ms':           'Fwd IAT Min',
        'dst2src_mean_piat_ms':          'Bwd IAT Mean',
        'dst2src_stddev_piat_ms':        'Bwd IAT Std',
        'dst2src_max_piat_ms':           'Bwd IAT Max',
        'dst2src_min_piat_ms':           'Bwd IAT Min',
        'bidirectional_mean_piat_ms':    'Flow IAT Mean',
        'bidirectional_stddev_piat_ms':  'Flow IAT Std',
        'bidirectional_max_piat_ms':     'Flow IAT Max',
        'bidirectional_min_piat_ms':     'Flow IAT Min',
        'bidirectional_syn_packets':     'SYN Flag Count',
        'bidirectional_fin_packets':     'FIN Flag Count',
        'bidirectional_rst_packets':     'RST Flag Count',
        'bidirectional_psh_packets':     'PSH Flag Count',
        'bidirectional_ack_packets':     'ACK Flag Count',
        'bidirectional_urg_packets':     'URG Flag Count',
        'bidirectional_ece_packets':     'ECE Flag Count',
        'bidirectional_cwr_packets':     'CWE Flag Count',
        'src2dst_syn_packets':           'Fwd PSH Flags',
        'src2dst_fin_packets':           'Fwd URG Flags',
        'dst2src_syn_packets':           'Bwd PSH Flags',
        'dst2src_fin_packets':           'Bwd URG Flags',
        'src2dst_duration_ms':           'Fwd IAT Total',
        'dst2src_duration_ms':           'Bwd IAT Total',
        'bidirectional_packets':         'Subflow Fwd Packets',
        'bidirectional_bytes':           'Subflow Fwd Bytes',
    }

    df = df.rename(columns=col_map)
    duration_s = df['Flow Duration'] / 1000.0  # ms to seconds

    safe_duration = duration_s.replace(0, np.nan)
    df['Flow Bytes/s'] = ((df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']) / safe_duration).fillna(0)
    df['Flow Packets/s'] = ((df['Total Fwd Packets'] + df['Total Backward Packets']) / safe_duration).fillna(0)
    df['Fwd Packets/s'] = (df['Total Fwd Packets'] / safe_duration).fillna(0)
    df['Bwd Packets/s'] = (df['Total Backward Packets'] / safe_duration).fillna(0)

    df['Packet Length Variance'] = df['Packet Length Std'] ** 2

    total_pkts = df['Total Fwd Packets'] + df['Total Backward Packets']
    safe_total_pkts = total_pkts.replace(0, np.nan)
    df['Average Packet Size'] = ((df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']) / safe_total_pkts).fillna(0)

    df['Avg Fwd Segment Size'] = df['Fwd Packet Length Mean']
    df['Avg Bwd Segment Size'] = df['Bwd Packet Length Mean']
    df['Fwd Header Length']   = df['Total Fwd Packets'] * 20
    df['Bwd Header Length']   = df['Total Backward Packets'] * 20
    df['Fwd Header Length.1'] = df['Fwd Header Length']
    df['Subflow Bwd Packets'] = df['Total Backward Packets']
    df['Subflow Bwd Bytes']   = df['Total Length of Bwd Packets']
    
    safe_fwd_pkts = df['Total Fwd Packets'].replace(0, np.nan)
    df['Down/Up Ratio'] = (df['Total Backward Packets'] / safe_fwd_pkts).fillna(0)

    # Approximations for missing features
    df['Init_Win_bytes_forward'] = df['Total Fwd Packets'] * 29200 
    df['Init_Win_bytes_backward'] = df['Total Backward Packets'] * 29200
    df['act_data_pkt_fwd'] = df['Total Fwd Packets']
    df['min_seg_size_forward'] = 20 
    df['Idle Mean'] = df['Flow IAT Mean']
    df['Idle Std'] = df['Flow IAT Std']
    df['Idle Max'] = df['Flow IAT Max']
    df['Idle Min'] = df['Flow IAT Min']
    df['Active Mean'] = 0
    df['Active Std'] = 0
    df['Active Max'] = 0
    df['Active Min'] = 0
    df['Inbound'] = 1 
    
    missing_cols = ['Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
                    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
    missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
    df = pd.concat([df, missing_df], axis=1)

    final_df = pd.DataFrame()
    
    if features is None:
        logger.error("Features list is not loaded. Cannot preprocess flows.")
        return None
        
    for col in features:
        if col in df.columns:
            final_df[col] = df[col].values
        else:
            final_df[col] = 0

    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.fillna(0, inplace=True)

    return final_df

def detect(df, cycle):
    """Run ML models and emit results to WebSocket"""
    if rf is None or xgb is None or iso is None or scaler is None or features is None:
        msg = f"Cycle {cycle} Verdict: ERROR - ML Models not loaded. Please replace corrupted .pkl files."
        logger.error(msg)
        socketio.emit('log', {'timestamp': time.strftime('%H:%M:%S'), 'level': 'ERROR', 'message': msg})
        payload = {
            'cycle': cycle, 'timestamp': time.strftime('%H:%M:%S'),
            'total_flows': len(df) if df is not None else 0, 'rf_percent': 0, 'xgb_percent': 0, 'iso_percent': 0,
            'status': "danger", 'verdict': "ERROR - Models missing"
        }
        socketio.emit('update_stats', payload)
        return
        
    if df is None or len(df) == 0:
        socketio.emit('log', {'timestamp': time.strftime('%H:%M:%S'), 'level': 'WARNING', 'message': f"Cycle {cycle}: No flows to analyse"})
        return

    total = len(df)
    
    # ── ML Models ───────────────────────────────────────
    X = scaler.transform(df)
    rf_preds  = rf.predict(X)
    xgb_preds = xgb.predict(X)
    iso_raw   = iso.predict(X)
    iso_preds = [1 if x == -1 else 0 for x in iso_raw]

    rf_ratio  = int(sum(rf_preds)) / total
    xgb_ratio = int(sum(xgb_preds)) / total
    iso_ratio = int(sum(iso_preds)) / total

    min_flows = 10 # Lowered slightly to show small bursts on UI
    enough_data = total >= min_flows

    supervised_alert = rf_ratio >= 0.30 and xgb_ratio >= 0.30 and enough_data
    high_confidence = rf_ratio >= 0.60 and xgb_ratio >= 0.60 and enough_data
    
    # Determine Final Verdict
    status = "normal"
    verdict = "BENIGN (Within normal range)"
    
    if high_confidence and enough_data:
        status = "danger"
        verdict = f"DDoS ATTACK DETECTED! ({rf_ratio:.1%} of flows flagged)"
    elif supervised_alert and enough_data:
        status = "warning"
        verdict = f"SUSPICIOUS TRAFFIC ({rf_ratio:.1%} of flows flagged)"
    elif not enough_data:
        verdict = f"BENIGN (Insufficient flows: {total})"
        
    # Log to backend and UI terminal
    msg = f"Cycle {cycle} Verdict: {verdict}"
    log_level = 'CRITICAL' if status == 'danger' else ('WARNING' if status == 'warning' else 'INFO')
    logger.info(msg)
    socketio.emit('log', {'timestamp': time.strftime('%H:%M:%S'), 'level': log_level, 'message': msg})

    # Emit Stats for the UI Dashboard
    payload = {
        'cycle': cycle,
        'timestamp': time.strftime('%H:%M:%S'),
        'total_flows': total,
        'rf_percent': round(rf_ratio * 100, 1),
        'xgb_percent': round(xgb_ratio * 100, 1),
        'iso_percent': round(iso_ratio * 100, 1),
        'status': status,
        'verdict': verdict
    }
    socketio.emit('update_stats', payload)

def analysis_worker():
    """Worker thread that processes pcaps from the queue."""
    global is_running
    while is_running:
        try:
            # timeout ensures thread checks `is_running` flag periodically
            item = analysis_queue.get(timeout=2) 
            if item is None:
                continue
            
            cycle, pcap_path = item
            
            if pcap_path is None:
                continue
                
            logger.info(f"Starting analysis for Cycle {cycle}")
            try:
                raw_df = pcap_to_flows(pcap_path)
                if raw_df is not None:
                    df = preprocess_flows(raw_df.copy())
                    detect(df, cycle)
                else:
                    # Emit empty stats to keep graph moving
                    payload = {
                        'cycle': cycle, 'timestamp': time.strftime('%H:%M:%S'),
                        'total_flows': 0, 'rf_percent': 0, 'xgb_percent': 0, 'iso_percent': 0,
                        'status': "normal", 'verdict': "No traffic captured."
                    }
                    socketio.emit('update_stats', payload)
            except Exception as e:
                logger.error(f"Error during analysis of Cycle {cycle}: {e}")
            finally:
                if pcap_path and os.path.exists(pcap_path):
                    try:
                        os.remove(pcap_path)
                    except Exception:
                        pass
                analysis_queue.task_done()
        except queue.Empty:
            continue

def capture_loop():
    """Main capture loop running in background."""
    global is_running
    cycle = 1
    while is_running:
        pcap_path = capture_traffic(cycle)
        if pcap_path:
            try:
                analysis_queue.put_nowait((cycle, pcap_path))
            except queue.Full:
                msg = f"Queue full! Analyzer is lagging. Dropping Cycle {cycle} to maintain real-time monitoring."
                logger.warning(msg)
                socketio.emit('log', {'timestamp': time.strftime('%H:%M:%S'), 'level': 'WARNING', 'message': msg})
                try:
                    os.remove(pcap_path)
                except:
                    pass
        cycle += 1

# ─── FLASK ROUTES ──────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', interface=INTERFACE, capture_time=CAPTURE_TIME)

@app.route('/api/status')
def status():
    return jsonify({'is_running': is_running})

@app.route('/api/start', methods=['POST'])
def start_ids():
    global is_running, INTERFACE, CAPTURE_TIME
    
    if is_running:
        return jsonify({'status': 'already running'})
        
    data = request.json
    if 'interface' in data and data['interface']:
        INTERFACE = data['interface']
    if 'capture_time' in data and data['capture_time']:
        try:
            CAPTURE_TIME = int(data['capture_time'])
        except ValueError:
            pass

    is_running = True
    
    # Start background threads
    global analyzer_thread, capture_thread
    analyzer_thread = threading.Thread(target=analysis_worker)
    analyzer_thread.start()
    capture_thread = threading.Thread(target=capture_loop)
    capture_thread.start()
    
    socketio.emit('log', {'timestamp': time.strftime('%H:%M:%S'), 'level': 'INFO', 'message': f"IDS Started on interface {INTERFACE}"})
    return jsonify({'status': 'started'})

# Global thread references to prevent duplicates
analyzer_thread = None
capture_thread = None

@app.route('/api/stop', methods=['POST'])
def stop_ids():
    global is_running, analyzer_thread, capture_thread
    is_running = False
    socketio.emit('log', {'timestamp': time.strftime('%H:%M:%S'), 'level': 'INFO', 'message': "IDS Stopping... Waiting for current cycle to finish."})
    
    # Empty queue
    while not analysis_queue.empty():
        try:
            analysis_queue.get_nowait()
        except queue.Empty:
            break
            
    # Send sentinel
    analysis_queue.put((None, None))
    
    # Wait for threads to finish so they don't overlap with a new start
    if capture_thread and capture_thread.is_alive():
        capture_thread.join(timeout=11)
    if analyzer_thread and analyzer_thread.is_alive():
        analyzer_thread.join(timeout=5)
        
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    # Lazy load models in the main process only to prevent multiprocessing conflicts on Windows
    load_models()
    
    # Run the web server
    logger.info("Starting Web Server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
