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
import xgboost as xgb_module
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
        xgb = xgb_module.XGBClassifier()
        xgb.load_model(os.path.join(MODEL_PATH, 'xgboost_model.json'))
        iso      = joblib.load(os.path.join(MODEL_PATH, 'isolation_forest.pkl'))
        scaler   = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        features = joblib.load(os.path.join(MODEL_PATH, 'feature_names.pkl'))
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Debugging aid for corrupted files
        for model_file in ['random_forest.pkl', 'xgboost_model.json', 'isolation_forest.pkl', 'scaler.pkl', 'feature_names.pkl']:
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
    """Convert pcap to flow features using cicflowmeter CLI"""
    try:
        if not os.path.exists(pcap_path):
            return None, None
            
        # Generate temporary CSV path
        base, _ = os.path.splitext(pcap_path)
        csv_path = base + ".csv"

        # Use pure python PcapReader and FlowSession to bypass Scapy's tcpdump requirement on Windows
        from cicflowmeter.flow_session import FlowSession
        from scapy.all import PcapReader

        session = FlowSession(output_mode="csv", output=csv_path)

        total_packets = 0
        tcp_udp_packets = 0
        try:
            for pkt in PcapReader(pcap_path):
                total_packets += 1
                # Only process IP and (TCP or UDP) packets
                if pkt.haslayer("IP") and (pkt.haslayer("TCP") or pkt.haslayer("UDP")):
                    tcp_udp_packets += 1
                    session.process(pkt)
        finally:
            session.flush_flows()

        logger.info(f"DEBUG PCAP: {pcap_path} contained {total_packets} total packets. {tcp_udp_packets} were TCP/UDP.")

        if not os.path.exists(csv_path):
            logger.warning(f"DEBUG PCAP: CSV file was not generated at {csv_path}")
            return None, csv_path

        # Read the generated CSV
        df = pd.read_csv(csv_path)

        # Strip leading/trailing spaces from column names
        df.columns = df.columns.str.strip()

        logger.info(f"DEBUG PCAP: CSV generated with {len(df)} flow rows.")

        if df is None or len(df) == 0:
            return None, csv_path

        return df, csv_path

    except Exception as e:
        logger.error(f"Flow extraction error: {e}")
        return None, locals().get('csv_path', None)

def preprocess_flows(df):
    """Process cicflowmeter columns to match training features and handle missing values/infinities"""
    col_map = {
        'protocol': 'Protocol',
        'flow_duration': 'Flow Duration',
        'tot_fwd_pkts': 'Total Fwd Packets',
        'tot_bwd_pkts': 'Total Backward Packets',
        'totlen_fwd_pkts': 'Total Length of Fwd Packets',
        'totlen_bwd_pkts': 'Total Length of Bwd Packets',
        'fwd_pkt_len_max': 'Fwd Packet Length Max',
        'fwd_pkt_len_min': 'Fwd Packet Length Min',
        'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
        'fwd_pkt_len_std': 'Fwd Packet Length Std',
        'bwd_pkt_len_max': 'Bwd Packet Length Max',
        'bwd_pkt_len_min': 'Bwd Packet Length Min',
        'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
        'bwd_pkt_len_std': 'Bwd Packet Length Std',
        'flow_byts_s': 'Flow Bytes/s',
        'flow_pkts_s': 'Flow Packets/s',
        'flow_iat_mean': 'Flow IAT Mean',
        'flow_iat_std': 'Flow IAT Std',
        'flow_iat_max': 'Flow IAT Max',
        'flow_iat_min': 'Flow IAT Min',
        'fwd_iat_tot': 'Fwd IAT Total',
        'fwd_iat_mean': 'Fwd IAT Mean',
        'fwd_iat_std': 'Fwd IAT Std',
        'fwd_iat_max': 'Fwd IAT Max',
        'fwd_iat_min': 'Fwd IAT Min',
        'bwd_iat_tot': 'Bwd IAT Total',
        'bwd_iat_mean': 'Bwd IAT Mean',
        'bwd_iat_std': 'Bwd IAT Std',
        'bwd_iat_max': 'Bwd IAT Max',
        'bwd_iat_min': 'Bwd IAT Min',
        'fwd_psh_flags': 'Fwd PSH Flags',
        'bwd_psh_flags': 'Bwd PSH Flags',
        'fwd_urg_flags': 'Fwd URG Flags',
        'bwd_urg_flags': 'Bwd URG Flags',
        'fwd_header_len': 'Fwd Header Length',
        'bwd_header_len': 'Bwd Header Length',
        'fwd_pkts_s': 'Fwd Packets/s',
        'bwd_pkts_s': 'Bwd Packets/s',
        'pkt_len_min': 'Min Packet Length',
        'pkt_len_max': 'Max Packet Length',
        'pkt_len_mean': 'Packet Length Mean',
        'pkt_len_std': 'Packet Length Std',
        'pkt_len_var': 'Packet Length Variance',
        'fin_flag_cnt': 'FIN Flag Count',
        'syn_flag_cnt': 'SYN Flag Count',
        'rst_flag_cnt': 'RST Flag Count',
        'psh_flag_cnt': 'PSH Flag Count',
        'ack_flag_cnt': 'ACK Flag Count',
        'urg_flag_cnt': 'URG Flag Count',
        'ece_flag_cnt': 'ECE Flag Count',
        'down_up_ratio': 'Down/Up Ratio',
        'pkt_size_avg': 'Average Packet Size',
        'fwd_seg_size_avg': 'Avg Fwd Segment Size',
        'bwd_seg_size_avg': 'Avg Bwd Segment Size',
        'fwd_byts_b_avg': 'Fwd Avg Bytes/Bulk',
        'fwd_pkts_b_avg': 'Fwd Avg Packets/Bulk',
        'fwd_blk_rate_avg': 'Fwd Avg Bulk Rate',
        'bwd_byts_b_avg': 'Bwd Avg Bytes/Bulk',
        'bwd_pkts_b_avg': 'Bwd Avg Packets/Bulk',
        'bwd_blk_rate_avg': 'Bwd Avg Bulk Rate',
        'subflow_fwd_pkts': 'Subflow Fwd Packets',
        'subflow_fwd_byts': 'Subflow Fwd Bytes',
        'subflow_bwd_pkts': 'Subflow Bwd Packets',
        'subflow_bwd_byts': 'Subflow Bwd Bytes',
        'init_fwd_win_byts': 'Init_Win_bytes_forward',
        'init_bwd_win_byts': 'Init_Win_bytes_backward',
        'fwd_act_data_pkts': 'act_data_pkt_fwd',
        'fwd_seg_size_min': 'min_seg_size_forward',
        'active_mean': 'Active Mean',
        'active_std': 'Active Std',
        'active_max': 'Active Max',
        'active_min': 'Active Min',
        'idle_mean': 'Idle Mean',
        'idle_std': 'Idle Std',
        'idle_max': 'Idle Max',
        'idle_min': 'Idle Min',
        'cwr_flag_count': 'CWE Flag Count'
    }

    df = df.rename(columns=col_map)

    # The Python cicflowmeter outputs time in seconds, but the ML models (trained on CIC-DDoS2019)
    # expect time features to be in microseconds.
    time_cols = [
        'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ]

    for tc in time_cols:
        if tc in df.columns:
            df[tc] = df[tc] * 1000000.0

    # Custom dataset additions
    if 'Fwd Header Length' in df.columns:
        df['Fwd Header Length.1'] = df['Fwd Header Length']
    df['Inbound'] = 1

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

    # Increased thresholds to reduce false positives on live traffic
    supervised_alert = rf_ratio >= 0.50 and xgb_ratio >= 0.50 and enough_data
    high_confidence = rf_ratio >= 0.75 and xgb_ratio >= 0.75 and enough_data
    
    # Determine Final Verdict
    status = "normal"
    verdict = "BENIGN (Within normal range)"
    
    if high_confidence and enough_data:
        status = "danger"
        verdict = f"DDoS ATTACK DETECTED! (RF: {rf_ratio:.1%} | XGB: {xgb_ratio:.1%} | ISO: {iso_ratio:.1%} flagged)"
    elif supervised_alert and enough_data:
        status = "warning"
        verdict = f"SUSPICIOUS TRAFFIC (RF: {rf_ratio:.1%} | XGB: {xgb_ratio:.1%} | ISO: {iso_ratio:.1%} flagged)"
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
            csv_path = None
            try:
                raw_df, csv_path = pcap_to_flows(pcap_path)
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
                if csv_path and os.path.exists(csv_path):
                    try:
                        os.remove(csv_path)
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
