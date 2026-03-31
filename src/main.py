"""
# LUMEN AI - Main CLI Entry Point
## Author: VIT Student (BYOP Project)
## Description: AI-based Log Anomaly Detector using Isolation Forest.
"""

import os
import sys
import argparse
import pandas as pd

# --- Path Fix for src module ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import train_and_predict
except ImportError:
    # Fallback import agar environment path different ho
    from src.model import train_and_predict

def parse_log_file(file_path):
    """Raw logs ko numerical features mein convert karta hai"""
    if not os.path.exists(file_path):
        print(f"[-] Error: File '{file_path}' nahi mili. Path check karein.")
        sys.exit(1)
        
    extracted_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Feature 1: Severity (ERROR/CRITICAL/WARN)
            severity = 0
            if "CRITICAL" in line: severity = 3
            elif "ERROR" in line: severity = 2
            elif "WARN" in line: severity = 1
            
            # Feature 2: Length of the log message
            log_length = len(line)
            
            # Feature 3: Complexity (Unique characters)
            complexity = len(set(line))
            
            extracted_data.append([severity, log_length, complexity, line])
            
    return pd.DataFrame(extracted_data, columns=['Severity', 'Length', 'Complexity', 'RawText'])

def main():
    # CLI Command Line Arguments Setup
    parser = argparse.ArgumentParser(description="LUMEN: AI Log Anomaly Detector")
    parser.add_argument("--file", required=True, help="Path to the log file (e.g., data/sample_logs.txt)")
    parser.add_argument("--threshold", type=float, default=0.1, help="Anomaly sensitivity (0.01 to 0.2)")
    
    args = parser.parse_args()

    print("\n" + "="*50)
    print("      LUMEN AI: INITIALIZING LOG SCANNER")
    print("="*50)
    
    # 1. Parsing Logs
    print(f"[*] Reading file: {args.file}...")
    df = parse_log_file(args.file)
    
    # 2. Running AI Model
    print(f"[*] Analyzing {len(df)} entries using Isolation Forest...")
    processed_df = train_and_predict(df, args.threshold)
    
    # 3. Filtering Results
    # -1 is the label for Anomaly in Isolation Forest
    anomalies = processed_df[processed_df['anomaly_score'] == -1]
    
    # 4. Final Output
    print("-" * 50)
    print(f"[*] Total Scanned  : {len(df)}")
    print(f"[*] Anomalies Found : {len(anomalies)}")
    print("-" * 50)

    if not anomalies.empty:
        print("\n[!] WARNING: Potential Security or System Risks:")
        # Sirf pehle 10 anomalies dikhayega taaki screen bhar na jaye
        for log in anomalies['RawText'].head(10):
            print(f" -> {log}")
    else:
        print("\n[+] SUCCESS: System status is normal. No anomalies detected.")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
