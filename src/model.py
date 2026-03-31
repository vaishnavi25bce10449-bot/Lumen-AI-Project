"""
# LUMEN AI - Model Logic
Using Isolation Forest for Unsupervised Outlier Detection.
"""

from sklearn.ensemble import IsolationForest
import pandas as pd

def train_and_predict(features_df, sensitivity=0.1):
    """
    Isolation Forest algorithm apply karta hai.
    -1 = Anomaly (Outlier)
    1  = Normal Data
    """
    # Model define karna
    # 'contamination' parameter batata hai ki kitne % anomalies expected hain
    clf = IsolationForest(contamination=sensitivity, random_state=42, n_estimators=100)
    
    # Fit and Predict on numerical columns only
    # Hum 'RawText' column ko exclude karenge prediction ke waqt
    numeric_cols = ['Severity', 'Length', 'Complexity']
    features_df['anomaly_score'] = clf.fit_predict(features_df[numeric_cols])
    
    return features_df
