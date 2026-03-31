"""
LUMEN AI - Model Logic PROJECT
Using Isolation Forest for Unsupervised Outlier Detection.
"""

from sklearn.ensemble import IsolationForest
import pandas as pd

def train_and_predict(features_df, sensitivity=0.1):
    """
    Isolation Forest algorithm apply karta hai.
    """
    # Model define karna
    clf = IsolationForest(contamination=sensitivity, random_state=42, n_estimators=100)
    
    # Fit and Predict on numerical columns only
    numeric_cols = ['Severity', 'Length', 'Complexity']
    features_df['anomaly_score'] = clf.fit_predict(features_df[numeric_cols])
    
    return features_df
