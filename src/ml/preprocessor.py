#!/usr/bin/env python3
"""
Preprocessing pipeline for DDoS detection features.
Handles scaling, outlier removal, and feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

class FeaturePreprocessor:
    """Preprocessing pipeline for ML features."""
    
    def __init__(self, use_scaling=True, use_outlier_removal=True, 
                 use_feature_selection=False, n_features=None):
        """
        Initialize preprocessor.
        
        Args:
            use_scaling: Whether to scale features (StandardScaler)
            use_outlier_removal: Whether to clip outliers
            use_feature_selection: Whether to select top features
            n_features: Number of features to select (if feature selection enabled)
        """
        self.use_scaling = use_scaling
        self.use_outlier_removal = use_outlier_removal
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features
        
        self.scaler = StandardScaler() if use_scaling else None
        self.feature_selector = None
        self.feature_names_ = None
        self.is_fitted = False
        
        # Outlier thresholds (percentiles)
        self.outlier_lower = 0.01  # 1st percentile
        self.outlier_upper = 0.99  # 99th percentile
    
    def fit(self, X, y=None):
        """
        Fit preprocessor on training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (optional, needed for feature selection)
        """
        X = np.asarray(X)
        
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        
        # Handle outliers (clip to percentiles)
        if self.use_outlier_removal:
            X = self._remove_outliers(X, fit=True)
        
        # Fit scaler
        if self.use_scaling and self.scaler:
            self.scaler.fit(X)
        
        # Fit feature selector
        if self.use_feature_selection and y is not None:
            if self.n_features is None:
                self.n_features = min(X.shape[1], 20)  # Default to top 20
            
            self.feature_selector = SelectKBest(f_classif, k=self.n_features)
            self.feature_selector.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = np.asarray(X)
        original_shape = X.shape
        
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = None
        
        # Remove outliers
        if self.use_outlier_removal:
            X = self._remove_outliers(X, fit=False)
        
        # Scale features
        if self.use_scaling and self.scaler:
            X = self.scaler.transform(X)
        
        # Feature selection
        if self.use_feature_selection and self.feature_selector:
            X = self.feature_selector.transform(X)
            if feature_names:
                selected_indices = self.feature_selector.get_support(indices=True)
                feature_names = [feature_names[i] for i in selected_indices]
        
        # Convert back to DataFrame if input was DataFrame
        if feature_names:
            X = pd.DataFrame(X, columns=feature_names)
        
        return X
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _remove_outliers(self, X, fit=False):
        """Remove outliers by clipping to percentiles."""
        # Ensure X is 2D array
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if fit:
            # Calculate percentiles on training data (axis=0 means along rows, preserving columns)
            self.lower_bounds_ = np.percentile(X, self.outlier_lower * 100, axis=0, keepdims=False)
            self.upper_bounds_ = np.percentile(X, self.outlier_upper * 100, axis=0, keepdims=False)
        
        # Ensure bounds are broadcastable
        if self.lower_bounds_.ndim == 0:
            self.lower_bounds_ = np.array([self.lower_bounds_])
        if self.upper_bounds_.ndim == 0:
            self.upper_bounds_ = np.array([self.upper_bounds_])
        
        # Clip values to bounds (preserve shape)
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_clipped
    
    def save(self, filepath):
        """Save preprocessor to file."""
        preprocessor_data = {
            'use_scaling': self.use_scaling,
            'use_outlier_removal': self.use_outlier_removal,
            'use_feature_selection': self.use_feature_selection,
            'n_features': self.n_features,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names_': self.feature_names_,
            'is_fitted': self.is_fitted
        }
        
        if self.use_outlier_removal:
            preprocessor_data['lower_bounds_'] = self.lower_bounds_
            preprocessor_data['upper_bounds_'] = self.upper_bounds_
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load preprocessor from file."""
        preprocessor_data = joblib.load(filepath)
        
        self.use_scaling = preprocessor_data['use_scaling']
        self.use_outlier_removal = preprocessor_data['use_outlier_removal']
        self.use_feature_selection = preprocessor_data['use_feature_selection']
        self.n_features = preprocessor_data['n_features']
        self.scaler = preprocessor_data['scaler']
        self.feature_selector = preprocessor_data['feature_selector']
        self.feature_names_ = preprocessor_data['feature_names_']
        self.is_fitted = preprocessor_data['is_fitted']
        
        if self.use_outlier_removal:
            self.lower_bounds_ = preprocessor_data['lower_bounds_']
            self.upper_bounds_ = preprocessor_data['upper_bounds_']
        
        print(f"Preprocessor loaded from {filepath}")
        return self

