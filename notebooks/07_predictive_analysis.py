"""
SpaceX Falcon 9 Landing Prediction - Predictive Modeling & Classification
IBM Data Science Professional Certificate Capstone Project

This module implements a comprehensive machine learning pipeline for predicting
SpaceX Falcon 9 first-stage landing success. It trains and evaluates multiple
classification models, performs hyperparameter tuning, and compares performance
to identify the best predictor.

Models Implemented:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)

Key Features:
- Data preprocessing (one-hot encoding, feature scaling)
- Train-test split (80-20)
- Hyperparameter tuning with GridSearchCV
- Model evaluation (accuracy, precision, recall, F1)
- Confusion matrices and classification reports
- Comparative model performance analysis

Author: Devendra Nath (devendranath997)
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = "../data/processed/spacex_launches_processed.csv"
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Model configurations
MODEL_CONFIGS = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    'SVM': {
        'model': SVC(random_state=RANDOM_STATE),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'params': {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading & Preparation
# ══════════════════════════════════════════════════════════════════════════════

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load processed data from CSV.

    Args:
        file_path: Path to input CSV

    Returns:
        Loaded DataFrame
    """
    print("\n" + "="*80)
    print("LOADING DATA FOR PREDICTIVE MODELING")
    print("="*80)

    try:
        df = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and target variable (y) from DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (features_df, target_series)
    """
    print("\n" + "="*80)
    print("PREPARING FEATURES AND TARGET")
    print("="*80)

    # Check for target variable
    if 'Class' not in df.columns:
        print("✗ Target variable 'Class' not found")
        return None, None

    # Separate features and target
    y = df['Class'].copy()
    X = df.drop(['Class'], axis=1)

    # Remove non-predictive columns
    cols_to_drop = []
    for col in X.columns:
        if col.lower() in ['id', 'index', 'unnamed: 0']:
            cols_to_drop.append(col)

    if cols_to_drop:
        X = X.drop(cols_to_drop, axis=1)
        print(f"✓ Dropped non-predictive columns: {cols_to_drop}")

    print(f"✓ Features prepared: {X.shape[1]} features")
    print(f"✓ Target prepared: {y.shape[0]} samples")
    print(f"  Target distribution:")
    print(f"    Class 0 (Failure): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"    Class 1 (Success): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")

    return X, y


def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess features: one-hot encoding and standardization.

    Args:
        X: Feature DataFrame

    Returns:
        Preprocessed feature array
    """
    print("\n" + "="*80)
    print("PREPROCESSING FEATURES")
    print("="*80)

    X_processed = X.copy()

    # Identify categorical and numerical columns
    categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()

    print(f"✓ Identified {len(categorical_cols)} categorical columns")
    print(f"✓ Identified {len(numerical_cols)} numerical columns")

    # One-hot encode categorical variables
    if len(categorical_cols) > 0:
        print(f"\nOne-hot encoding categorical variables...")
        X_encoded = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
        print(f"✓ One-hot encoding complete")
        print(f"  Features expanded to: {X_encoded.shape[1]}")
    else:
        X_encoded = X_processed
        print("✓ No categorical variables to encode")

    # Standardize numerical features
    print(f"\nStandardizing numerical features with StandardScaler...")
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    print(f"✓ Standardization complete")

    return X_encoded


def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """
    Split data into training and testing sets.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )

    print(f"✓ Data split successfully")
    print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[1]} features)")
    print(f"  Testing set: {X_test.shape[0]} samples ({X_test.shape[1]} features)")
    print(f"\n  Training set target distribution:")
    print(f"    Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"    Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# Model Training & Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════

def train_and_tune_model(model_name: str, model_config: Dict,
                         X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
    """
    Train model and perform hyperparameter tuning using GridSearchCV.

    Args:
        model_name: Name of the model
        model_config: Configuration dictionary with model and parameters
        X_train: Training features
        y_train: Training target

    Returns:
        Tuple of (best_model, grid_search_object)
    """
    print(f"\n{'─'*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'─'*80}")

    model = model_config['model']
    params = model_config['params']

    print(f"Performing GridSearchCV with {len(params)} parameter combinations...")

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        model, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"✓ Grid search complete")
    print(f"  Best CV Score (Accuracy): {best_score:.4f}")
    print(f"  Best Parameters:")
    for param, value in best_params.items():
        print(f"    - {param}: {value}")

    return best_model, grid_search


# ══════════════════════════════════════════════════════════════════════════════
# Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: pd.Series, y_test: pd.Series, model_name: str) -> Dict:
    """
    Evaluate trained model on both training and testing sets.

    Args:
        model: Trained model
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        model_name: Name of model for reporting

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'─'*80}")
    print(f"EVALUATION: {model_name}")
    print(f"{'─'*80}")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Print results
    print(f"\nAccuracy Scores:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    print(f"\nTesting Set Metrics:")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Failure', 'Success']))

    return {
        'model_name': model_name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'confusion_matrix': cm,
        'model': model
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(results: list) -> None:
    """
    Compare all trained models and identify the best performer.

    Args:
        results: List of evaluation result dictionaries
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Train Accuracy': f"{r['train_accuracy']:.4f}",
            'Test Accuracy': f"{r['test_accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1_score']:.4f}"
        }
        for r in results
    ])

    print("\n" + comparison_df.to_string(index=False))

    # Find best model
    best_idx = np.argmax([r['test_accuracy'] for r in results])
    best_result = results[best_idx]

    print("\n" + "="*80)
    print("BEST PERFORMING MODEL")
    print("="*80)
    print(f"✓ Model: {best_result['model_name']}")
    print(f"  Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    print(f"  F1-Score: {best_result['f1_score']:.4f}")

    return best_result


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main execution function for predictive modeling pipeline.
    """
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "SpaceX Falcon 9 - Predictive Modeling & Classification".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)

    start_time = datetime.now()
    print(f"\nExecution started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data(DATA_PATH)
    if df is None:
        print("✗ Failed to load data. Exiting.")
        return

    # Prepare features and target
    X, y = prepare_features_and_target(df)
    if X is None or y is None:
        print("✗ Failed to prepare features and target. Exiting.")
        return

    # Preprocess data
    X_processed = preprocess_data(X)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_processed, y)

    # Train and evaluate all models
    results = []

    for model_name, model_config in MODEL_CONFIGS.items():
        print("\n" + "█"*80)

        # Train model with hyperparameter tuning
        best_model, grid_search = train_and_tune_model(
            model_name, model_config, X_train, y_train
        )

        # Evaluate model
        evaluation_result = evaluate_model(
            best_model, X_train, X_test, y_train, y_test, model_name
        )

        results.append(evaluation_result)

    # Compare all models
    print("\n" + "█"*80)
    best_model_result = compare_models(results)

    # Final summary
    print("\n" + "="*80)
    print("PREDICTIVE ANALYSIS SUMMARY")
    print("="*80)
    print(f"✓ Models trained and evaluated: {len(results)}")
    print(f"  - Logistic Regression")
    print(f"  - Support Vector Machine (SVM)")
    print(f"  - Decision Tree Classifier")
    print(f"  - K-Nearest Neighbors (KNN)")
    print(f"\n✓ Best Model: {best_model_result['model_name']}")
    print(f"  Test Accuracy: {best_model_result['test_accuracy']*100:.2f}%")
    print(f"\nNote: Decision Tree and SVM typically perform best for this dataset,")
    print(f"with success prediction accuracy in the 70-80% range depending on features.")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n✓ Execution completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
