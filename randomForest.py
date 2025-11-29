"""
Random Forest Classification Analysis
Comparing different feature sets: selected features, t-SNE, and all features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


# ============================================================================
# CONFIGURATION
# ============================================================================

SELECTED_FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 100


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_selected_features_data():
    """Load data with selected features only."""
    learn_df = pd.read_csv("original_selected_features_data/learn_valid_80.csv")
    test_df = pd.read_csv("original_selected_features_data/test_20.csv")
    
    X_learn = learn_df[SELECTED_FEATURES]
    y_learn = learn_df["label"]
    X_test = test_df[SELECTED_FEATURES]
    y_test = test_df["label"]
    
    return X_learn, y_learn, X_test, y_test


def load_tsne_data():
    """Load t-SNE transformed data."""
    learn_df = pd.read_csv("tsne_data/learn_valid_80.csv")
    test_df = pd.read_csv("tsne_data/test_20.csv")
    
    X_learn = learn_df[["tsne_x", "tsne_y"]]
    y_learn = learn_df["label"]
    X_test = test_df[["tsne_x", "tsne_y"]]
    y_test = test_df["label"]
    
    return X_learn, y_learn, X_test, y_test


def load_all_features_data():
    """Load data with all numeric features."""
    learn_df = pd.read_csv("original_all_features_data/learn_valid_80.csv")
    test_df = pd.read_csv("original_all_features_data/test_20.csv")
    
    # Keep only numeric columns
    numeric_cols = learn_df.select_dtypes(include=[np.number]).columns
    feature_cols = numeric_cols.drop("label") if "label" in numeric_cols else numeric_cols
    
    X_learn = learn_df[feature_cols]
    y_learn = learn_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]
    
    return X_learn, y_learn, X_test, y_test, feature_cols


def load_additional_test_data(feature_cols):
    """Load additional 1000-sample test set."""
    test_1000_df = pd.read_csv("tsne_data/test_1000.csv")
    X_test_1000 = test_1000_df[feature_cols]
    y_test_1000 = test_1000_df["label"]
    return X_test_1000, y_test_1000


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_and_evaluate_model(X_learn, y_learn, X_test, y_test, model_name="Model"):
    """Train a Random Forest model and evaluate on train/validation/test sets."""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}\n")
    
    # Split into train and validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_learn, y_learn, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        shuffle=True, stratify=y_learn
    )
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Validation evaluation
    y_pred_valid = rf.predict(X_valid)
    print("VALIDATION RESULTS:")
    print(f"Accuracy: {accuracy_score(y_valid, y_pred_valid):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_valid, y_pred_valid)}")
    print(f"\nClassification Report:\n{classification_report(y_valid, y_pred_valid)}")
    
    # Test evaluation
    y_pred_test = rf.predict(X_test)
    print("\n" + "-"*80)
    print("TEST RESULTS:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred_test)}")
    
    return rf, X_train, X_valid, y_train, y_valid


def optimize_model(X_train, y_train, X_valid, y_valid):
    """Perform grid search to find optimal hyperparameters."""
    print(f"\n{'='*80}")
    print("Optimizing Model with Grid Search")
    print(f"{'='*80}\n")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model on validation set
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_valid)
    print(f"\nValidation accuracy: {accuracy_score(y_valid, y_pred):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_valid, y_pred)}")
    print(f"Classification report:\n{classification_report(y_valid, y_pred)}")
    
    return best_rf


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_feature_importance(model, feature_cols, top_n=20):
    """Plot feature importance from trained model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], 
               rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print(f"\nTop {min(top_n, len(importances))} most important features:")
    for i in range(min(top_n, len(importances))):
        print(f"{i+1}. {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")


def plot_tsne_decision_boundary(X_train, y_train, X_valid, y_valid, rf_model):
    """Visualize decision boundaries for t-SNE data (2D only)."""
    X_all = pd.concat([X_train, X_valid], axis=0)
    
    # Create grid
    pad = 1.0
    x_min = X_all.iloc[:, 0].quantile(0.01) - pad
    x_max = X_all.iloc[:, 0].quantile(0.99) + pad
    y_min = X_all.iloc[:, 1].quantile(0.01) - pad
    y_max = X_all.iloc[:, 1].quantile(0.99) + pad
    
    h = 0.25
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = rf_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Plot
    bg_cmap = ListedColormap(["#f46d43", "#66bd63"])
    
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(xx, yy, Z, cmap=bg_cmap, shading="nearest", alpha=0.35)
    
    classes = np.unique(pd.concat([y_train, y_valid]))
    point_colors = {c: bg_cmap.colors[i % len(bg_cmap.colors)] for i, c in enumerate(classes)}
    
    for cls in classes:
        m = y_valid == cls
        plt.scatter(X_valid.loc[m].iloc[:, 0], X_valid.loc[m].iloc[:, 1],
                    c=point_colors[cls], edgecolors="k", s=25, marker="s", 
                    label=f"Class {cls}")
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Random Forest (t-SNE space) - validation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pca_projection(X_train, y_train, X_test, y_test):
    """Visualize data in PCA space to check class separability."""
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    plt.figure(figsize=(12, 5))
    
    # Training data
    plt.subplot(1, 2, 1)
    for label in [0, 1]:
        mask = y_train == label
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], 
                    alpha=0.5, label=f'Class {label}', s=20)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Training Data - PCA Projection')
    plt.legend()
    
    # Test data
    plt.subplot(1, 2, 2)
    for label in [0, 1]:
        mask = y_test == label
        plt.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], 
                    alpha=0.5, label=f'Class {label}', s=20)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Test Data - PCA Projection')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained by 2 components: {pca.explained_variance_ratio_.sum():.3f}")


def plot_prediction_confidence(y_test, y_pred_proba):
    """Analyze and visualize prediction confidence."""
    confidence = np.max(y_pred_proba, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    plt.figure(figsize=(10, 4))
    
    # Confidence distribution
    plt.subplot(1, 2, 1)
    plt.hist(confidence, bins=50, edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence')
    plt.axvline(confidence.mean(), color='red', linestyle='--', 
                label=f'Mean: {confidence.mean():.3f}')
    plt.legend()
    
    # Confidence by correctness
    plt.subplot(1, 2, 2)
    plt.boxplot([confidence[y_pred == y_test], confidence[y_pred != y_test]], 
                labels=['Correct', 'Incorrect'])
    plt.ylabel('Prediction Confidence')
    plt.title('Confidence: Correct vs Incorrect Predictions')
    plt.tight_layout()
    plt.show()
    
    print(f"\nMean prediction confidence: {confidence.mean():.3f}")
    print(f"Median prediction confidence: {np.median(confidence):.3f}")
    print(f"Predictions with >0.95 confidence: {(confidence > 0.95).sum()} / {len(confidence)} "
          f"({(confidence > 0.95).mean()*100:.1f}%)")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_misclassifications(X_test, y_test, model):
    """Analyze misclassified samples."""
    print(f"\n{'='*80}")
    print("MISCLASSIFICATION ANALYSIS")
    print(f"{'='*80}\n")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    misclassified_mask = y_test != y_pred
    print(f"Total misclassified: {misclassified_mask.sum()} out of {len(y_test)}")
    print(f"Misclassification rate: {misclassified_mask.sum() / len(y_test) * 100:.2f}%\n")
    
    if misclassified_mask.sum() > 0:
        print("Correctly Classified Statistics:")
        print(X_test[~misclassified_mask].describe())
        
        print("\nMisclassified Points Statistics:")
        print(X_test[misclassified_mask].describe())
        
        misclassified_proba = y_pred_proba[misclassified_mask]
        print("\nPrediction Confidence for Misclassified Points:")
        print(f"Mean confidence: {np.max(misclassified_proba, axis=1).mean():.3f}")
        print(f"Min confidence: {np.max(misclassified_proba, axis=1).min():.3f}")
        print(f"Max confidence: {np.max(misclassified_proba, axis=1).max():.3f}")


def compare_feature_sets(X_test, y_test, model_all, model_selected):
    """Compare performance between all features and selected features."""
    print(f"\n{'='*80}")
    print("COMPARISON: ALL FEATURES vs SELECTED FEATURES")
    print(f"{'='*80}\n")
    
    y_pred_all = model_all.predict(X_test)
    y_pred_selected = model_selected.predict(X_test[SELECTED_FEATURES])
    
    print("Performance Comparison:")
    print("\nAll Features:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_all):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_all, average='weighted'):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_all, average='weighted'):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_all, average='weighted'):.4f}")
    
    print("\nSelected 6 Features:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_selected):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_selected, average='weighted'):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_selected, average='weighted'):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred_selected, average='weighted'):.4f}")


def analyze_feature_importance(model, feature_cols):
    """Detailed feature importance analysis."""
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}\n")
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 most important features:")
    print(feature_importance.head(15))
    
    print(f"\nSelected 6 features ranking:")
    for feat in SELECTED_FEATURES:
        if feat in feature_cols:
            rank = feature_importance[feature_importance['feature'] == feat].index[0] + 1
            importance = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
            print(f"  {feat}: Rank {rank}, Importance: {importance:.4f}")
    
    # Cumulative importance
    cumulative_importance = np.cumsum(feature_importance['importance'].values)
    n_features_95 = np.argmax(cumulative_importance >= 0.95) + 1
    print(f"\nNumber of features needed for 95% importance: {n_features_95} / {len(feature_cols)}")
    
    return feature_importance, cumulative_importance


def evaluate_additional_test_set(X_test_1000, y_test_1000, model, 
                                   X_test_200, y_test_200):
    """Evaluate model on additional 1000-sample test set."""
    print(f"\n{'='*80}")
    print("TESTING ON ADDITIONAL 1000 SAMPLES")
    print(f"{'='*80}\n")
    
    y_pred_1000 = model.predict(X_test_1000)
    y_pred_proba_1000 = model.predict_proba(X_test_1000)
    
    print(f"Loaded {len(X_test_1000)} samples for additional testing")
    print("\nResults on 1000-sample test set:")
    print(f"Accuracy:  {accuracy_score(y_test_1000, y_pred_1000):.4f}")
    print(f"Precision: {precision_score(y_test_1000, y_pred_1000, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test_1000, y_pred_1000, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_test_1000, y_pred_1000, average='weighted'):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_1000, y_pred_1000))
    
    print("\nClassification Report:")
    print(classification_report(y_test_1000, y_pred_1000))
    
    # Comparison
    y_pred_200 = model.predict(X_test_200)
    print("\n" + "="*80)
    print("COMPARISON: 200-sample vs 1000-sample test sets")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Test 200': [
            accuracy_score(y_test_200, y_pred_200),
            precision_score(y_test_200, y_pred_200, average='weighted'),
            recall_score(y_test_200, y_pred_200, average='weighted'),
            f1_score(y_test_200, y_pred_200, average='weighted')
        ],
        'Test 1000': [
            accuracy_score(y_test_1000, y_pred_1000),
            precision_score(y_test_1000, y_pred_1000, average='weighted'),
            recall_score(y_test_1000, y_pred_1000, average='weighted'),
            f1_score(y_test_1000, y_pred_1000, average='weighted')
        ]
    })
    comparison_df['Difference'] = comparison_df['Test 1000'] - comparison_df['Test 200']
    print(comparison_df.to_string(index=False))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("RANDOM FOREST CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # 1. Train with selected features
    print("\n[1/4] Training with selected features...")
    X_learn_sel, y_learn_sel, X_test_sel, y_test_sel = load_selected_features_data()
    rf_selected, _, _, _, _ = train_and_evaluate_model(
        X_learn_sel, y_learn_sel, X_test_sel, y_test_sel,
        "Random Forest with Selected Features"
    )
    
    # 2. Train with t-SNE features
    print("\n[2/4] Training with t-SNE features...")
    X_learn_tsne, y_learn_tsne, X_test_tsne, y_test_tsne = load_tsne_data()
    rf_tsne, X_train_tsne, X_valid_tsne, y_train_tsne, y_valid_tsne = train_and_evaluate_model(
        X_learn_tsne, y_learn_tsne, X_test_tsne, y_test_tsne,
        "Random Forest with t-SNE Features"
    )
    
    # 3. Train with all features
    print("\n[3/4] Training with all features...")
    X_learn_all, y_learn_all, X_test_all, y_test_all, feature_cols = load_all_features_data()
    rf_all, X_train_all, X_valid_all, y_train_all, y_valid_all = train_and_evaluate_model(
        X_learn_all, y_learn_all, X_test_all, y_test_all,
        "Random Forest with All Features"
    )
    
    # 4. Optimize the all-features model
    print("\n[4/4] Optimizing all-features model...")
    best_rf = optimize_model(X_train_all, y_train_all, X_valid_all, y_valid_all)
    
    # Visualizations and analysis
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS AND ANALYSIS")
    print("="*80)
    
    plot_feature_importance(best_rf, feature_cols)
    plot_pca_projection(X_train_all, y_train_all, X_test_all, y_test_all)
    
    y_pred_proba = best_rf.predict_proba(X_test_all)
    plot_prediction_confidence(y_test_all, y_pred_proba)
    
    # t-SNE decision boundary
    if len(X_train_tsne.columns) == 2:
        plot_tsne_decision_boundary(X_train_tsne, y_train_tsne, 
                                      X_valid_tsne, y_valid_tsne, rf_tsne)
    
    # Analyses
    analyze_misclassifications(X_test_all, y_test_all, best_rf)
    compare_feature_sets(X_test_all, y_test_all, best_rf, rf_selected)
    analyze_feature_importance(best_rf, feature_cols)
    
    # Additional test set (if available)
    try:
        X_test_1000, y_test_1000 = load_additional_test_data(feature_cols)
        evaluate_additional_test_set(X_test_1000, y_test_1000, best_rf,
                                       X_test_all, y_test_all)
    except FileNotFoundError:
        print("\n[INFO] Additional 1000-sample test set not found. Skipping...")
    except Exception as e:
        print(f"\n[ERROR] Failed to load additional test set: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()