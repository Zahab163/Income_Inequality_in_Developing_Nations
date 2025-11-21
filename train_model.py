# train_enhanced_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scikitplot as skplt
from imblearn.over_sampling import SMOTE
import os
import warnings
warnings.filterwarnings('ignore')

file_path = "C:\Income_Inequality_Predictions\income_data - data.csv"

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset with enhanced preprocessing"""
    print("Loading and preprocessing dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Display target distribution
    target_dist = df['income_above_limit'].value_counts()
    print(f"\n Target distribution:\n{target_dist}")
    print(f"Target ratio: {target_dist['Above limit']/len(df)*100:.2f}% Above limit")
    
    # Create copy for preprocessing
    df_clean = df.copy()
    
    # Convert target to binary
    df_clean['income_above_limit'] = df_clean['income_above_limit'].map({
        'Below limit': 0, 
        'Above limit': 1
    })
    
    # Drop ID column
    if 'ID' in df_clean.columns:
        df_clean = df_clean.drop('ID', axis=1)
    
    # Handle missing values
    print("\n Handling missing values...")
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Advanced missing value handling
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
    
    print(f"Missing values after preprocessing: {df_clean.isnull().sum().sum()}")
    
    return df_clean

def create_advanced_preprocessor(X):
    """Create advanced preprocessing pipeline"""
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    return preprocessor, categorical_cols, numerical_cols

def train_multiple_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate multiple models"""
    print("\n Training Multiple Models...")
    
    # Define models with their parameter grids
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10, 15],
                'classifier__learning_rate': [0.01, 0.1]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'saga']
            }
        },
        'SVM': {
            'model': SVC(random_state=42, class_weight='balanced', probability=True),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'classifier__n_neighbors': [3, 5, 7, 9],
                'classifier__weights': ['uniform', 'distance']
            }
        }
    }
    
    results = {}
    best_models = {}
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline, 
            config['params'], 
            cv=3, 
            scoring='f1', 
            n_jobs=-1, 
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = best_model.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': best_model,
            'f1_score': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'best_params': grid_search.best_params_,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} - F1: {f1:.4f}, Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")
    
    return results, best_models

def create_comprehensive_visualizations(results, X_test, y_test, model_info):
    """Create comprehensive visualizations for model comparison"""
    print("\n Creating comprehensive visualizations...")
    
    # 1. Model Comparison Bar Chart
    models = list(results.keys())
    f1_scores = [results[model]['f1_score'] for model in models]
    accuracies = [results[model]['accuracy'] for model in models]
    auc_scores = [results[model]['roc_auc'] for model in models]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # F1 Scores
    bars1 = ax1.barh(models, f1_scores, color='skyblue')
    ax1.set_xlabel('F1 Score')
    ax1.set_title('Model Comparison - F1 Score')
    ax1.bar_label(bars1, fmt='%.3f')
    
    # Accuracies
    bars2 = ax2.barh(models, accuracies, color='lightgreen')
    ax2.set_xlabel('Accuracy')
    ax2.set_title('Model Comparison - Accuracy')
    ax2.bar_label(bars2, fmt='%.3f')
    
    # AUC Scores
    bars3 = ax3.barh(models, auc_scores, color='salmon')
    ax3.set_xlabel('ROC AUC Score')
    ax3.set_title('Model Comparison - ROC AUC')
    ax3.bar_label(bars3, fmt='%.3f')
    
    # Combined metrics
    x = np.arange(len(models))
    width = 0.25
    ax4.bar(x - width, f1_scores, width, label='F1 Score', color='skyblue')
    ax4.bar(x, accuracies, width, label='Accuracy', color='lightgreen')
    ax4.bar(x + width, auc_scores, width, label='AUC', color='salmon')
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Scores')
    ax4.set_title('Combined Model Metrics')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrices for top 3 models
    top_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, result) in enumerate(top_models):
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues',
                   xticklabels=['Below', 'Above'], 
                   yticklabels=['Below', 'Above'])
        axes[idx].set_title(f'{name}\nF1: {result["f1_score"]:.3f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curves for all models
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = skplt.metrics.roc_curve(y_test, result['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['probabilities'])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Model Comparison')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" Visualizations saved!")

def save_models_and_results(results, model_info, best_overall_model):
    """Save all models and results"""
    print("\n Saving models and results...")
    
    # Save individual models
    for name, result in results.items():
        # Clean model name for filename
        clean_name = name.lower().replace(' ', '_')
        joblib.dump(result['model'], f'model_{clean_name}.pkl')
    
    # Save the best overall model separately
    joblib.dump(best_overall_model, 'best_model.pkl')
    
    # Enhanced model info
    enhanced_model_info = {
        'all_models': list(results.keys()),
        'model_performance': {name: {
            'f1_score': result['f1_score'],
            'accuracy': result['accuracy'],
            'roc_auc': result['roc_auc'],
            'best_params': result['best_params']
        } for name, result in results.items()},
        'best_model': max(results.items(), key=lambda x: x[1]['f1_score'])[0],
        'best_model_f1': max(results.items(), key=lambda x: x[1]['f1_score'])[1]['f1_score'],
        **model_info
    }
    
    joblib.dump(enhanced_model_info, 'enhanced_model_info.pkl')
    
    # Create performance summary
    performance_df = pd.DataFrame({
        'Model': list(results.keys()),
        'F1_Score': [results[model]['f1_score'] for model in results],
        'Accuracy': [results[model]['accuracy'] for model in results],
        'ROC_AUC': [results[model]['roc_auc'] for model in results]
    }).sort_values('F1_Score', ascending=False)
    
    performance_df.to_csv('model_performance_summary.csv', index=False)
    print(" Models and results saved!")

def main():
    # Update this path to your CSV file
    CSV_FILE_PATH = "your_dataset.csv"  # Change to your actual file name
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f" File '{CSV_FILE_PATH}' not found!")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    
    # Load and preprocess data
    df_processed = load_and_preprocess_data(CSV_FILE_PATH)
    
    # Prepare features and target
    X = df_processed.drop('income_above_limit', axis=1)
    y = df_processed['income_above_limit']
    
    # Create preprocessor
    preprocessor, categorical_cols, numerical_cols = create_advanced_preprocessor(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n Data split:")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Positive class in training: {y_train.sum()}/{len(y_train)} ({y_train.mean():.2%})")
    print(f"Positive class in test: {y_test.sum()}/{len(y_test)} ({y_test.mean():.2%})")
    
    # Handle class imbalance with SMOTE
    print("\n Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        preprocessor.fit_transform(X_train), 
        y_train
    )
    
    print(f"After SMOTE - Training set: {X_train_resampled.shape}")
    print(f"After SMOTE - Positive class: {y_train_resampled.sum()}/{len(y_train_resampled)} ({y_train_resampled.mean():.2%})")
    
    # Train multiple models
    results, best_models = train_multiple_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    best_overall_model = best_models[best_model_name]
    
    print(f"\n Best Model: {best_model_name}")
    print(f" Best F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Model info for saving
    model_info = {
        'categorical_columns': categorical_cols,
        'numerical_columns': numerical_cols,
        'all_columns': X.columns.tolist(),
        'feature_names': preprocessor.get_feature_names_out().tolist(),
        'class_distribution_original': dict(y.value_counts()),
        'best_model_name': best_model_name
    }
    
    # Create visualizations
    create_comprehensive_visualizations(results, X_test, y_test, model_info)
    
    # Save models and results
    save_models_and_results(results, model_info, best_overall_model)
    
    print(f"\n Enhanced training completed!")
    print(f" Generated files:")
    print(f"   - 7 trained models (model_*.pkl)")
    print(f"   - Best model (best_model.pkl)")
    print(f"   - Enhanced model info (enhanced_model_info.pkl)")
    print(f"   - Model performance summary (model_performance_summary.csv)")
    print(f"   - 4 visualization files (*.png)")
    print(f"\n You can now run: streamlit run enhanced_app.py")

if __name__ == "__main__":
    main()