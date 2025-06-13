#!/usr/bin/env python3
"""
🚀 Advanced ML Trainer with Animations & Comprehensive Metrics
=============================================================

Features:
- Real-time training animations
- Comprehensive metrics (F1, Precision, Recall, AUC-ROC, etc.)
- Weights & Biases integration
- Statistical model comparison
- Animated visualizations
- Professional ML workflow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, auc
)
from sklearn.preprocessing import StandardScaler

# Progress & Animation
from tqdm import tqdm
import time
import joblib
import os

# W&B Integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not installed. Run: pip install wandb")

# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Run: pip install xgboost")

class MLMetricsCalculator:
    """🔢 Comprehensive ML Metrics Calculator"""
    
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_all_metrics(self, y_true, y_pred, y_proba=None, model_name="Model"):
        """Calculate comprehensive metrics for a model"""
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Add probabilistic metrics if available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                
                # Precision-Recall AUC
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba[:, 1])
                metrics['pr_auc'] = auc(recall_curve, precision_curve)
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Store in history
        self.metrics_history[model_name] = metrics
        
        return metrics
    
    def get_metrics_comparison_df(self):
        """Get DataFrame comparing all models"""
        if not self.metrics_history:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, metrics in self.metrics_history.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}",
                'PR-AUC': f"{metrics.get('pr_auc', 0):.4f}"
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)

class WandBTracker:
    """📊 Weights & Biases Integration"""
    
    def __init__(self, project_name="titanic-ml-advanced"):
        self.project_name = project_name
        self.run = None
        self.enabled = WANDB_AVAILABLE
        
    def init_experiment(self, experiment_name, config=None):
        """Initialize W&B experiment"""
        if not self.enabled:
            print("⚠️  W&B not available, skipping tracking")
            return
            
        try:
            self.run = wandb.init(
                project=self.project_name,
                name=experiment_name,
                config=config or {}
            )
            print(f"🚀 W&B experiment started: {experiment_name}")
        except Exception as e:
            print(f"⚠️  W&B init failed: {e}")
            self.enabled = False
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to W&B"""
        if not self.enabled or not self.run:
            return
            
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"⚠️  W&B logging failed: {e}")
    
    def finish(self):
        """Finish W&B run"""
        if self.enabled and self.run:
            wandb.finish()

class AdvancedMLTrainer:
    """🎯 Advanced ML Training System"""
    
    def __init__(self, use_wandb=True):
        self.metrics_calc = MLMetricsCalculator()
        self.wandb_tracker = WandBTracker() if use_wandb else None
        
        # Model configurations
        self.models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'needs_scaling': False,
                'description': '🌳 Ensemble of decision trees - robust and interpretable'
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'needs_scaling': True,
                'description': '📈 Linear classifier - fast and interpretable'
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'needs_scaling': True,
                'description': '🎯 Support Vector Machine - powerful for complex boundaries'
            },
            'NeuralNetwork': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
                'needs_scaling': True,
                'description': '🧠 Multi-layer perceptron - learns complex patterns'
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models_config['XGBoost'] = {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                'needs_scaling': False,
                'description': '⚡ Gradient boosting - often wins competitions'
            }
        
        self.scaler = StandardScaler()
        self.results = {}
        self.training_history = []
    
    def prepare_data(self, X, y):
        """Prepare data with feature engineering"""
        print("🔧 Preparing data with advanced feature engineering...")
        
        # Create a copy to avoid modifying original
        X_processed = X.copy()
        
        # Advanced feature engineering
        if 'Age' in X_processed.columns and 'Fare' in X_processed.columns:
            # Age bands
            X_processed['Age_Band'] = pd.cut(X_processed['Age'], 
                                           bins=[0, 12, 18, 35, 60, 100], 
                                           labels=[0, 1, 2, 3, 4])
            X_processed['Age_Band'] = X_processed['Age_Band'].astype(int)
            
            # Fare bands
            X_processed['Fare_Band'] = pd.qcut(X_processed['Fare'], 
                                             q=4, labels=[0, 1, 2, 3])
            X_processed['Fare_Band'] = X_processed['Fare_Band'].astype(int)
        
        return X_processed, y
    
    def train_single_model(self, model_name, model_config, X_train, y_train, X_val, y_val):
        """Train a single model with comprehensive evaluation"""
        
        print(f"\n🚀 Training {model_name}...")
        print(f"   {model_config['description']}")
        
        model = model_config['model']
        
        # Scale data if needed
        if model_config['needs_scaling']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            scaler = None
        
        # Start timing
        start_time = time.time()
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled) if hasattr(model, 'predict_proba') else None
        
        training_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calc.calculate_all_metrics(
            y_val, y_pred, y_proba, model_name
        )
        metrics['training_time'] = training_time
        
        # Cross-validation for robustness
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Log to W&B
        if self.wandb_tracker:
            self.wandb_tracker.log_metrics({
                f"{model_name}/accuracy": metrics['accuracy'],
                f"{model_name}/f1_score": metrics['f1_score'],
                f"{model_name}/precision": metrics['precision'],
                f"{model_name}/recall": metrics['recall'],
                f"{model_name}/roc_auc": metrics.get('roc_auc', 0),
                f"{model_name}/cv_mean": metrics['cv_mean'],
                f"{model_name}/cv_std": metrics['cv_std'],
                f"{model_name}/training_time": training_time
            })
        
        # Store results
        self.results[model_name] = {
            'model': model,
            'metrics': metrics,
            'scaler': scaler
        }
        
        # Store training history for animation
        self.training_history.append({
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'step': len(self.training_history)
        })
        
        # Print results
        print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"   ✅ F1-Score: {metrics['f1_score']:.4f}")
        print(f"   ✅ CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        print(f"   ⏱️  Time: {training_time:.2f}s")
        
        return model, metrics
    
    def train_all_models(self, X, y, test_size=0.2):
        """Train all models with comprehensive comparison"""
        
        print("🎯 Starting Advanced ML Training Pipeline")
        print("=" * 50)
        
        # Prepare data
        X_processed, y_processed = self.prepare_data(X, y)
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_processed, test_size=test_size, 
            random_state=42, stratify=y_processed
        )
        
        print(f"📊 Dataset split: {len(X_train)} train, {len(X_val)} validation")
        
        # Initialize W&B experiment
        if self.wandb_tracker:
            config = {
                'dataset_size': len(X),
                'n_features': X_processed.shape[1],
                'test_size': test_size,
                'models': list(self.models_config.keys())
            }
            self.wandb_tracker.init_experiment("titanic_advanced_training", config)
        
        # Train each model with progress bar
        for model_name, model_config in tqdm(self.models_config.items(), 
                                           desc="🚀 Training Models"):
            
            model, metrics = self.train_single_model(
                model_name, model_config, X_train, y_train, X_val, y_val
            )
        
        # Create ensemble model
        self.create_ensemble_model(X_train, y_train, X_val, y_val)
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        # Create visualizations
        self.create_training_animation()
        self.plot_comprehensive_analysis()
        
        # Finish W&B
        if self.wandb_tracker:
            self.wandb_tracker.finish()
        
        return self.results
    
    def create_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Create and evaluate ensemble model"""
        
        print("\n🎭 Creating Ensemble Model...")
        
        # Select best models for ensemble
        best_models = []
        for name, result in self.results.items():
            if result['metrics']['accuracy'] > 0.75:  # Only good models
                model = result['model']
                if result['scaler']:
                    # Create pipeline for scaled models
                    from sklearn.pipeline import Pipeline
                    pipeline = Pipeline([
                        ('scaler', result['scaler']),
                        ('model', model)
                    ])
                    best_models.append((name, pipeline))
                else:
                    best_models.append((name, model))
        
        if len(best_models) >= 2:
            # Create voting classifier
            ensemble = VotingClassifier(
                estimators=best_models,
                voting='soft'  # Use probabilities
            )
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            y_pred = ensemble.predict(X_val)
            y_proba = ensemble.predict_proba(X_val)
            
            metrics = self.metrics_calc.calculate_all_metrics(
                y_val, y_pred, y_proba, "Ensemble"
            )
            
            self.results['Ensemble'] = {
                'model': ensemble,
                'metrics': metrics,
                'scaler': None
            }
            
            print(f"   ✅ Ensemble Accuracy: {metrics['accuracy']:.4f}")
            print(f"   ✅ Ensemble F1-Score: {metrics['f1_score']:.4f}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive training report"""
        
        print("\n📊 COMPREHENSIVE MODEL COMPARISON REPORT")
        print("=" * 60)
        
        # Get comparison DataFrame
        comparison_df = self.metrics_calc.get_metrics_comparison_df()
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['metrics']['accuracy'])
        best_accuracy = self.results[best_model_name]['metrics']['accuracy']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   🎯 Accuracy: {best_accuracy:.4f}")
        print(f"   📈 F1-Score: {self.results[best_model_name]['metrics']['f1_score']:.4f}")
        
        # Performance insights
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        print(f"   • Models trained: {len(self.results)}")
        print(f"   • Best accuracy: {best_accuracy:.4f}")
        print(f"   • Average accuracy: {np.mean([r['metrics']['accuracy'] for r in self.results.values()]):.4f}")
        
        return comparison_df
    
    def create_training_animation(self):
        """Create animated training progress visualization"""
        
        if not self.training_history:
            return
        
        print("\n🎬 Creating Training Animation...")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        models = [h['model'] for h in self.training_history]
        accuracies = [h['accuracy'] for h in self.training_history]
        f1_scores = [h['f1_score'] for h in self.training_history]
        steps = range(len(models))
        
        # Accuracy plot
        ax1.bar(steps, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('🎯 Model Accuracy Progression', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(steps)
        ax1.set_xticklabels(models, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # F1-Score plot
        ax2.bar(steps, f1_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('📈 Model F1-Score Progression', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('F1-Score')
        ax2.set_xticks(steps)
        ax2.set_xticklabels(models, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('training_animation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Animation saved as 'training_animation.png'")
    
    def plot_comprehensive_analysis(self):
        """Create comprehensive analysis plots"""
        
        if not self.results:
            print("⚠️  No results to plot")
            return
        
        print("\n📊 Creating Comprehensive Analysis Plots...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy Comparison', 'F1-Score Comparison',
                          'ROC-AUC Comparison', 'Training Time Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['metrics']['accuracy'] for m in models]
        f1_scores = [self.results[m]['metrics']['f1_score'] for m in models]
        roc_aucs = [self.results[m]['metrics'].get('roc_auc', 0) for m in models]
        times = [self.results[m]['metrics'].get('training_time', 0) for m in models]
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='blue'),
            row=1, col=1
        )
        
        # F1-Score comparison
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name='F1-Score', marker_color='red'),
            row=1, col=2
        )
        
        # ROC-AUC comparison
        fig.add_trace(
            go.Bar(x=models, y=roc_aucs, name='ROC-AUC', marker_color='green'),
            row=2, col=1
        )
        
        # Training time comparison
        fig.add_trace(
            go.Bar(x=models, y=times, name='Time (s)', marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="🚀 Advanced ML Models Comprehensive Analysis",
            showlegend=False,
            height=800
        )
        
        fig.write_html("comprehensive_analysis.html")
        fig.show()
        
        print("   ✅ Analysis saved as 'comprehensive_analysis.html'")
        
        return fig

def explain_metrics():
    """📚 Explain all metrics used in the system"""
    
    explanations = {
        "Accuracy": {
            "definition": "Proportion of correct predictions out of total predictions",
            "formula": "(TP + TN) / (TP + TN + FP + FN)",
            "when_to_use": "Balanced datasets where all classes are equally important",
            "limitation": "Can be misleading with imbalanced datasets"
        },
        
        "Precision": {
            "definition": "Proportion of positive predictions that are actually correct",
            "formula": "TP / (TP + FP)",
            "when_to_use": "When false positives are costly (e.g., spam detection)",
            "limitation": "Doesn't account for false negatives"
        },
        
        "Recall (Sensitivity)": {
            "definition": "Proportion of actual positives that are correctly identified",
            "formula": "TP / (TP + FN)",
            "when_to_use": "When false negatives are costly (e.g., disease detection)",
            "limitation": "Doesn't account for false positives"
        },
        
        "F1-Score": {
            "definition": "Harmonic mean of precision and recall",
            "formula": "2 * (Precision * Recall) / (Precision + Recall)",
            "when_to_use": "When you need balance between precision and recall",
            "limitation": "Gives equal weight to precision and recall"
        },
        
        "ROC-AUC": {
            "definition": "Area under the Receiver Operating Characteristic curve",
            "formula": "Integral of TPR vs FPR curve",
            "when_to_use": "Binary classification, measures discriminative ability",
            "limitation": "Can be overly optimistic for imbalanced datasets"
        },
        
        "PR-AUC": {
            "definition": "Area under the Precision-Recall curve",
            "formula": "Integral of Precision vs Recall curve",
            "when_to_use": "Imbalanced datasets, focuses on positive class",
            "limitation": "Less intuitive than ROC-AUC"
        }
    }
    
    print("\n📚 COMPREHENSIVE METRICS EXPLANATION")
    print("=" * 50)
    
    for metric, info in explanations.items():
        print(f"\n🔍 {metric.upper()}")
        print(f"   Definition: {info['definition']}")
        print(f"   Formula: {info['formula']}")
        print(f"   When to use: {info['when_to_use']}")
        print(f"   Limitation: {info['limitation']}")
    
    print(f"\n💡 WHY MULTIPLE METRICS?")
    print(f"   • Each metric captures different aspects of model performance")
    print(f"   • No single metric tells the complete story")
    print(f"   • Different business contexts require different priorities")
    print(f"   • Comprehensive evaluation reduces blind spots")

def explain_wandb():
    """📊 Explain Weights & Biases integration"""
    
    print("\n📊 WEIGHTS & BIASES (W&B) INTEGRATION")
    print("=" * 50)
    
    print("🎯 WHAT IS W&B?")
    print("   • MLOps platform for experiment tracking")
    print("   • Visualize metrics, hyperparameters, and model artifacts")
    print("   • Compare experiments and collaborate with team")
    print("   • Industry standard for ML workflow management")
    
    print("\n🚀 FEATURES WE USE:")
    print("   • Real-time metrics logging")
    print("   • Model artifact storage")
    print("   • Hyperparameter tracking")
    print("   • Experiment comparison")
    print("   • Interactive dashboards")
    
    print("\n💼 WHY IT MATTERS:")
    print("   • Reproducibility: Track every experiment")
    print("   • Collaboration: Share results with team")
    print("   • Optimization: Compare different approaches")
    print("   • Production: Monitor model performance")
    print("   • Debugging: Understand what works and why")
    
    print("\n🔧 SETUP:")
    print("   1. pip install wandb")
    print("   2. wandb login (create free account)")
    print("   3. Run training script")
    print("   4. View results at wandb.ai")

if __name__ == "__main__":
    # Demo usage
    print("🚀 Advanced ML Trainer Demo")
    print("This would normally load your Titanic data and run training")
    print("Use this in your main training script!")
    
    # Explain metrics and W&B
    explain_metrics()
    explain_wandb() 