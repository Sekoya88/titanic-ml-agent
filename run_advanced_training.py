#!/usr/bin/env python3
"""
🚀 Run Advanced ML Training Demo
===============================

This script demonstrates the advanced ML training system with:
- Multiple model comparison
- Comprehensive metrics
- Animations and visualizations
- W&B integration
- Professional reporting
"""

import pandas as pd
import numpy as np
from advanced_trainer import AdvancedMLTrainer, explain_metrics, explain_wandb
import os

def load_titanic_data():
    """Load and prepare Titanic data"""
    
    print("📊 Loading Titanic Dataset...")
    
    # Load data
    try:
        train_df = pd.read_csv('data/train.csv')
        print(f"   ✅ Loaded {len(train_df)} training samples")
    except FileNotFoundError:
        print("   ❌ data/train.csv not found!")
        print("   💡 Please ensure you have the Titanic dataset in data/ folder")
        return None, None
    
    # Prepare features and target
    # Basic preprocessing
    df = train_df.copy()
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Extract title from name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Title encoding
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'].fillna(0, inplace=True)
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                'Embarked', 'FamilySize', 'IsAlone', 'Title']
    
    X = df[features]
    y = df['Survived']
    
    print(f"   ✅ Features: {list(X.columns)}")
    print(f"   ✅ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def main():
    """Main training pipeline"""
    
    print("🎯 ADVANCED ML TRAINING SYSTEM")
    print("=" * 50)
    
    # Load data
    X, y = load_titanic_data()
    if X is None:
        return
    
    # Initialize advanced trainer
    print("\n🔧 Initializing Advanced ML Trainer...")
    trainer = AdvancedMLTrainer(use_wandb=True)
    
    # Train all models
    print("\n🚀 Starting comprehensive model training...")
    results = trainer.train_all_models(X, y, test_size=0.2)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETE!")
    print("=" * 60)
    
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"   • Models trained: {len(results)}")
    print(f"   • Best model: {max(results.keys(), key=lambda x: results[x]['metrics']['accuracy'])}")
    print(f"   • Best accuracy: {max([r['metrics']['accuracy'] for r in results.values()]):.4f}")
    
    print(f"\n📁 FILES GENERATED:")
    print(f"   • training_animation.png - Training progress visualization")
    print(f"   • comprehensive_analysis.html - Interactive analysis plots")
    if trainer.wandb_tracker and trainer.wandb_tracker.enabled:
        print(f"   • W&B dashboard - Check wandb.ai for experiment tracking")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   • Review the generated visualizations")
    print(f"   • Check W&B dashboard for detailed metrics")
    print(f"   • Use best model for predictions")
    print(f"   • Experiment with hyperparameter tuning")
    
    return results

if __name__ == "__main__":
    # Explain what we're doing
    print("📚 UNDERSTANDING THE SYSTEM")
    print("=" * 30)
    
    # Explain metrics
    explain_metrics()
    
    # Explain W&B
    explain_wandb()
    
    print("\n" + "=" * 60)
    input("Press Enter to start advanced training...")
    
    # Run training
    results = main()
    
    print("\n🎉 Advanced ML training complete!")
    print("Check the generated files and W&B dashboard for detailed results.") 