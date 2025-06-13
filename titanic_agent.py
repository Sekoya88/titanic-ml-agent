"""
Créer un agent ML simple pour le challenge Titanic Kaggle.

Objectifs :
- Lire et prétraiter les données
- Entraîner un modèle de classification
- Générer un fichier de soumission Kaggle

Packages nécessaires : pandas, numpy, scikit-learn, joblib
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib

class TitanicModelAgent:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
    
    def load_data(self):
        """Load train and test datasets"""
        self.train_df = pd.read_csv("data/train.csv")
        self.test_df = pd.read_csv("data/test.csv")
        self.test_passenger_ids = self.test_df["PassengerId"]
        print(f"✅ Loaded {len(self.train_df)} training samples, {len(self.test_df)} test samples")
    
    def quick_eda(self):
        """Quick EDA as per roadmap"""
        print("\n📊 Quick EDA:")
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Missing values:\n{self.train_df.isnull().sum()}")
        print(f"\nSurvival rate: {self.train_df['Survived'].mean():.3f}")
        print(f"Sex distribution:\n{self.train_df['Sex'].value_counts()}")
        print(f"Pclass distribution:\n{self.train_df['Pclass'].value_counts()}")
        print(f"Age stats:\n{self.train_df['Age'].describe()}")
    
    def preprocess(self, df, is_training=True):
        """Preprocess data with feature engineering"""
        df = df.copy()
        
        # Handle missing values
        df = df.copy()
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Embarked"] = df["Embarked"].fillna("S")
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        
        # Feature engineering
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        
        # Optional: Extract title from Name
        df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.')
        df["Title"] = df["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df["Title"] = df["Title"].replace('Mlle', 'Miss')
        df["Title"] = df["Title"].replace('Ms', 'Miss')
        df["Title"] = df["Title"].replace('Mme', 'Mrs')
        
        # Encode categorical variables
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        
        if is_training:
            self.label_encoders["Embarked"] = LabelEncoder()
            self.label_encoders["Title"] = LabelEncoder()
            df["Embarked"] = self.label_encoders["Embarked"].fit_transform(df["Embarked"])
            df["Title"] = self.label_encoders["Title"].fit_transform(df["Title"])
        else:
            df["Embarked"] = self.label_encoders["Embarked"].transform(df["Embarked"])
            df["Title"] = self.label_encoders["Title"].transform(df["Title"])
        
        # Select features
        features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
        return df[features]
    
    def train_model(self):
        """Train the model with cross-validation"""
        X = self.preprocess(self.train_df, is_training=True)
        y = self.train_df["Survived"]
        
        # Cross-validation
        scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"\n🎯 Cross-validation accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Train final model
        self.model.fit(X, y)
        print("✅ Model trained successfully!")
        
        # Feature importance
        feature_names = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
        importances = self.model.feature_importances_
        for feature, importance in zip(feature_names, importances):
            print(f"  {feature}: {importance:.3f}")
    
    def predict_and_save(self):
        """Generate predictions and save submission file"""
        X_test = self.preprocess(self.test_df, is_training=False)
        predictions = self.model.predict(X_test)
        
        submission = pd.DataFrame({
            "PassengerId": self.test_passenger_ids,
            "Survived": predictions
        })
        
        submission.to_csv("submission.csv", index=False)
        print(f"✅ submission.csv created with {len(predictions)} predictions!")
        print(f"Predicted survival rate: {predictions.mean():.3f}")
    
    def save_model(self):
        """Save the trained model"""
        joblib.dump(self.model, "titanic_model.pkl")
        joblib.dump(self.label_encoders, "label_encoders.pkl")
        print("✅ Model saved!")

if __name__ == "__main__":
    agent = TitanicModelAgent()
    agent.load_data()
    agent.quick_eda()
    agent.train_model()
    agent.predict_and_save()
    agent.save_model() 