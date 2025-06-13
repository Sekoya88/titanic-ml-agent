"""
SmollAgent - Multi-Model ML Agent
Teste automatiquement plusieurs modèles et choisit le meilleur
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using alternatives")

class SmollAgent:
    """
    Agent intelligent qui teste plusieurs modèles ML automatiquement
    et choisit le meilleur pour le problème Titanic
    """
    
    def __init__(self):
        self.models = self._initialize_models()
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def _initialize_models(self):
        """Initialise la collection de modèles à tester"""
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'description': 'Ensemble d\'arbres de décision',
                'pros': ['Robuste', 'Feature importance', 'Peu de tuning'],
                'cons': ['Peut overfitter', 'Moins bon sur données linéaires']
            },
            
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'description': 'Régression logistique classique',
                'pros': ['Rapide', 'Interprétable', 'Probabilités calibrées'],
                'cons': ['Assume linéarité', 'Sensible aux outliers']
            },
            
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'description': 'Support Vector Machine',
                'pros': ['Efficace haute dimension', 'Polyvalent avec kernels'],
                'cons': ['Lent sur gros datasets', 'Difficile à interpréter']
            },
            
            'NeuralNetwork': {
                'model': MLPClassifier(random_state=42, max_iter=500),
                'description': 'Réseau de neurones multicouches',
                'pros': ['Apprend patterns complexes', 'Non linéaire'],
                'cons': ['Boîte noire', 'Besoin de données', 'Hyperparamètres']
            }
        }
        
        # Ajouter XGBoost si disponible
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                'description': 'Gradient Boosting optimisé',
                'pros': ['Très performant', 'Gère valeurs manquantes', 'Feature importance'],
                'cons': ['Overfitting possible', 'Plus de hyperparamètres']
            }
        
        return models
    
    def load_data(self):
        """Charge et prépare les données Titanic"""
        print("📊 Chargement des données...")
        self.train_df = pd.read_csv("data/train.csv")
        self.test_df = pd.read_csv("data/test.csv")
        self.test_passenger_ids = self.test_df["PassengerId"]
        
        print(f"✅ Données chargées: {len(self.train_df)} train, {len(self.test_df)} test")
        
    def preprocess(self, df, is_training=True):
        """Preprocessing avancé avec feature engineering"""
        df = df.copy()
        
        # Gestion valeurs manquantes
        df["Age"].fillna(df["Age"].median(), inplace=True)
        df["Embarked"].fillna("S", inplace=True)
        df["Fare"].fillna(df["Fare"].median(), inplace=True)
        
        # Feature engineering
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        
        # Extraction titre
        df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.')
        df["Title"] = df["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df["Title"] = df["Title"].replace('Mlle', 'Miss')
        df["Title"] = df["Title"].replace('Ms', 'Miss')
        df["Title"] = df["Title"].replace('Mme', 'Mrs')
        
        # Features avancées
        df["Age_Band"] = pd.cut(df["Age"], 5, labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        df["Fare_Band"] = pd.qcut(df["Fare"], 4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        # Encodage
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        
        categorical_features = ["Embarked", "Title", "Age_Band", "Fare_Band"]
        
        if is_training:
            for feature in categorical_features:
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature])
        else:
            for feature in categorical_features:
                df[feature] = self.label_encoders[feature].transform(df[feature])
        
        # Sélection features
        features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Title", "Age_Band", "Fare_Band"]
        return df[features]
    
    def battle_models(self, cv_folds=5):
        """Battle royale des modèles ML ! 🥊"""
        print("\n🥊 SMOLL AGENT MODEL BATTLE ROYALE!")
        print("=" * 60)
        
        # Préparation des données
        X = self.preprocess(self.train_df, is_training=True)
        y = self.train_df["Survived"]
        
        # Normalisation pour SVM et Neural Network
        X_scaled = self.scaler.fit_transform(X)
        
        self.results = {}
        
        for name, model_info in self.models.items():
            print(f"\n🔥 Testing {name}...")
            print(f"   📝 {model_info['description']}")
            
            start_time = time.time()
            
            # Utiliser données normalisées pour certains modèles
            if name in ['SVM', 'NeuralNetwork', 'LogisticRegression']:
                X_model = X_scaled
            else:
                X_model = X
            
            # Cross-validation
            scores = cross_val_score(
                model_info['model'], 
                X_model, 
                y, 
                cv=cv_folds, 
                scoring='accuracy'
            )
            
            train_time = time.time() - start_time
            
            # Stockage des résultats
            self.results[name] = {
                'model': model_info['model'],
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores,
                'train_time': train_time,
                'description': model_info['description'],
                'pros': model_info['pros'],
                'cons': model_info['cons']
            }
            
            print(f"   🎯 Score: {scores.mean():.4f} (±{scores.std():.4f})")
            print(f"   ⏱️ Time: {train_time:.2f}s")
            
            # Mise à jour du meilleur modèle
            if scores.mean() > self.best_score:
                self.best_score = scores.mean()
                self.best_model = name
        
        self._display_battle_results()
        return self.results
    
    def _display_battle_results(self):
        """Affiche les résultats du battle avec style"""
        print("\n" + "="*60)
        print("🏆 BATTLE RESULTS - LEADERBOARD")
        print("="*60)
        
        # Trier par score
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['mean_score'], 
            reverse=True
        )
        
        for i, (name, result) in enumerate(sorted_results):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
            print(f"{medal} {name:15} | Score: {result['mean_score']:.4f} | Time: {result['train_time']:.2f}s")
        
        print(f"\n🎊 WINNER: {self.best_model} avec {self.best_score:.4f} de précision!")
    
    def train_best_model(self):
        """Entraîne le meilleur modèle sur toutes les données"""
        if not self.best_model:
            raise ValueError("Aucun modèle n'a été testé. Lance battle_models() d'abord!")
        
        print(f"\n🚀 Entraînement du champion: {self.best_model}")
        
        X = self.preprocess(self.train_df, is_training=True)
        y = self.train_df["Survived"]
        
        # Normalisation si nécessaire
        if self.best_model in ['SVM', 'NeuralNetwork', 'LogisticRegression']:
            X = self.scaler.fit_transform(X)
        
        # Entraînement
        best_model_obj = self.results[self.best_model]['model']
        best_model_obj.fit(X, y)
        
        print("✅ Champion entraîné et prêt pour les prédictions!")
        
        return best_model_obj
    
    def predict_and_save(self, filename="smoll_agent_submission.csv"):
        """Génère les prédictions avec le meilleur modèle"""
        if not self.best_model:
            raise ValueError("Entraîne un modèle d'abord!")
        
        X_test = self.preprocess(self.test_df, is_training=False)
        
        # Normalisation si nécessaire
        if self.best_model in ['SVM', 'NeuralNetwork', 'LogisticRegression']:
            X_test = self.scaler.transform(X_test)
        
        best_model_obj = self.results[self.best_model]['model']
        predictions = best_model_obj.predict(X_test)
        
        # Sauvegarde
        submission = pd.DataFrame({
            "PassengerId": self.test_passenger_ids,
            "Survived": predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"✅ {filename} créé avec {len(predictions)} prédictions!")
        print(f"📊 Taux de survie prédit: {predictions.mean():.3f}")
        
        return submission
    
    def get_model_comparison(self):
        """Retourne un DataFrame comparatif des modèles"""
        if not self.results:
            return None
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['mean_score'],
                'Std': result['std_score'],
                'Train_Time': result['train_time'],
                'Description': result['description']
            })
        
        return pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)

def main():
    """Fonction principale pour tester SmollAgent"""
    print("🤖 SMOLL AGENT INITIALIZED")
    print("🎯 Mission: Dominer le challenge Titanic!")
    
    agent = SmollAgent()
    agent.load_data()
    
    # Battle des modèles
    results = agent.battle_models()
    
    # Entraînement du champion
    champion = agent.train_best_model()
    
    # Prédictions
    submission = agent.predict_and_save()
    
    print("\n🎉 Mission accomplie! SmollAgent a dominé le Titanic!")

if __name__ == "__main__":
    main() 