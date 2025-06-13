# 📊 Guide Complet des Métriques ML Avancées

## 🎯 Table des Matières
1. [Introduction aux Métriques](#introduction)
2. [Métriques de Base](#metriques-base)
3. [Métriques Avancées](#metriques-avancees)
4. [F1-Score et Moyenne Harmonique](#f1-score)
5. [Courbes ROC et AUC](#roc-auc)
6. [Validation Croisée](#validation-croisee)
7. [Weights & Biases (W&B)](#wandb)
8. [Animations et Visualisations](#animations)
9. [Comparaison de Modèles](#comparaison)
10. [Cas Pratiques](#cas-pratiques)

---

## 🎯 Introduction aux Métriques {#introduction}

### Pourquoi Plusieurs Métriques ?

Dans le machine learning, **aucune métrique unique ne raconte l'histoire complète** de la performance d'un modèle. Chaque métrique capture un aspect différent :

- **Accuracy** : Vue d'ensemble générale
- **Precision** : Qualité des prédictions positives
- **Recall** : Capacité à détecter les positifs
- **F1-Score** : Équilibre entre precision et recall
- **ROC-AUC** : Capacité discriminative globale

### 🔍 Exemple Concret : Détection de Fraude

Imaginons un système de détection de fraude bancaire :

```
Dataset : 10,000 transactions
- 9,900 légitimes (99%)
- 100 frauduleuses (1%)
```

**Modèle Naïf** : Prédit toujours "légitime"
- **Accuracy** : 99% (excellent !)
- **Recall** : 0% (catastrophique !)

➡️ **L'accuracy seule est trompeuse !**

---

## 📈 Métriques de Base {#metriques-base}

### 🎯 Accuracy (Exactitude)

**Définition** : Proportion de prédictions correctes

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Quand l'utiliser** :
- ✅ Datasets équilibrés
- ✅ Toutes les classes sont importantes
- ❌ Datasets déséquilibrés

**Exemple Titanic** :
```python
# 80% accuracy = 80% des passagers correctement classifiés
# Mais ne dit pas si on prédit bien les survivants vs non-survivants
```

### 📊 Precision (Précision)

**Définition** : Proportion de prédictions positives qui sont correctes

```
Precision = TP / (TP + FP)
```

**Question** : "Parmi tous ceux que j'ai prédits comme survivants, combien ont vraiment survécu ?"

**Quand l'utiliser** :
- ✅ Faux positifs coûteux (spam, diagnostic médical)
- ✅ Ressources limitées pour agir sur les positifs

**Exemple** :
```
Prédictions "Survivant" : 100 passagers
Vraiment survivants : 85 passagers
Precision = 85/100 = 85%
```

### 🎪 Recall (Rappel/Sensibilité)

**Définition** : Proportion de vrais positifs correctement identifiés

```
Recall = TP / (TP + FN)
```

**Question** : "Parmi tous les survivants réels, combien ai-je correctement identifiés ?"

**Quand l'utiliser** :
- ✅ Faux négatifs coûteux (maladie, sécurité)
- ✅ Important de "ne rien rater"

**Exemple** :
```
Survivants réels : 120 passagers
Correctement identifiés : 100 passagers
Recall = 100/120 = 83.3%
```

### 🔄 Trade-off Precision vs Recall

Il existe toujours un **compromis** entre precision et recall :

```
Seuil Bas (0.3) → Plus de prédictions positives
├── Recall ↑ (on rate moins de survivants)
└── Precision ↓ (plus de faux positifs)

Seuil Haut (0.7) → Moins de prédictions positives
├── Precision ↑ (prédictions plus sûres)
└── Recall ↓ (on rate plus de survivants)
```

---

## ⚖️ F1-Score et Moyenne Harmonique {#f1-score}

### 🎯 Qu'est-ce que le F1-Score ?

Le F1-Score est la **moyenne harmonique** de la precision et du recall :

```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### 🤔 Pourquoi Moyenne Harmonique ?

**Moyenne Arithmétique** vs **Moyenne Harmonique** :

```python
# Exemple : Precision=90%, Recall=10%

# Moyenne arithmétique
moyenne_arith = (90 + 10) / 2 = 50%  # Trompeuse !

# Moyenne harmonique (F1-Score)
f1_score = 2 * (90 * 10) / (90 + 10) = 18%  # Plus réaliste !
```

**Propriété clé** : La moyenne harmonique **pénalise les déséquilibres**

### 📊 Interprétation du F1-Score

| F1-Score | Interprétation | Action |
|----------|----------------|---------|
| 0.90+ | Excellent | Modèle prêt pour production |
| 0.80-0.89 | Très bon | Optimisations mineures |
| 0.70-0.79 | Bon | Améliorations possibles |
| 0.60-0.69 | Moyen | Révision nécessaire |
| <0.60 | Faible | Revoir complètement |

### 🎯 Exemple Pratique Titanic

```python
# Modèle A
Precision = 85%
Recall = 75%
F1-Score = 2 * (85 * 75) / (85 + 75) = 79.7%

# Modèle B  
Precision = 95%
Recall = 60%
F1-Score = 2 * (95 * 60) / (95 + 60) = 74.2%

# Modèle A est meilleur malgré une precision plus faible !
```

---

## 📈 Courbes ROC et AUC {#roc-auc}

### 🎯 Qu'est-ce que ROC ?

**ROC** (Receiver Operating Characteristic) = Courbe qui montre la performance à tous les seuils

**Axes** :
- **X** : Taux de Faux Positifs (FPR) = FP / (FP + TN)
- **Y** : Taux de Vrais Positifs (TPR) = TP / (TP + FN) = Recall

### 📊 AUC (Area Under Curve)

**AUC** = Aire sous la courbe ROC

| AUC | Interprétation | Qualité |
|-----|----------------|---------|
| 1.0 | Parfait | Impossible en pratique |
| 0.9-0.99 | Excellent | Très rare |
| 0.8-0.89 | Bon | Objectif réaliste |
| 0.7-0.79 | Moyen | Améliorable |
| 0.5-0.69 | Faible | Peu mieux qu'aléatoire |
| 0.5 | Aléatoire | Aucune capacité prédictive |

### 🔍 Interprétation Intuitive

**AUC = 0.85** signifie :
> "Si je prends un survivant au hasard et un non-survivant au hasard, il y a 85% de chances que mon modèle donne un score plus élevé au survivant"

### 🎯 ROC vs Precision-Recall

**Utiliser ROC-AUC quand** :
- ✅ Dataset équilibré
- ✅ Intérêt pour les deux classes
- ✅ Comparaison générale de modèles

**Utiliser PR-AUC quand** :
- ✅ Dataset déséquilibré
- ✅ Focus sur la classe positive
- ✅ Faux positifs très coûteux

---

## 🔄 Validation Croisée {#validation-croisee}

### 🎯 Pourquoi la Validation Croisée ?

**Problème** : Un seul split train/test peut être "chanceux" ou "malchanceux"

**Solution** : Tester sur plusieurs splits différents

### 📊 K-Fold Cross-Validation

```python
# 5-Fold CV
Dataset divisé en 5 parties égales

Fold 1: Train[2,3,4,5] → Test[1] → Score₁
Fold 2: Train[1,3,4,5] → Test[2] → Score₂  
Fold 3: Train[1,2,4,5] → Test[3] → Score₃
Fold 4: Train[1,2,3,5] → Test[4] → Score₄
Fold 5: Train[1,2,3,4] → Test[5] → Score₅

Score Final = Moyenne(Score₁...Score₅) ± Écart-type
```

### 🔍 Interprétation des Résultats

```python
# Exemple de résultats CV
Scores: [0.82, 0.79, 0.85, 0.81, 0.78]
Moyenne: 0.81
Écart-type: 0.027

# Interprétation
"Le modèle a une accuracy de 81% ± 2.7%"
```

**Écart-type faible** (< 5%) = Modèle **stable**
**Écart-type élevé** (> 10%) = Modèle **instable** (overfitting possible)

### 🎯 Stratified K-Fold

Pour datasets déséquilibrés, utiliser **Stratified K-Fold** :
- Maintient la proportion des classes dans chaque fold
- Plus représentatif pour le Titanic (62% morts, 38% survivants)

---

## 📊 Weights & Biases (W&B) {#wandb}

### 🎯 Qu'est-ce que W&B ?

**Weights & Biases** = Plateforme MLOps pour le tracking d'expériences

### 🚀 Fonctionnalités Clés

#### 1. **Experiment Tracking**
```python
import wandb

# Initialiser une expérience
wandb.init(project="titanic-ml", name="random-forest-v1")

# Logger des métriques
wandb.log({
    "accuracy": 0.81,
    "f1_score": 0.79,
    "precision": 0.85,
    "recall": 0.75
})

# Logger des hyperparamètres
wandb.config.update({
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.1
})
```

#### 2. **Model Artifacts**
```python
# Sauvegarder le modèle
artifact = wandb.Artifact("titanic-model", type="model")
artifact.add_file("model.joblib")
wandb.log_artifact(artifact)
```

#### 3. **Visualisations Automatiques**
- Courbes de métriques en temps réel
- Comparaison d'expériences
- Distribution des hyperparamètres
- Matrices de confusion

### 💼 Pourquoi Utiliser W&B ?

#### **Reproductibilité**
```python
# Chaque expérience est tracée
Expérience #1: RandomForest(n_estimators=100) → Accuracy=0.81
Expérience #2: RandomForest(n_estimators=200) → Accuracy=0.83
Expérience #3: XGBoost(learning_rate=0.1) → Accuracy=0.85
```

#### **Collaboration**
- Partage des résultats avec l'équipe
- Commentaires sur les expériences
- Comparaison de différentes approches

#### **Optimisation**
- Sweep automatique d'hyperparamètres
- Détection des meilleures configurations
- Analyse de sensibilité

#### **Production**
- Monitoring des modèles en production
- Détection de drift
- A/B testing

### 🔧 Setup Rapide

```bash
# Installation
pip install wandb

# Login (compte gratuit)
wandb login

# Dans votre code
import wandb
wandb.init(project="mon-projet")
```

---

## 🎬 Animations et Visualisations {#animations}

### 🎯 Pourquoi des Animations ?

Les animations permettent de :
- **Visualiser l'évolution** de l'entraînement
- **Détecter les problèmes** (overfitting, convergence)
- **Comparer les modèles** visuellement
- **Communiquer les résultats** efficacement

### 📊 Types d'Animations

#### 1. **Training Progress**
```python
# Animation de l'accuracy au fil des epochs
plt.plot(epochs, train_accuracy, label='Train')
plt.plot(epochs, val_accuracy, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

#### 2. **Model Comparison**
```python
# Barres animées comparant les modèles
models = ['RandomForest', 'SVM', 'XGBoost', 'Neural Network']
accuracies = [0.81, 0.79, 0.85, 0.83]

plt.bar(models, accuracies, color=['blue', 'red', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()
```

#### 3. **Feature Importance Evolution**
```python
# Évolution de l'importance des features
for epoch in range(n_epochs):
    importance = model.feature_importances_
    plt.bar(feature_names, importance)
    plt.title(f'Feature Importance - Epoch {epoch}')
    plt.pause(0.1)
```

### 🔍 Détection de Problèmes

#### **Overfitting**
```
Train Accuracy: 95% ↗️
Val Accuracy: 75% ↘️
→ Modèle mémorise au lieu d'apprendre
```

#### **Underfitting**
```
Train Accuracy: 65% →
Val Accuracy: 64% →
→ Modèle trop simple
```

#### **Convergence**
```
Accuracy: 81% → 81% → 81%
→ Modèle a convergé, arrêter l'entraînement
```

---

## 🏆 Comparaison de Modèles {#comparaison}

### 🎯 Méthodologie de Comparaison

#### 1. **Métriques Multiples**
```python
comparison_df = pd.DataFrame({
    'Model': ['RandomForest', 'SVM', 'XGBoost', 'Neural Network'],
    'Accuracy': [0.81, 0.79, 0.85, 0.83],
    'Precision': [0.85, 0.82, 0.87, 0.84],
    'Recall': [0.75, 0.74, 0.81, 0.80],
    'F1-Score': [0.79, 0.78, 0.84, 0.82],
    'ROC-AUC': [0.88, 0.85, 0.91, 0.89],
    'Training_Time': [2.3, 15.7, 8.1, 45.2]
})
```

#### 2. **Tests Statistiques**
```python
from scipy.stats import ttest_rel

# Test de significativité entre deux modèles
scores_model_a = [0.81, 0.79, 0.83, 0.80, 0.82]  # CV scores
scores_model_b = [0.85, 0.84, 0.87, 0.83, 0.86]  # CV scores

t_stat, p_value = ttest_rel(scores_model_a, scores_model_b)

if p_value < 0.05:
    print("Différence statistiquement significative")
else:
    print("Pas de différence significative")
```

#### 3. **Analyse Multi-Critères**

| Critère | Poids | RandomForest | XGBoost | Neural Network |
|---------|-------|--------------|---------|----------------|
| Accuracy | 30% | 0.81 | 0.85 | 0.83 |
| Speed | 25% | 0.9 | 0.7 | 0.3 |
| Interpretability | 20% | 0.8 | 0.6 | 0.2 |
| Robustness | 15% | 0.9 | 0.8 | 0.6 |
| Memory | 10% | 0.7 | 0.8 | 0.5 |

**Score Final** = Σ(Critère × Poids)

### 🔍 Choix du Meilleur Modèle

#### **Contexte Business**
```python
# Détection de fraude → Privilégier Recall
if use_case == "fraud_detection":
    best_metric = "recall"
    
# Recommandation produit → Privilégier Precision  
elif use_case == "recommendation":
    best_metric = "precision"
    
# Classification générale → Privilégier F1-Score
else:
    best_metric = "f1_score"
```

#### **Contraintes Techniques**
```python
# Production temps réel → Privilégier vitesse
if deployment == "real_time":
    weight_speed = 0.4
    
# Analyse batch → Privilégier accuracy
elif deployment == "batch":
    weight_accuracy = 0.4
    
# Edge computing → Privilégier taille mémoire
elif deployment == "edge":
    weight_memory = 0.4
```

---

## 🎯 Cas Pratiques {#cas-pratiques}

### 🚢 Cas 1 : Titanic - Prédiction de Survie

**Contexte** : Prédire la survie des passagers

**Métriques Prioritaires** :
1. **F1-Score** (équilibre precision/recall)
2. **Recall** (ne pas rater de survivants)
3. **Accuracy** (performance générale)

**Résultats Typiques** :
```python
RandomForest:
├── Accuracy: 81.2%
├── Precision: 84.5%
├── Recall: 76.8%
├── F1-Score: 80.5%
└── ROC-AUC: 87.3%

Interprétation:
✅ Bon équilibre precision/recall
✅ Capacité discriminative élevée
⚠️  Peut améliorer le recall (survivants ratés)
```

### 🏥 Cas 2 : Diagnostic Médical

**Contexte** : Détecter une maladie grave

**Métriques Prioritaires** :
1. **Recall** (ne rater aucun malade)
2. **Sensitivity** (détecter tous les positifs)
3. **NPV** (Negative Predictive Value)

**Seuil Optimisé** :
```python
# Seuil bas pour maximiser le recall
threshold = 0.3  # Au lieu de 0.5 par défaut

# Résultat
Recall: 95% ✅ (on rate que 5% des malades)
Precision: 60% ⚠️ (40% de faux positifs)

# Acceptable car faux négatif = danger de mort
```

### 💳 Cas 3 : Détection de Fraude

**Contexte** : Détecter les transactions frauduleuses

**Métriques Prioritaires** :
1. **Precision** (éviter de bloquer clients légitimes)
2. **PR-AUC** (dataset très déséquilibré)
3. **F1-Score** (équilibre global)

**Défis Spécifiques** :
```python
# Dataset déséquilibré
Fraudes: 0.1% (100 sur 100,000)
Légitimes: 99.9% (99,900 sur 100,000)

# Accuracy trompeuse
Model_naive = "Toujours légitime"
Accuracy = 99.9% ❌ (mais Recall = 0%)

# Métriques appropriées
PR-AUC = 0.75 ✅
F1-Score = 0.68 ✅
```

### 🛒 Cas 4 : Recommandation E-commerce

**Contexte** : Recommander des produits

**Métriques Prioritaires** :
1. **Precision@K** (qualité du top-K)
2. **NDCG** (ordre des recommandations)
3. **Diversity** (variété des recommandations)

**Évaluation Spécifique** :
```python
# Top-5 recommandations
recommendations = [prod_A, prod_B, prod_C, prod_D, prod_E]
user_bought = [prod_A, prod_C]

Precision@5 = 2/5 = 40%
Recall@5 = 2/10 = 20% (sur 10 produits pertinents)

# Métrique business
Revenue_lift = +15% ✅
Click_through_rate = +8% ✅
```

---

## 🎯 Conclusion et Bonnes Pratiques

### ✅ Checklist Métriques ML

#### **Avant l'Entraînement**
- [ ] Analyser le déséquilibre des classes
- [ ] Définir les métriques prioritaires selon le contexte business
- [ ] Choisir la stratégie de validation (K-Fold, Stratified, etc.)
- [ ] Configurer le tracking d'expériences (W&B)

#### **Pendant l'Entraînement**
- [ ] Monitorer plusieurs métriques simultanément
- [ ] Visualiser l'évolution avec des animations
- [ ] Détecter l'overfitting (train vs validation)
- [ ] Logger les hyperparamètres et résultats

#### **Après l'Entraînement**
- [ ] Comparer statistiquement les modèles
- [ ] Analyser les matrices de confusion
- [ ] Tester sur des données non vues
- [ ] Valider avec des experts métier

### 🚀 Recommandations Avancées

#### **Pour Thomas (ou tout évaluateur)**
```python
# Questions à poser
1. "Pourquoi avoir choisi ces métriques ?"
2. "Comment gérez-vous le déséquilibre des classes ?"
3. "Quelle est la significativité statistique ?"
4. "Comment détectez-vous l'overfitting ?"
5. "Quel est l'impact business de ces résultats ?"
```

#### **Réponses Préparées**
```python
# Métriques choisies
"F1-Score pour l'équilibre, ROC-AUC pour la capacité discriminative,
 Cross-validation pour la robustesse"

# Déséquilibre
"Stratified K-Fold + métriques adaptées (PR-AUC) + 
 analyse par classe séparée"

# Overfitting  
"Écart-type CV < 5%, courbes train/val convergentes,
 validation sur données temporelles séparées"

# Impact business
"Amélioration de 15% du taux de survie prédit = 
 meilleure allocation des ressources de sauvetage"
```

### 🎯 Ressources pour Aller Plus Loin

#### **Livres**
- "Hands-On Machine Learning" - Aurélien Géron
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- "Pattern Recognition and Machine Learning" - Christopher Bishop

#### **Outils**
- **Weights & Biases** : wandb.ai
- **MLflow** : mlflow.org  
- **TensorBoard** : tensorflow.org/tensorboard
- **Optuna** : optuna.org (hyperparameter tuning)

#### **Datasets pour Pratiquer**
- **Kaggle** : kaggle.com/competitions
- **UCI ML Repository** : archive.ics.uci.edu/ml
- **OpenML** : openml.org

---

**🎉 Félicitations !** Tu es maintenant armé pour impressionner Thomas avec une compréhension approfondie des métriques ML avancées, des animations, et des meilleures pratiques MLOps ! 🚀 