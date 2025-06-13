# 🚢 Guide Complet - Titanic ML Agent

## 📖 Table des Matières
1. [Compréhension du Problème](#problème)
2. [Analyse des Données (EDA)](#eda)
3. [Preprocessing & Feature Engineering](#preprocessing)
4. [Modélisation ML](#modélisation)
5. [Évaluation & Métriques](#évaluation)
6. [SmollAgent Multi-Modèles](#smollagent)
7. [Interface Streamlit](#streamlit)

---

## 🎯 Compréhension du Problème {#problème}

### Le Challenge Titanic
- **Objectif**: Prédire qui survit au naufrage du Titanic
- **Type**: Classification binaire (0 = mort, 1 = survivant)
- **Dataset**: 891 passagers d'entraînement, 418 à prédire
- **Métrique**: Accuracy (pourcentage de bonnes prédictions)

### Pourquoi ce problème est intéressant ?
1. **Données réelles historiques**
2. **Features mixtes** (numériques + catégorielles)
3. **Valeurs manquantes** à gérer
4. **Insights sociologiques** (classe sociale, genre, âge)

---

## 📊 Analyse des Données (EDA) {#eda}

### Ce que ton EDA révèle:

```
Training data shape: (891, 12)
Survival rate: 0.384 (38.4% de survivants)
```

### Insights Clés:
- **Taux de survie global**: 38.4% (tragique mais réaliste)
- **Données manquantes**:
  - Age: 177/891 (19.8%) - critique à remplir
  - Cabin: 687/891 (77%) - trop de manquant, on peut ignorer
  - Embarked: 2/891 (0.2%) - facile à remplir

### Distribution par Sexe:
```
male: 577 (64.7%)
female: 314 (35.3%)
```
**Hypothèse**: "Women and children first" → Les femmes survivent plus.

### Distribution par Classe:
```
Classe 3: 491 (55.1%) - Classe populaire
Classe 1: 216 (24.2%) - Première classe  
Classe 2: 184 (20.7%) - Classe moyenne
```
**Hypothèse**: Classe sociale = accès aux canots de sauvetage.

---

## 🔧 Preprocessing & Feature Engineering {#preprocessing}

### 1. Gestion des Valeurs Manquantes

```python
# Age: Remplacer par la médiane (28 ans)
df["Age"].fillna(df["Age"].median(), inplace=True)

# Embarked: Remplacer par 'S' (Southampton, le plus fréquent)
df["Embarked"].fillna("S", inplace=True)

# Fare: Remplacer par la médiane
df["Fare"].fillna(df["Fare"].median(), inplace=True)
```

**Pourquoi ces choix ?**
- **Médiane vs Moyenne**: Résistante aux outliers (âges extrêmes)
- **Mode pour catégorielle**: Southampton était le port principal

### 2. Feature Engineering - Créer de Nouvelles Variables

#### A. FamilySize (Taille de la famille)
```python
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
```
**Logique**: SibSp (siblings/spouse) + Parch (parents/children) + soi-même

#### B. IsAlone (Voyager seul)
```python
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
```
**Logique**: Voyager seul = plus difficile à s'entraider

#### C. Title (Titre social)
```python
df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.')
# Mr, Mrs, Miss, Master, Dr, Rev, etc.
```
**Logique**: Le titre révèle âge, statut social, et priorité d'évacuation

### 3. Encodage des Variables Catégorielles

```python
# Sex: male=0, female=1 (simple mapping)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Embarked & Title: Label Encoding
LabelEncoder().fit_transform(df["Embarked"])
```

---

## 🧠 Modélisation ML {#modélisation}

### Pourquoi RandomForest ?

1. **Robuste aux outliers** et valeurs manquantes
2. **Gère automatiquement** les interactions entre features
3. **Peu de hyperparamètres** à tuner
4. **Rapide à entraîner** et performant "out-of-the-box"
5. **Feature importance** automatique

### Configuration du Modèle
```python
RandomForestClassifier(
    n_estimators=100,    # 100 arbres
    random_state=42      # Reproductibilité
)
```

### Features Sélectionnées
```
["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
```

---

## 📈 Évaluation & Métriques {#évaluation}

### Tes Résultats:
```
Cross-validation accuracy: 0.8047 (+/- 0.0637)
```

**Traduction**: 
- **80.47%** de précision moyenne
- **±6.37%** d'écart-type (stabilité correcte)

### Feature Importance:
```
Fare: 0.253      (25.3%) - Prix du billet = classe sociale
Age: 0.237       (23.7%) - Age = priorité d'évacuation  
Sex: 0.212       (21.2%) - Sexe = "Women first"
Title: 0.103     (10.3%) - Titre social
Pclass: 0.084    (8.4%)  - Classe du billet
FamilySize: 0.067 (6.7%) - Taille famille
Embarked: 0.032  (3.2%)  - Port d'embarquement
IsAlone: 0.012   (1.2%)  - Voyager seul
```

### Validation Croisée 5-Fold
```python
cross_val_score(model, X, y, cv=5, scoring='accuracy')
```
**Principe**: Divise les données en 5 parties, entraîne sur 4, teste sur 1, répète 5 fois.

---

## 🤖 SmollAgent Multi-Modèles {#smollagent}

### Concept du SmollAgent
Un agent intelligent qui teste automatiquement plusieurs modèles ML et choisit le meilleur.

### Modèles à Tester:
1. **RandomForest** (baseline)
2. **XGBoost** (gradient boosting)
3. **SVM** (Support Vector Machine)
4. **Logistic Regression** (simple et rapide)
5. **Neural Network** (MLPClassifier)

### Architecture SmollAgent:
```python
class SmollAgent:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(),
            'SVM': SVC(),
            'LogisticRegression': LogisticRegression(),
            'NeuralNetwork': MLPClassifier()
        }
        self.results = {}
    
    def compete_models(self, X, y):
        # Teste tous les modèles
        # Retourne le meilleur
```

---

## 🌐 Interface Streamlit {#streamlit}

### Fonctionnalités Prévues:
1. **Dashboard principal** avec métriques
2. **Visualisations EDA** interactives
3. **Comparaison des modèles** SmollAgent
4. **Prédiction en temps réel** sur nouveaux passagers
5. **Feature importance** interactive
6. **Matrice de confusion** et courbes ROC

### Structure de l'App:
```python
st.title("🚢 Titanic ML Agent Dashboard")

# Sidebar pour navigation
page = st.sidebar.selectbox("Navigation", [
    "📊 Dashboard",
    "🔍 EDA Exploratoire", 
    "🤖 SmollAgent Battle",
    "🎯 Prédictions Live",
    "📈 Métriques Avancées"
])
```

---

## 🔬 Techniques ML Avancées

### 1. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
```

### 2. Ensemble Methods
Combiner plusieurs modèles pour de meilleures performances:
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier()),
    ('svm', SVC(probability=True))
])
```

### 3. Feature Selection
Sélectionner automatiquement les meilleures features:
```python
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=5)
X_selected = selector.fit_transform(X, y)
```

---

## 🚀 Prochaines Étapes

1. **Créer l'app Streamlit** interactive
2. **Implémenter SmollAgent** avec battle des modèles
3. **Ajouter visualisations** avancées
4. **Optimiser hyperparamètres** automatiquement
5. **Déployer** sur Streamlit Cloud

---

## 💡 Points Clés à Retenir

### Ce que tu as appris:
1. **EDA** révèle les patterns cachés dans les données
2. **Feature Engineering** peut booster les performances
3. **Cross-validation** évite l'overfitting
4. **Feature importance** explique les décisions du modèle
5. **80.47%** est un excellent score pour Titanic

### Prochaines compétences:
- Streamlit pour créer des webapps ML
- Comparaison automatique de modèles
- Visualisation interactive des résultats
- Déploiement d'applications ML

**Tu es maintenant capable de comprendre et expliquer tout le pipeline ML du Titanic !** 🎓 