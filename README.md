# 🚢 Titanic ML Agent - Advanced Machine Learning Pipeline

A comprehensive machine learning system for predicting Titanic passenger survival with professional-grade features including real-time training animations, comprehensive metrics analysis, and MLOps integration.

## 🎯 Key Features

- **🚀 Advanced Training Pipeline** - Multiple model comparison with real-time animations
- **📊 Comprehensive Metrics** - Accuracy, Precision, Recall, F1-Score, ROC-AUC analysis
- **🌐 Interactive Web Interface** - Professional Streamlit dashboard
- **📈 Real-time Visualizations** - Live training progress and epoch monitoring
- **🔧 MLOps Integration** - Weights & Biases experiment tracking
- **🤖 Automated Model Selection** - SmollAgent for intelligent model comparison

## 📦 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd titanic-ml-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Launch the Application
```bash
# Start the interactive web interface
streamlit run streamlit_app.py
```

## 🏗️ Architecture

### Core Components

```
titanic-ml-agent/
├── 📊 data/                    # Dataset storage
├── 🌐 streamlit_app.py         # Main web application
├── 🧠 advanced_trainer.py      # Advanced ML training system
├── 🤖 smoll_agent.py          # Automated model comparison
├── 📈 titanic_agent.py         # Core ML agent
├── 🚀 run_advanced_training.py # Advanced training demo
├── 📚 GUIDE_METRICS_AVANCES.md # Comprehensive ML guide
├── 🔧 setup.sh                # Setup script
└── 📋 requirements.txt        # Dependencies
```

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Missing value imputation
   - Feature engineering (FamilySize, IsAlone, Title extraction)
   - Categorical encoding
   - Feature scaling

2. **Model Training**
   - RandomForest, SVM, Neural Networks, XGBoost
   - Cross-validation with confidence intervals
   - Hyperparameter optimization
   - Real-time progress monitoring

3. **Evaluation & Analysis**
   - Comprehensive metrics calculation
   - Statistical model comparison
   - Confusion matrix analysis
   - Feature importance ranking

## 🚀 Usage Examples

### Interactive Web Interface
```bash
streamlit run streamlit_app.py
```
Navigate through:
- **📊 Dashboard** - Overview and key metrics
- **🔍 EDA** - Exploratory data analysis
- **🤖 SmollAgent** - Automated model battle
- **🎯 Predictions** - Live prediction interface
- **🚀 Advanced Training** - Professional ML pipeline

### Programmatic Usage
```python
from titanic_agent import TitanicModelAgent

# Initialize agent
agent = TitanicModelAgent()

# Load and explore data
agent.load_data()
agent.quick_eda()

# Train model
agent.train_model()

# Generate predictions
agent.predict_and_save()
```

### Advanced Training
```python
from advanced_trainer import AdvancedMLTrainer

# Initialize with W&B tracking
trainer = AdvancedMLTrainer(use_wandb=True)

# Train multiple models
results = trainer.train_all_models(X, y, test_size=0.2)

# Compare performance
comparison = trainer.compare_models(results)
```

## 📊 Model Performance

### Baseline Results
- **Cross-Validation Accuracy**: 80-82%
- **F1-Score**: 0.78-0.81
- **ROC-AUC**: 0.85-0.88

### Feature Importance
1. **Sex** - Primary survival predictor
2. **Fare** - Economic status indicator  
3. **Age** - Age-based survival patterns
4. **Title** - Social status from name
5. **Pclass** - Passenger class

## 🔧 Advanced Features

### Real-time Training Monitoring
- Live accuracy/loss curves
- Epoch-by-epoch progress tracking
- Training logs with timestamps
- Interactive Plotly visualizations

### MLOps Integration
- Weights & Biases experiment tracking
- Model versioning and artifacts
- Hyperparameter logging
- Team collaboration features

### Comprehensive Metrics Analysis
- Detailed explanation of ML metrics
- Business context examples
- Performance interpretation guides
- Best practices recommendations

## 📚 Educational Content

The system includes comprehensive educational materials:

- **ML Metrics Deep Dive** - Understanding accuracy, precision, recall, F1-score
- **W&B MLOps Guide** - Complete Weights & Biases integration tutorial
- **Cross-Validation Explained** - Robust model evaluation techniques
- **Business Context Examples** - Real-world applications and interpretations

## 🛠️ Development

### Running Tests
```bash
python validate_setup.py
```

### Code Structure
- **Modular Design** - Separate concerns for data, models, and UI
- **Type Hints** - Full type annotation support
- **Documentation** - Comprehensive docstrings and comments
- **Error Handling** - Robust exception management

## 📈 Performance Optimization

### Training Efficiency
- Parallel model training
- Optimized data preprocessing
- Memory-efficient operations
- Progress tracking with minimal overhead

### Scalability
- Configurable model parameters
- Flexible data input formats
- Extensible architecture for new models
- Cloud deployment ready

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kaggle Titanic Competition for the dataset
- Scikit-learn for machine learning algorithms
- Streamlit for the web interface framework
- Weights & Biases for MLOps capabilities
- Plotly for interactive visualizations

---

**Ready for production ML workflows! 🚀** 