# NutriClass - Food Classification Using Nutrition Data

NutriClass is a machine learning project designed to classify foods based on their nutritional composition using traditional ML models.

## 📊 Features
- Data cleaning & preprocessing
- Binary encoding + one-hot encoding
- Standard scaling
- Model training (LR, SVM, RF, KNN, XGBoost, GB)
- Evaluation using accuracy, precision, recall, F1-score
- Heatmap-based ranking visualization
- Supports model saving (.pkl)

## 📁 Project Structure
```
NutriClass-ML/
│
├── fooddata.csv              # Dataset used for training and testing
│
├── main.py                   # Main entry point that runs the entire ML pipeline
│
├── requirements.txt          # Python dependencies for the project
│
├── models/                   # Auto-saved trained models (.pkl files)
│   ├── Logistic_Regression.pkl
│   ├── Decision_Tree.pkl
│   ├── Random_Forest.pkl
│   ├── KNN.pkl
│   ├── SVM.pkl
│   ├── XGBoost.pkl
│   └── Gradient_Boosting.pkl
│
├── src/                      # Source code package
│   ├── preprocessing.py      # Data loading, cleaning, encoding, scaling
│   ├── model_training.py     # Training all ML models
│   ├── evaluation.py         # Computes metrics (accuracy, precision, recall, F1)
│   └── visualization.py      # Heatmap and ranking visualizations
│
└── README.md                 # Project documentation (this file)
```


## 🚀 How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Place your dataset:
   /fooddata.csv

3. Run the project:
   python main.py

## 🏆 Best Models
Based on ranking:
- SVM (Best overall)
- XGBoost (Best recall)

