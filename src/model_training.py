from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import os

def train_models(X_train, y_train):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")

    return models
