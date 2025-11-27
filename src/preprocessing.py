import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(path):
    df = pd.read_csv(path)

    # Fill missing values
    median = df.median(numeric_only=True)
    df.fillna(median, inplace=True)

    # Binary encoding
    df['Is_Vegan'] = df['Is_Vegan'].astype(int)
    df['Is_Gluten_Free'] = df['Is_Gluten_Free'].astype(int)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Meal_Type', 'Preparation_Method'], dtype=int)

    X = df.drop('Food_Name', axis=1)
    y = df['Food_Name']

    return X, y


def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_cols = [
        'Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber',
        'Sodium', 'Cholesterol', 'Glycemic_Index',
        'Water_Content', 'Serving_Size'
    ]

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Label Encode y
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test
