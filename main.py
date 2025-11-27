from src.preprocessing import load_data, prepare_data
from src.model_training import train_models
from src.evaluation import evaluate_all
from src.visualization import ranking_heatmap

def main():

    X, y = load_data("fooddata.csv")
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    models = train_models(X_train, y_train)

    results_df = evaluate_all(models, X_test, y_test)
    results_df.index = results_df.index + 1
    print(results_df)

    ranking_heatmap(results_df)

if __name__ == "__main__":
    main()
