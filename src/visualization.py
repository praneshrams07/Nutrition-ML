import seaborn as sns
import matplotlib.pyplot as plt

def ranking_heatmap(results_df):
    rank_df = results_df.copy()

    for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
        rank_df[metric] = rank_df[metric].rank(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        rank_df.set_index("Model"),
        annot=True,
        fmt=".0f",
        cmap="YlGnBu_r",
        linewidths=0.5
    )
    plt.title("Model Ranking Heatmap (1 = Best, Darker = Better)")
    plt.show()

