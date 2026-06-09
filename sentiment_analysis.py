import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_twitter_sentiment():
    csv_file = "twitter_training.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Place this script right next to your downloaded dataset!")
        return

    # Twitter training dataset from Kaggle doesn't have headers included by default
    column_names = ['Tweet_ID', 'Entity', 'Sentiment', 'Tweet_Content']
    
    print("Loading data... (This dataset contains ~74k entries, please wait a moment)")
    df = pd.read_csv(csv_file, names=column_names)
    print("--- Dataset Loaded ---")
    print(f"Dataset Shape: {df.shape}\n")

    # Drop missing rows in content or sentiment labels if any exist
    df.dropna(subset=['Sentiment', 'Entity'], inplace=True)

    # Clean string labels to avoid formatting discrepancies
    df['Sentiment'] = df['Sentiment'].str.strip()
    df['Entity'] = df['Entity'].str.strip()

    # Set up our chart aesthetics
    sns.set_theme(style="whitegrid")

    # -------------------------------------------------------------
    # VISUALIZATION 1: Global Sentiment Value Counts (Bar Chart)
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    sentiment_counts = df['Sentiment'].value_counts()
    
    sns.barplot(
        x=sentiment_counts.index, 
        y=sentiment_counts.values, 
        hue=sentiment_counts.index,
        palette='viridis', 
        legend=False
    )
    plt.title('Overall Distribution of Twitter Sentiments', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment Class', fontsize=12)
    plt.ylabel('Number of Tweets', fontsize=12)
    plt.tight_layout()
    plt.savefig('overall_sentiment_distribution.png', dpi=300)
    plt.show()

    # -------------------------------------------------------------
    # VISUALIZATION 2: Top Entities Sentiment Analysis (Grouped Bar Chart)
    # -------------------------------------------------------------
    # Isolate the top 5 most frequently mentioned brands/entities for a legible plot
    top_5_entities = df['Entity'].value_counts().head(5).index
    df_filtered = df[df['Entity'].isin(top_5_entities)]

    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df_filtered, 
        x='Entity', 
        hue='Sentiment', 
        palette='magma'
    )
    plt.title('Sentiment Profiles Across the Top 5 Mentioned Entities/Brands', fontsize=14, fontweight='bold')
    plt.xlabel('Brand / Entity Name', fontsize=12)
    plt.ylabel('Tweet Count', fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(title='Sentiment Class', loc='upper right')
    plt.tight_layout()
    plt.savefig('brand_sentiment_comparison.png', dpi=300)
    plt.show()

    print("Analysis finished successfully! Graphic files saved to your working folder.")

if __name__ == "__main__":
    analyze_twitter_sentiment()
