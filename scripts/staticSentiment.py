import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# -------------------------------------------
# Load the main labelled data
# -------------------------------------------
df_labeled = pd.read_csv("processed_data_labeled.csv")

# Check that required columns exist in the labelled data
if 'label' not in df_labeled.columns:
    raise ValueError("The CSV file must contain a 'label' column.")
if 'source' not in df_labeled.columns:
    raise ValueError("The CSV file must contain a 'source' column.")

# -------------------------------------------
# Load Twitter data
# -------------------------------------------
df_twitter = pd.read_csv('twitter_data_with_sentiment_updated.csv')

# Ensure that 'created_at' is parsed as datetime
df_twitter['created_at'] = pd.to_datetime(df_twitter['created_at'])

# Create a 'date' column by extracting the date part of 'created_at'
# Convert to datetime for better plotting with Seaborn
df_twitter['date'] = pd.to_datetime(df_twitter['created_at'].dt.date)

# Group the data by 'date' and 'sentiment', counting the number of tweets per group
sentiment_over_time = df_twitter.groupby(['date', 'sentiment']).size().reset_index(name='count')

# -------------------------------------------
# Twitter Sentiment Over Time 
# -------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=sentiment_over_time, 
    x='date', 
    y='count', 
    hue='sentiment', 
    marker='o'
)
plt.xlabel("Date")
plt.ylabel("Number of Tweets")
plt.title("Twitter Sentiment Over Time")
plt.tight_layout()
plt.savefig("twitter_sentiment_over_time.png")
plt.show()

# -------------------------------------------
# Overall Sentiment Distribution (Labelled Data)
# -------------------------------------------
sentiment_counts = df_labeled['label'].value_counts().sort_index()

plt.figure(figsize=(8, 6))
sns.barplot(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    hue=sentiment_counts.index,
    palette='viridis',
    dodge=False,
    legend=False
)
plt.xlabel("Sentiment Label (0: Negative, 1: Neutral, 2: Positive)")
plt.ylabel("Number of Entries")
plt.title("Overall Sentiment Distribution")
plt.xticks([0, 1, 2])
plt.tight_layout()
plt.savefig("overall_sentiment_distribution.png")
plt.show()

# -------------------------------------------
# Sentiment Distribution by Data Source (Labelled Data)
# -------------------------------------------
source_sentiment = df_labeled.groupby(['source', 'label']).size().reset_index(name='count')
source_sentiment_pivot = source_sentiment.pivot(index='source', columns='label', values='count').fillna(0)
source_sentiment_pivot = source_sentiment_pivot.sort_index()

plt.figure(figsize=(10, 8))
source_sentiment_pivot.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 8))
plt.xlabel("Data Source")
plt.ylabel("Number of Entries")
plt.title("Sentiment Distribution by Data Source")
plt.legend(title="Sentiment", labels=["Negative", "Neutral", "Positive"])
plt.tight_layout()
plt.savefig("sentiment_distribution_by_source.png")
plt.show()

# -------------------------------------------
# Word Clouds for Each Sentiment Category (Labelled Data)
# -------------------------------------------
# Assuming the data contains a 'cleaned_text' column with the text data.
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
for label, name in sentiment_labels.items():
    # Concatenate all text entries for a given sentiment
    text_data = " ".join(
        df_labeled[df_labeled['label'] == label]['cleaned_text']
        .dropna()
        .astype(str)
        .tolist()
    )
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for {name} Sentiment")
    plt.tight_layout()
    plt.savefig(f"wordcloud_{name.lower()}.png")
    plt.show()
