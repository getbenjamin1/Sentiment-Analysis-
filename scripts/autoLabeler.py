import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download("vader_lexicon")

# Load the dataset
file_path = "processed_data.csv"  # Make sure this is the correct path
df = pd.read_csv(file_path)

# Ensuring the necessary column exists(just for error sake)
if "cleaned_text" not in df.columns:
    raise ValueError("The CSV file must contain a 'cleaned_text' column.")

# Initialise Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to assign sentiment labels
def get_sentiment_label(text):
    """Classifies text as Negative (0), Neutral (1), or Positive (2)."""
    sentiment_score = analyzer.polarity_scores(str(text))["compound"]
    
    if sentiment_score >= 0.05:
        return 2  # Positive
    elif sentiment_score <= -0.05:
        return 0  # Negative
    else:
        return 1  # Neutral

# Apply sentiment labeling
df["label"] = df["cleaned_text"].apply(get_sentiment_label)

# Save the auto-labeled dataset
df.to_csv("processed_data_labeled.csv", index=False)

print("Auto-labeling complete! Labeled dataset saved as 'processed_data_labeled.csv'.")
