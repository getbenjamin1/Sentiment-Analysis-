import re
import pandas as pd
import plotly.express as px
from transformers import pipeline, BertTokenizer
import datetime

# 1. Load Twitter data
df = pd.read_csv("twitter_data.csv")

# 2. Clean tweet text
def clean_tweet(text):
    """
    Clean tweet text by:
      - Removing URLs, mentions, hashtags, and punctuation
      - Converting text to lowercase and stripping extra whitespace
    """
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)      # Remove @mentions
    text = re.sub(r'#\w+', '', text)      # Remove hashtags (if desired)
    text = re.sub(r'[^\w\s]', '', text)    # Remove punctuation
    text = re.sub(r'\s+', ' ', text)       # Replace multiple spaces with a single space
    return text.strip().lower()

# Apply cleaning function
df['cleaned_text'] = df['text'].astype(str).apply(clean_tweet)

# 3. Apply fine-tuned BERT model for sentiment classification
# Define the model path
model_path = "/Users/dennislaflare/Library/CloudStorage/OneDrive-AtlanticTU/Year 4/Semester 2/Project Development/Datas/fine_tuned_bert"

# Create a sentiment-analysis pipeline with truncation enabled and a max_length of 512 tokens
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model=model_path, 
    tokenizer=BertTokenizer.from_pretrained(model_path),
    truncation=True,
    max_length=512
)

# Mapping from model labels to human-readable sentiment strings.
label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def predict_sentiment(text):
    # Pass truncation parameters explicitly in case it's needed for individual calls.
    result = sentiment_pipeline(text, truncation=True, max_length=512)[0]
    label = result['label']
    return label_mapping.get(label, label)

# Apply sentiment prediction
df['sentiment'] = df['cleaned_text'].apply(predict_sentiment)

# 4. Process the timestamp and create a date column
if 'created_at' in df.columns:
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
else:
    df['created_at'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df.index, unit='D')
df['date'] = df['created_at'].dt.date

# 5. Group the data by date and sentiment
sentiment_over_time = df.groupby(['date', 'sentiment']).size().reset_index(name='count')

# 6. Plot sentiment over time using Plotly Express
fig = px.line(
    sentiment_over_time,
    x='date',
    y='count',
    color='sentiment',
    markers=True,
    title="Sentiment Over Time"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Number of Tweets",
    legend_title="Sentiment"
)

fig.show()

# 7. (Optional) Save the DataFrame with predicted sentiments
df.to_csv("twitter_data_with_sentiment.csv", index=False)
