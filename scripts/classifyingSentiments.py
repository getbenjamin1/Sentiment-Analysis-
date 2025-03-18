#!/usr/bin/env python3
"""
This script loads the fine-tuned BERT model and uses it to classify sentiments in the entire preprocessed dataset.
The predicted sentiment labels (0: Negative, 1: Neutral, 2: Positive) are added to the dataset and saved to a new CSV file.
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Set the path to your saved model directory
model_path = "/Users/dennislaflare/Library/CloudStorage/OneDrive-AtlanticTU/Year 4/Semester 2/Project Development/Datas/fine_tuned_bert"

# Load the fine-tuned model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set up the device (MPS for M1 Mac, or fallback to CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()  # Put model in evaluation mode

# Load the preprocessed dataset
df = pd.read_csv("processed_data.csv")

# Check if the 'cleaned_text' column exists
if "cleaned_text" not in df.columns:
    raise ValueError("The CSV file must contain a 'cleaned_text' column.")

# Function to classify sentiment for a given text
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return prediction

# Apply sentiment classification to the entire dataset
df["predicted_label"] = df["cleaned_text"].apply(classify_sentiment)

# Optionally, map numeric predictions to human-readable labels
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
df["predicted_sentiment"] = df["predicted_label"].map(label_mapping)

# Save the dataset with predicted sentiments to a new CSV file
output_file = "processed_data_with_predictions.csv"
df.to_csv(output_file, index=False)
print(f"Sentiment classification complete! The results have been saved to '{output_file}'.")
