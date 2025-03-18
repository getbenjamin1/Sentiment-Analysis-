import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

def predict_sentiment(text, model, tokenizer, device):
    """
    Tokenises the input text, passes it through the model, applies softmax to obtain probabilities,
    and returns the predicted sentiment label along with the probabilities.
    
    Args:
        text (str): The input text to classify.
        model (BertForSequenceClassification): The fine-tuned BERT model.
        tokenizer (BertTokenizer): The BERT tokenizer.
        device (torch.device): The device to run the model on.
        
    Returns:
        label (int): The predicted sentiment label (0: Negative, 1: Neutral, 2: Positive).
        probs (Tensor): The probabilities for each class.
    """
    # Tokenise input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Pass through the model
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
     logits = outputs.logits
    
    # Apply softmax to obtain probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Determine the sentiment label (0: Negative, 1: Neutral, 2: Positive)
    label = torch.argmax(probs, dim=-1).item()
    return label, probs

def apply_predictions_to_dataset(df, model, tokenizer, device):
    """
    Applies the predict_sentiment() function across the dataset and adds a new column 'predicted_label'
    with the corresponding sentiment for each text entry.
    
    Args:
        df (pandas.DataFrame): The dataframe containing a 'cleaned_text' column.
        model (BertForSequenceClassification): The fine-tuned BERT model.
        tokenizer (BertTokenizer): The BERT tokenizer.
        device (torch.device): The device to run the model on.
    
    Returns:
        df (pandas.DataFrame): The updated dataframe with an additional 'predicted_label' column.
    """
    predicted_labels = []
    for text in df['cleaned_text']:
        label, _ = predict_sentiment(text, model, tokenizer, device)
        predicted_labels.append(label)
    df['predicted_label'] = predicted_labels
    return df

if __name__ == '__main__':
    # Load the fine-tuned model and tokenizer from the saved directory
    model_path = "/Users/dennislaflare/Library/CloudStorage/OneDrive-AtlanticTU/Year 4/Semester 2/Project Development/Datas/fine_tuned_bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    
    # Set the device (using GPU if available, otherwise CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    # Load your dataset (assumed to have a 'cleaned_text' column)
    import pandas as pd
    df = pd.read_csv("processed_data.csv")
    
    # Apply the prediction function to label the dataset
    df_labeled = apply_predictions_to_dataset(df, model, tokenizer, device)
    
    # Save or inspect the predictions
    df_labeled.to_csv("processed_data_predicted.csv", index=False)
    print("Predictions applied and saved to 'processed_data_predicted.csv'.")
