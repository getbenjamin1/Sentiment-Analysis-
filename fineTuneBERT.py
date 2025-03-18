"""
This script fine-tunes BERT for sequence classification using the Hugging Face Trainer API.
It loads the labelled CSV data, tokenises the text, splits the data into training (70%),
validation (15%), and test (15%) sets, and then fine-tunes the model.
"""

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    """
    Computes evaluation metrics: accuracy, precision, recall and F1 score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def tokenize_function(examples, tokenizer):
    """
    Tokenises the 'cleaned_text' field for each example.
    """
    return tokenizer(examples['cleaned_text'], truncation=True, padding='max_length', max_length=128)

def main():
    # -----------------------------------------------------------------------------
    # Load Labelled CSV Data
    # -----------------------------------------------------------------------------
    # The labelled CSV (processed_data_labeled.csv) has the columns 'cleaned_text' and 'label'
    df = pd.read_csv('processed_data_labeled.csv')
    
    # Check for required columns.
    if 'cleaned_text' not in df.columns or 'label' not in df.columns:
        raise ValueError("The CSV file must contain 'cleaned_text' and 'label' columns.")
    
    # -----------------------------------------------------------------------------
    # Split Data into Training (70%), Validation (15%), and Test (15%) Sets
    # -----------------------------------------------------------------------------
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # Convert DataFrames to Hugging Face Datasets.
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # -----------------------------------------------------------------------------
    # Initialise BERT Tokeniser and Model
    # -----------------------------------------------------------------------------
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    
    # Tokenise the datasets.
    tokenize_fn = lambda examples: tokenize_function(examples, tokenizer)
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)
    test_dataset = test_dataset.map(tokenize_fn, batched=True)
    
    # Set the datasets to return PyTorch tensors.
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # -----------------------------------------------------------------------------
    # Prepare Data Collator for Dynamic Padding
    # -----------------------------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # -----------------------------------------------------------------------------
    # Set Up Training Arguments
    # -----------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy'
    )
    
    # -----------------------------------------------------------------------------
    # Initialise the Trainer
    # -----------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # -----------------------------------------------------------------------------
    # Train the Model
    # -----------------------------------------------------------------------------
    trainer.train()
    
    # Evaluate the model on the test set.
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test evaluation results:", test_results)
    
    # Save the fine-tuned model and tokeniser to the specified directory.
    save_dir = "/Users/dennislaflare/Library/CloudStorage/OneDrive-AtlanticTU/Year 4/Semester 2/Project Development/Datas/fine_tuned_bert"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"Training complete. The fine-tuned model is saved in '{save_dir}'.")

if __name__ == '__main__':
    main()
