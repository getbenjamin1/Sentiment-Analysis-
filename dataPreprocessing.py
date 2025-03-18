import csv
import re
import nltk
from typing import List, Dict

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Dictionary of common slang terms
SLANG_DICT = {
    "u": "you", "r": "are", "lol": "laughing out loud", "idk": "i do not know", "btw": "by the way", "smh": "shaking my head", "tbh": "to be honest",
    "imho": "in my humble opinion", "omg": "oh my god", "w/": "with", "w/o": "without", "b/c": "because", "b4": "before",
    "brb": "be right back", "lmao": "laughing my ass off", "wtf": "what the fuck", "gonna": "going to", "wanna": "want to", "gotta": "have to", "cuz": "because",
    "dunno": "do not know", "kinda": "kind of", "sorta": "sort of", "tbf": "to be fair", "tho": "though", "nah": "no", "bruh": "brother or friend", "af": "as fuck",
    "gg": "good game", "sus": "suspicious or suspect", "lowkey": "a little bit or secretly", "highkey": "very much or openly", "noob": "newbie or inexperienced person",
    "cap": "lie or false", "bet": "okay or agreement", "yeet": "to throw or express excitement", "dope": "cool or awesome", "flex": "brag or show off",
    "vibe": "a feeling or atmosphere", "based": "confident in one's beliefs, not caring about others' opinions"
}

# Precompiles a regular expression pattern for all slang words
SLANG_REGEX = re.compile(
    r'\b(' + '|'.join(re.escape(key) for key in SLANG_DICT.keys()) + r')\b',
    flags=re.IGNORECASE
)

def replace_slang(text: str) -> str:
    """
    Replaces slang words in the given text with their proper meanings.
    Uses a precompiled regular expression for efficiency.
    """
    return SLANG_REGEX.sub(lambda match: SLANG_DICT[match.group(0).lower()], text)

def clean_text(text: str) -> str:
    """
    Performs text cleaning by: Converting to lowercase, Removing URLs, Removing non-alphabetic characters, Removing extra whitespace.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Removes URLs
    text = re.sub(r'[^a-z\s]', '', text)         # Removes punctuation, digits, etc.
    text = re.sub(r'\s+', ' ', text).strip()      # Removes extra whitespace
    return text

def get_wordnet_pos(tag: str) -> str:
    """
    Convert POS tag to a format compatible with WordNet lemmatisation.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if unknown

def tokenize_and_lemmatize(text: str) -> List[str]:
    """
    Tokenises the text, removes stopwords, and lemmatises tokens using POS tagging.
    """
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Get POS tags for each token
    tagged_tokens = pos_tag(tokens)
    
    # Lemmatising each token with the proper POS tag
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
    ]
    return lemmatized_tokens

def preprocess_csv_in_memory(input_csv: str, title_column: str, body_column: str, source_name: str) -> List[Dict[str, str]]:
    """
    Reads a CSV, merges the title and body (if available),
    applies slang mapping, cleans the text, tokenises, and lemmatises.
    Returns a list of dictionaries containing the processed data.
    """
    results = []
    
    with open(input_csv, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.DictReader(infile)
        
        for row in reader:
            # Safely retrieve title and body text
            title_text = row.get(title_column, "") if title_column else ""
            body_text = row.get(body_column, "") if body_column else ""
            
            # Merge title and body text
            merged_text = (title_text + " " + body_text).strip()
            
            # Replace slang in the merged text (this ensures that 'u' becomes 'you', etc.)
            processed_text = replace_slang(merged_text)
            
            # Clean the processed text
            cleaned = clean_text(processed_text)
            
            # Tokenise and lemmatise the cleaned text
            tokens_list = tokenize_and_lemmatize(cleaned)
            
            processed_row = {
                "source": source_name,
                # The combined_text now reflects the slang-mapped version
                "combined_text": processed_text,
                "cleaned_text": cleaned,
                "tokens": " ".join(tokens_list),
                "lemmatized_tokens": " ".join(tokens_list)  
            }
            
            results.append(processed_row)
    return results

def main():
    """
    Processes multiple CSV sources, deduplicates based on the processed combined_text,
    and writes the final processed data to an output CSV.
    """
    # Process each dataset (ensure the CSV files exist and have the correct column names)
    reddit_rows = preprocess_csv_in_memory('reddit_data.csv', 'Post Title', 'Post Content', 'Reddit')
    twitter_rows = preprocess_csv_in_memory('twitter_data.csv', '', 'text', 'Twitter')
    guardian_rows = preprocess_csv_in_memory('guardian_news.csv', 'webTitle', 'trailText', 'Guardian')
    mediastack_rows = preprocess_csv_in_memory('mediastack_news.csv', 'title', 'description', 'Mediastack')

    # Combine all processed rows into a single list
    all_rows = reddit_rows + twitter_rows + guardian_rows + mediastack_rows

    # Deduplicate rows based on the processed combined_text field
    deduped = {row["combined_text"]: row for row in all_rows}
    unique_rows = list(deduped.values())
    
    # Write the unique, processed data to a new CSV file
    output_csv = 'processed_data.csv'
    fieldnames = ["source", "combined_text", "cleaned_text", "tokens", "lemmatized_tokens"]
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in unique_rows:
            writer.writerow(row)
    
    print(f"All data sources merged, slang replaced, cleaned, tokenised, lemmatised, and saved to {output_csv}")

if __name__ == '__main__':
    main()
