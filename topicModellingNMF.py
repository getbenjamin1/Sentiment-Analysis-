import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser  # Ensure these are imported
import re

def custom_filter(tokens):
    # Normalize tokens: lower-case and strip whitespace.
    tokens = [token.lower().strip() for token in tokens]
    # Define default stopwords from NLTK.
    default_stopwords = set(stopwords.words('english'))
    # Extend with additional generic words.
    custom_stopwords = {
    "say", "create", "great", "something", "dont", "think", "leave",
    "didnt", "keep", "start", "still", "give", "story", "game", "review",
    "play", "find", "design", "job", "come", "feel", "let", "try", "u",
    "get", "see", "take", "go", "like", "im", "make", "one", "would", "also",
    "amp", "use", "tell", "call", "even", "time", "know", "could", "look",
    "back", "thing", "way", "sporefuneve", "sporedotfun", "ruthless", "naive",
    "weak", "memes", "survive", "talk", "post", "link", "need", "year", 
    "content", "world", "new", "agent", "model", "mind", "first", "every", "day"
    }

    all_stopwords = default_stopwords.union(custom_stopwords)
    # Define a whitelist for important short tokens.
    whitelist = {"ai", "ml", "agi", "cv", "nlp", "us", "uk", "eu"}
    # Keep token if it's in the whitelist or if it is not in the stopwords and is longer than 1 character.
    return [token for token in tokens if (token in whitelist or (token not in all_stopwords and len(token) > 1))]

def main():
    # Load the processed data (assumed to contain a 'lemmatized_tokens' column)
    df = pd.read_csv('processed_data.csv')
    
    # Use 'lemmatized_tokens' if available; otherwise, use 'cleaned_text'
    if 'lemmatized_tokens' in df.columns:
        texts = df['lemmatized_tokens'].dropna().tolist()
        # Convert space-separated tokens into a list, then apply custom filtering.
        tokenized_texts = [custom_filter(text.split()) for text in texts]
    else:
        texts = df['cleaned_text'].dropna().tolist()
        
        def preprocess(text):
            # Remove numbers and non-alphabetic characters.
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Convert to lowercase and split into tokens.
            tokens = text.lower().split()
            # Apply custom filtering.
            return custom_filter(tokens)
        
        tokenized_texts = [preprocess(text) for text in texts]
     # Generate bigrams to capture common phrases.
    bigram = Phrases(tokenized_texts, min_count=50, threshold=150)
    bigram_mod = Phraser(bigram)
    tokenized_texts = [bigram_mod[doc] for doc in tokenized_texts]
    # Generate trigrams to capture longer phrases.
    trigram = Phrases(tokenized_texts, min_count=50, threshold=150)
    trigram_mod = Phraser(trigram)
    tokenized_texts = [trigram_mod[doc] for doc in tokenized_texts]
    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(tokenized_texts)
    # Tighter filtering: remove tokens that appear in fewer than 30 documents or in more than 30% of documents.
    dictionary.filter_extremes(no_below=30, no_above=0.3)
    # Create the corpus: a list of bag-of-words representations for each document.
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    # Initialise TfidfVectorizer with custom stopwords.
    default_stopwords = set(stopwords.words('english'))
    custom_stopwords = {
    "say", "create", "great", "something", "dont", "think", "leave","didnt", "keep", "start", "still",
    "give", "story", "game", "review","play", "find", "design", "job", "come", "feel", "let", "try", "u", "get",
    "see", "take", "go", "like", "im", "make", "one", "would", "also", "amp", "use", "tell", "call", "even", "time", "know", "could", "look", "back", "thing",
    "way", "sporefuneve", "sporedotfun", "ruthless", "naive", "weak", "memes", "survive", "talk", "post", "link", "need", "year","content", "world", "new", "agent",
    "model", "mind", "first", "every", "day"
    }

    all_stopwords = default_stopwords.union(custom_stopwords)
    vectorizer = TfidfVectorizer(stop_words=list(all_stopwords),
                                 max_df=0.3, min_df=50,
                                 max_features=50)
    tfidf = vectorizer.fit_transform(texts)
    # Train the NMF model.
    num_topics = 3
    nmf_model = NMF(n_components=num_topics, random_state=42)
    nmf_features = nmf_model.fit_transform(tfidf)
    feature_names = vectorizer.get_feature_names_out()
    # Extract the top words for each topic.
    topics = []
    topics_list = []  # For coherence calculation.
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_indices = topic.argsort()[::-1][:10]
        top_words = [feature_names[i] for i in top_indices]
        words_str = ", ".join(top_words)
        print(f"Topic #{topic_idx}:")
        print(words_str)
        print("-" * 40)
        topics.append({"Topic": f"Topic #{topic_idx}", "Top Words": words_str})
        topics_list.append(top_words)

    # Compute coherence score for NMF topics using the 'u_mass' measure.
    coherence_model = CoherenceModel(topics=topics_list, texts=tokenized_texts,
                                     dictionary=dictionary, coherence='u_mass')
    overall_coherence = coherence_model.get_coherence()
    print("Overall Coherence Score:", overall_coherence)
    
    per_topic_coherence = coherence_model.get_coherence_per_topic()
    for idx, score in enumerate(per_topic_coherence):
        print(f"Topic #{idx} Coherence: {score}")
    
    # Save the topics and coherence scores to a CSV file.
    for idx in range(len(topics)):
        topics[idx]["Coherence"] = per_topic_coherence[idx]
    topics_df = pd.DataFrame(topics)
    topics_df.to_csv("nmf_topics.csv", index=False)
    
    print("\nTopic modelling complete. Topics saved to 'nmf_topics.csv'.")

if __name__ == '__main__':
    main()

