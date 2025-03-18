#!/usr/bin/env python3
"""
Topic Modelling Script using LDA.
Assumes the CSV contains a 'lemmatized_tokens' column.
"""

import pandas as pd
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
import re

def custom_filter(tokens):
    # Merge NLTK stopwords with custom generic words; retain important domain tokens.
    default_stopwords = set(stopwords.words('english'))
    custom_stopwords = {"say", "create", "great", "something", "dont", "think", "leave",
                        "didnt", "keep", "start", "still", "give", "story", "game", "review", "play", "find",
                        "design", "job", "come", "feel", "let", "try", "u", "get", "see", "take", "go",
                        "like", "im", "make", "one", "would", "also", "amp", "use", "tell", "call", "even",
                        "time", "know", "could", "look", "back", "thing", "way"}
    all_stopwords = default_stopwords.union(custom_stopwords)
    whitelist = {"ai", "ml", "agi", "cv", "nlp", "us", "uk", "eu"}
    # Keep token if it is in the whitelist or if it is not a stopword and longer than 1 character.
    return [token for token in tokens if (token in whitelist or (token not in all_stopwords and len(token) > 1))]

def main():
    # Load data and tokenize using the 'lemmatized_tokens' column.
    df = pd.read_csv('processed_data.csv')
    texts = df['lemmatized_tokens'].dropna().tolist()
    tokenized_texts = [custom_filter(text.split()) for text in texts]
    
    # Generate bigrams and trigrams to capture two and three word phrases.
    bigram_mod = Phraser(Phrases(tokenized_texts, min_count=5, threshold=100))
    tokenized_texts = [bigram_mod[doc] for doc in tokenized_texts]
    trigram_mod = Phraser(Phrases(tokenized_texts, min_count=5, threshold=100))
    tokenized_texts = [trigram_mod[doc] for doc in tokenized_texts]
    
    # Build a dictionary mapping from words to unique IDs.
    dictionary = corpora.Dictionary(tokenized_texts)
    # Tighter filtering: remove tokens that appear in fewer than 30 documents or in more than 30% of documents.
    dictionary.filter_extremes(no_below=30, no_above=0.3)
    # Create the corpus: a list of bag-of-words representations for each document.
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # Train the LDA model.
    num_topics = 8 # Number of topics to be extracted
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics,
                                random_state=42, passes=20, iterations=400)
    
    # Compute overall and per-topic coherence using the 'c_v' measure.
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_texts,
                                          dictionary=dictionary, coherence='c_v')
    overall_coherence = coherence_model_lda.get_coherence()
    print("Overall Coherence Score:", overall_coherence)
    # Compute and print coherence for each topic.
    coherence_per_topic = coherence_model_lda.get_coherence_per_topic()
    for idx, score in enumerate(coherence_per_topic):
        print(f"Topic #{idx} Coherence: {score}")
    # Identify the topic with the highest coherence score.
    top_topic_idx = coherence_per_topic.index(max(coherence_per_topic))
    print("\nTop Topic by Coherence:")
    print(f"Topic #{top_topic_idx} with Coherence Score: {coherence_per_topic[top_topic_idx]}")
    
    # Extract and print top words for each topic; save results.
    topics = []
    for topic_id in range(num_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=10)
        words_str = ", ".join([word for word, prob in topic_terms])
        print(f"\nTopic #{topic_id}:")
        print(words_str)
        print("-" * 40)
        topics.append({"Topic": f"Topic #{topic_id}",
                       "Top Words": words_str,
                       "Coherence": coherence_per_topic[topic_id]})
    
    topics_df = pd.DataFrame(topics)
    topics_df.to_csv("lda_topics.csv", index=False)
    
    print("\nTopic modelling complete. Topics saved to 'lda_topics.csv'.")

if __name__ == '__main__':
    main()
