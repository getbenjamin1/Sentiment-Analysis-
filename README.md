# Sentiment-Analysis
Sentiment Analysis on AGI in Public Discourse

The scripts have to be executed in a specific order to get the wanted outcome >
1. Collect all the data using RedditScrapper.py, MediaStack.py, GuardianScraper.py and also use apify to collect data from X
2. Pre-process the data using dataPreprocessing.py
3. Label data sentiment using autoLabeler.py
4. Fine-tune BERT using fineTuneBERT.py
5. Classify sentiment using classifyingSentiments.py
6. Apply the finetuned model with applyingBERT.py
7. Do topic modelling using both topicModellingLDA.py and topicModellingNMF.py
8. Get static visualisation using staticSentiment.py
9. Get interactive visualisation using interactiveSentiment.py
