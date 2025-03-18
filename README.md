# Sentiment-Analysis
Sentiment Analysis on AGI in Public Discourse

**Execution Order**
The scripts have to be executed in a specific order to get the wanted outcome >

1. Data Collection
  Run the following scripts to collect raw data:
 **_RedditScraper.py MediaStack.py GuardianScraper.py_**
  use apify to collect data from X (formerly Twitter).

3. Data Preprocessing
  Run **_dataPreprocessing.py_** to clean, tokenise, and normalise the collected data.

5. Automatic Sentiment Labelling
  Execute **_autoLabeler.py_** to label data sentiment using VADER.

6. Fine-Tuning BERT
  Run **_fineTuneBERT.py_** to fine-tune a pre-trained BERT model on your labelled        dataset.

7. Sentiment Classification
  Execute **_classifyingSentiments.py_** to classify sentiments using your fine-tuned BERT model.

8. Applying the Fine-Tuned Model
  Run **_applyingBERT.py_** to predict sentiment on new text inputs using the fine-tuned model.

9. Topic Modelling
Execute both **_topicModellingLDA.py_** and **_topicModellingNMF.py_** to perform unsupervised topic modelling and extract key themes.

10. Static Visualisation
Run **_staticSentiment.py_** to generate static plots (e.g. bar charts, word clouds).

11. Interactive Visualisation
Execute **_interactiveSentiment.py_** to launch an interactive dashboard for exploring sentiment trends and topics.

**Installation**
1. Clone the Repository
```
git clone https://github.com/getbenjamin1/Sentiment-Analysis.git
cd Sentiment-Analysis
```

3. Create a Virtual Environment
```
python3 -m venv env
source env/bin/activate   # On Windows use: env\Scripts\activate
```

5. Install Dependencies
```
pip install -r requirements.txt
Note: Ensure you have Python 3.8 or above installed.
```


**Usage**

After following the installation steps and setting up your configuration, execute the scripts sequentially as outlined above. For example, to run data preprocessing:
```
python scripts/dataPreprocessing.py
```

Similarly, run each script in the given order to progress through the full analysis pipeline.
