# Sentiment-Analysis
Sentiment Analysis on AGI in Public Discourse

**Execution Order**
The scripts have to be executed in a specific order to get the wanted outcome >

1. Data Collection
  Run the following scripts to collect raw data:
 **_ RedditScraper.py
  MediaStack.py
  GuardianScraper.py_**
  use apify to collect data from X (formerly Twitter).

2. Data Preprocessing
  Run **_dataPreprocessing.py_** to clean, tokenise, and normalise the collected data.

3. Automatic Sentiment Labelling
  Execute **_autoLabeler.py_** to label data sentiment using VADER.

4. Fine-Tuning BERT
  Run **_fineTuneBERT.py_** to fine-tune a pre-trained BERT model on your labelled        dataset.

5. Sentiment Classification
  Execute **_classifyingSentiments.py_** to classify sentiments using your fine-      tuned BERT model.

6. Applying the Fine-Tuned Model
  Run **_applyingBERT.py_** to predict sentiment on new text inputs using the fine-       tuned model.

7. Topic Modelling
Execute both **_topicModellingLDA.py_** and **_topicModellingNMF.py_** to perform unsupervised topic modelling and extract key themes.

8. Static Visualisation
Run **_staticSentiment.py_** to generate static plots (e.g. bar charts, word clouds).

9. Interactive Visualisation
Execute **_interactiveSentiment.py_** to launch an interactive dashboard for exploring sentiment trends and topics.

**Installation**
1. Clone the Repository
git clone https://github.com/yourusername/Sentiment-Analysis.git
cd Sentiment-Analysis

2. Create a Virtual Environment
python3 -m venv env
source env/bin/activate   # On Windows use: env\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt
Note: Ensure you have Python 3.8 or above installed.

4. Configure API Keys
Create a configuration file (e.g. config.py) and add your API keys and relevant settings for Reddit, The Guardian, Mediastack, and Apify.

**Usage**
After following the installation steps and setting up your configuration, execute the scripts sequentially as outlined above. For example, to run data preprocessing:
python scripts/dataPreprocessing.py

Similarly, run each script in the given order to progress through the full analysis pipeline.
