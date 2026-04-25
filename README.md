# YouTube Sentiment Analysis — MLOps Pipeline

A production-grade MLOps pipeline that classifies user-generated text 
as positive, negative, or neutral. Built with a focus on reproducibility, 
experiment tracking, and automated cloud deployment.

---

## Demo

Click on the image below to watch the demo:

[![Watch the demo](demo.gif)](your_google_drive_link_here)

---

## Architecture

The system is structured as a 5-stage DVC pipeline, with all experiments 
tracked in MLflow and the final model deployed to AWS EC2 via GitHub CI/CD.

**1. Data Ingestion**
Loads a Reddit sentiment dataset (37,000+ comments labeled as positive, 
neutral, or negative) from a remote URL, removes duplicates and missing 
values, and saves train/test splits to a local data directory.

**2. Data Preprocessing**
Applies a full NLP preprocessing pipeline: lowercasing, whitespace 
stripping, special character removal, stop word filtering (preserving 
negation words such as "not" and "however"), and lemmatization via NLTK. 
Saves the preprocessed data for downstream stages.

**3. Model Building**
Trains a LightGBM classifier using the best configuration identified 
through systematic experimentation: TF-IDF vectorization with trigram 
features, 1,000 max features, and SMOTE oversampling to handle class 
imbalance. Saves the trained model and vectorizer as pickle files.

**4. Model Evaluation**
Evaluates the model on the held-out test set, logs all metrics 
(accuracy, precision, recall, F1 per class) and artifacts (confusion 
matrix, classification report) to the MLflow tracking server hosted on 
AWS EC2. Saves run ID and model path for the registration stage.

**5. Model Registration**
Registers the best model to the MLflow Model Registry, making it 
available for serving via the Flask API.

---

## Experiment Tracking

All experiments were tracked in MLflow hosted on AWS EC2, covering:

- Vectorization strategy: Bag of Words vs TF-IDF, unigram/bigram/trigram
- Max features: 1,000 to 10,000
- Imbalance handling: class weights, SMOTE, random undersampling, ENN
- Model selection: Random Forest, XGBoost, Decision Tree, KNN, LightGBM

Final configuration selected based on recall for the minority 
(negative) class across all experiments.

---

## Stack

| Component | Technology |
|---|---|
| ML pipeline orchestration | DVC |
| Experiment tracking | MLflow |
| Model | LightGBM + TF-IDF |
| NLP preprocessing | NLTK |
| API | Flask |
| Containerization | Docker |
| Cloud deployment | AWS EC2 |
| CI/CD | GitHub Actions |

---

## CI/CD Deployment

Pushing to the main branch triggers a GitHub Actions workflow that:
- Builds a Docker image of the Flask API
- Pushes the image to AWS ECR
- Pulls and runs the updated container on AWS EC2 automatically

---

*Built following the MLOps course by Bappy Ahmed (DS with Bappy)*
