import json
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, balanced_accuracy_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import torch
from sklearn.metrics import roc_auc_score, roc_curve

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load the JSON files
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

burnout_data = load_data("clean_burnout.json")
no_burnout_data = load_data("no_burnout.json")

# Combine data and create labels
texts = [item["title"] + " " + item["body"] for item in burnout_data] + \
        [item["title"] + " " + item["body"] for item in no_burnout_data]
labels = [1] * len(burnout_data) + [0] * len(no_burnout_data)

# Extract subreddits for batching
subreddits = [item["subreddit"] for item in burnout_data + no_burnout_data]

# Text preprocessing using NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)

texts = [preprocess_text(text) for text in texts]

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Convert texts to BERT embeddings
def bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        # Take the [CLS] token's embedding as a sentence-level representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(cls_embedding)
    return np.vstack(embeddings)

X = bert_embeddings(texts)
y = np.array(labels)

# Train-test split (70/30) and SMOTE for data augmentation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Function to evaluate model and plot confusion matrix
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    mean_cv_accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))
    mean_cv_f1 = np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='f1'))
    
    # Fit the model and make predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Check if the model supports predict_proba or decision_function for AUC-ROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None

    # Calculate performance metrics
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    
    # Classification report
    print(f"Model: {model.__class__.__name__}")
    print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")
    print(f"Mean CV F1 Score: {mean_cv_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}" if y_prob is not None else "AUC-ROC: Not available")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Burnout', 'Burnout'], 
                yticklabels=['No Burnout', 'Burnout'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model.__class__.__name__}")
    plt.show()

# Models to compare
models = [
    LogisticRegression(max_iter=1000, random_state=42),
    make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=42)),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(random_state=42),
    AdaBoostClassifier(random_state=42)
]

# Evaluate each model and plot results
for model in models:
    evaluate_model(model, X_train, y_train, X_test, y_test)