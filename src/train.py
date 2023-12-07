import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib

from sklearn.metrics import roc_auc_score

import re

import argparse

nltk.download('punkt')

def train_embeddings(dataset:str, output_path:str, split:float = 0.9, classifier:str = "LR", embeddings_file:str = None, device = 'cpu'):
    print("==Train with embedding==")
    if output_path[-1] != '/' and output_path[-1] != '\\':
        output_path += '/'

    df = pd.read_csv(dataset)

    if embeddings_file is None:
        model = SentenceTransformer("llmrails/ember-v1", device = device)
        embeddings = model.encode(df['comment_text'])
    else:
        embeddings = np.load(embeddings_file)
        assert embeddings.shape[0] == df.shape[0]

    split_n = int(df.shape[0]*split)

    X_train = embeddings[:split_n, :]
    X_val = embeddings[split_n:, :]

    Y_train = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']][:split_n]
    Y_val = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']][split_n:]

    print("Training ...")
    lr_models = {}
    lr_ra = []
    for col in Y_train.columns:
        best_model = None
        best_ra = 0
        for C in [10**i for i in range(-1, 4)]:
            if classifier == "LR":
                model = LogisticRegression(max_iter = 5000, C = C)
            elif classifier == "SVC":
                model = SVC(max_iter = 5000, C = C)
            model.fit(X_train, Y_train[col])
            Y_pred = model.predict(X_val)

            ra = roc_auc_score(Y_val[col], Y_pred)
            
            if ra > best_ra:
                best_ra = ra
                best_model = model
            
            print(f"{col}, C: {C}")
            print(f"Roc Auc: {ra}")
        lr_models[col] = best_model
        lr_ra.append(best_ra)
    print(f"Mean Roc Auc: {np.mean(lr_ra)}")

    for col, model in lr_models.items():
        joblib.dump(model, output_path+f'{col}.joblib')

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

def tokenize_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower()]
    return ' '.join(words)

def train_no_embeddings(dataset:str, output_path:str, split:float = 0.9, classifier:str = "LR"):
    print("==Train no embedding==")
    if output_path[-1] != '/' and output_path[-1] != '\\':
        output_path += '/'
    
    df = pd.read_csv(dataset)

    texts = [text for text in df['comment_text']]
    cleaned_texts = [clean_text(text) for text in texts]
    tokenized_texts = [tokenize_text(text) for text in cleaned_texts]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(tokenized_texts)

    split_n = int(df.shape[0]*split)

    X_train = vectors[:split_n, :]
    X_val = vectors[split_n:, :]

    Y_train = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']][:split_n]
    Y_val = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']][split_n:]

    print("Training ...")
    lr_models = {}
    lr_ra = []
    for col in Y_train.columns:
        best_model = None
        best_ra = 0
        for C in [10**i for i in range(-1, 4)]:
            if classifier == "LR":
                model = LogisticRegression(max_iter = 5000, C = C)
            elif classifier == "SVC":
                model = SVC(max_iter = 5000, C = C)
            model.fit(X_train, Y_train[col])
            Y_pred = model.predict(X_val)

            ra = roc_auc_score(Y_val[col], Y_pred)
            
            if ra > best_ra:
                best_ra = ra
                best_model = model
            
            print(f"{col}, C: {C}")
            print(f"Roc Auc: {ra}")
        lr_models[col] = best_model
        lr_ra.append(best_ra)
    print(f"Mean Roc Auc: {np.mean(lr_ra)}")

    for col, model in lr_models.items():
        joblib.dump(model, output_path+f'{col}.joblib')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help = "path to csv file which contains 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate' columns", type = str)
    parser.add_argument("output_path", help = "path and folder where to save model", type = str)
    parser.add_argument("-c", "--classifier", help = "type of classifier: SVC, LR (Logistic Regression) (default: LR)", type = str, default = "LR")
    parser.add_argument("-s", "--split", help = "train, validation split (default: 0.9)", type = float, default=0.9)
    parser.add_argument("-ue", "--use_embeddings", help = "use embedding model llmrails/ember-v1", action='store_true')
    parser.add_argument("-ef", "--embeddings_file", help = "path to npy file with embeddings", type = str, default = None)
    parser.add_argument("-d", "--device", help = "cpu or cuda", type = str, default = "cpu")

    args = parser.parse_args()
    if args.use_embeddings:
        train_embeddings(args.dataset, args.output_path, args.split, args.classifier, args.embeddings_file, args.device)
    else:
        train_no_embeddings(args.dataset, args.output_path, args.split, args.classifier)