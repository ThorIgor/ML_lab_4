import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

import os
import re

import argparse


def inference_embeddings(input_file:str, output_file:str, models_folder:str, embeddings_file:str = None, device:str = 'cpu'):
    if models_folder[-1] != '/' and models_folder[-1] != '\\':
        models_folder += '/'

    df = pd.read_csv(input_file)

    if embeddings_file is None:
        model = SentenceTransformer("llmrails/ember-v1", device = device)
        embeddings = model.encode(df['comment_text'])
    else:
        embeddings = np.load(embeddings_file)

    lr_models = {}
    for file in os.listdir(models_folder):
        if 'joblib' in file:
            lr_models[file.split(".")[0]] = joblib.load(models_folder + file)

    sb = pd.DataFrame(df['id'])
    for key, model in lr_models.items():
        sb[key] = model.predict(embeddings)

    sb.to_csv(output_file, index = False)

def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

def tokenize_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower()]
    return ' '.join(words)

def inference_no_embeddings(input_file:str, output_file:str, models_folder:str):
    if models_folder[-1] != '/' and models_folder[-1] != '\\':
        models_folder += '/'

    df = pd.read_csv(input_file)

    texts = [text for text in df['comment_text']]
    cleaned_texts = [clean_text(text) for text in texts]
    tokenized_texts = [tokenize_text(text) for text in cleaned_texts]
    vectorizer = TfidfVectorizer(vocabulary = eval(open(models_folder + "vocabualry.txt", "r").read()))
    vectors = vectorizer.fit_transform(tokenized_texts)

    lr_models = {}
    for file in os.listdir(models_folder):
        if 'joblib' in file:
            lr_models[file.split(".")[0]] = joblib.load(models_folder + file)

    sb = pd.DataFrame(df['id'])
    for key, model in lr_models.items():
        sb[key] = model.predict(vectors)

    sb.to_csv(output_file, index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help = "path to csv file which contains 'comment_text' column", type = str)
    parser.add_argument("output_file", help = "path and name for output file", type = str)
    parser.add_argument("models", help = "path to folder with classification model", type = str, default = "models/LR_no_embeddings")
    parser.add_argument("-ue", "--use_embeddings", help = "use embedding model llmrails/ember-v1", action="store_false")
    parser.add_argument("-ef", "--embeddings_file", help = "path to npy file with embeddings", type = str, default = None)
    parser.add_argument("-d", "--device", help = "cpu or cuda", type = str, default = "cpu")
    
    args = parser.parse_args()
    if args.use_embeddings:
        inference_embeddings(args.input_file, args.output_file, args.models, args.embeddings_file, args.device)
    else:
        inference_no_embeddings(args.input_file, args.output_file, args.models)
    

