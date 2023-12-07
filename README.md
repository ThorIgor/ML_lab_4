# Machine Learning Lab 4

This repository contains scripts for inference and training of classification models

Requirements installation:

```
pip install -r requirements.txt
```

# Training

Training models:

```
python ./src/train.py path/to/dataset.csv path/to/output_models 
```

<details>
<summary>train.py usage</summary>

```
usage: train.py [-h] [-c CLASSIFIER] [-s SPLIT] [-ue] [-ef EMBEDDINGS_FILE] [-d DEVICE] dataset output_path

positional arguments:
  dataset               path to csv file which contains 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate' columns
  output_path           path and folder where to save model

options:
  -h, --help            show this help message and exit
  -c CLASSIFIER, --classifier CLASSIFIER
                        type of classifier: SVC, LR (Logistic Regression) (default: LR)
  -s SPLIT, --split SPLIT
                        train, validation split (default: 0.9)
  -ue, --use_embeddings
                        use embedding model llmrails/ember-v1
  -ef EMBEDDINGS_FILE, --embeddings_file EMBEDDINGS_FILE
                        path to npy file with embeddings
  -d DEVICE, --device DEVICE
                        cpu or cuda
```

</details>

# Inference

Inference:

```
python ./src/inference.py  path/to/input.csv path/to/output.csv path/to/models
```

<details>
<summary>inference.py usage</summary>

```
usage: inference.py [-h] [-ue] [-ef EMBEDDINGS_FILE] [-d DEVICE] input_file output_file models

positional arguments:
  input_file            path to csv file which contains 'comment_text' column
  output_file           path and name for output file
  models                path to folder with classification model

options:
  -h, --help            show this help message and exit
  -ue, --use_embeddings
                        use embedding model llmrails/ember-v1
  -ef EMBEDDINGS_FILE, --embeddings_file EMBEDDINGS_FILE
                        path to npy file with embeddings
  -d DEVICE, --device DEVICE
                        cpu or cuda
```

</details>

# Tests

Run tests:

```
python -m pytest -cov=src ./tests/test.py
```