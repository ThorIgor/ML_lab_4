from src.inference import inference_embeddings, inference_no_embeddings
from src.train import train_embeddings, train_no_embeddings


def test_train_embeddings():
    train_embeddings("test_data/test_dataset.csv", "test_models_embeddings")

def test_train_embeddings_with_embeddings_file():
    train_embeddings("test_data/test_dataset.csv", "test_models_embeddings", "test_data/test_embeddings.npy")

def test_train_no_embeddings():
    train_no_embeddings("test_data/test_dataset.csv", "test_models_no_embeddings")

def test_inference_embeddings():
    inference_embeddings("test_data/test_inf_dataset.csv", "test_data/test_inf_output.csv", "test_models_embeddings")

def test_inference_embeddings_with_embeddings_file():
    inference_embeddings("test_data/test_inf_dataset.csv", "test_data/test_inf_output.csv", "test_models_embeddings", "test_data/test_inf_embeddings.npy")

def test_inference_no_embeddings():
    inference_no_embeddings("test_data/test_inf_dataset.csv", "test_data/test_inf_output.csv", "test_models_no_embeddings")

