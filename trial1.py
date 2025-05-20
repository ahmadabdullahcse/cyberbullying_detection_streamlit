import os
import requests
import zipfile
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    GPT2Tokenizer, GPT2ForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification
)

# Google Drive file ID of your zipped models
GOOGLE_DRIVE_FILE_ID = "1XtJ-8o_iNjvWbCgLWik6jLhDBJvx3DyC"
MODELS_DIR = "models"
ZIP_PATH = MODELS_DIR + ".zip"

# Functions to download from Google Drive
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_and_unzip_google_drive(file_id, extract_to):
    if not os.path.exists(extract_to):
        st.info(f"Downloading model files (~1.8GB). This may take a few minutes...")
        download_file_from_google_drive(file_id, ZIP_PATH)
        st.info("Extracting files...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(ZIP_PATH)
        st.success("Models downloaded and extracted successfully!")
    else:
        st.info("Model files already present, skipping download.")

# Download models if not present
download_and_unzip_google_drive(GOOGLE_DRIVE_FILE_ID, MODELS_DIR)

# Labels
labels = ['not bully', 'religious', 'others', 'sexual']

# Load models and tokenizers
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    models = []
    tokenizers = []

    # BERT
    bert_tokenizer = BertTokenizer.from_pretrained(f"{MODELS_DIR}/saved_bert_model")
    bert_model = BertForSequenceClassification.from_pretrained(f"{MODELS_DIR}/saved_bert_model").eval()
    tokenizers.append(bert_tokenizer)
    models.append(bert_model)

    # RoBERTa
    roberta_tokenizer = RobertaTokenizer.from_pretrained(f"{MODELS_DIR}/saved_roberta_model")
    roberta_model = RobertaForSequenceClassification.from_pretrained(f"{MODELS_DIR}/saved_roberta_model").eval()
    tokenizers.append(roberta_tokenizer)
    models.append(roberta_model)

    # ALBERT
    albert_tokenizer = AlbertTokenizer.from_pretrained(f"{MODELS_DIR}/saved_albert_model")
    albert_model = AlbertForSequenceClassification.from_pretrained(f"{MODELS_DIR}/saved_albert_model").eval()
    tokenizers.append(albert_tokenizer)
    models.append(albert_model)

    # GPT-2
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(f"{MODELS_DIR}/saved_gpt2_model")
    gpt2_model = GPT2ForSequenceClassification.from_pretrained(f"{MODELS_DIR}/saved_gpt2_model").eval()
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    tokenizers.append(gpt2_tokenizer)
    models.append(gpt2_model)

    # XLNet
    xlnet_tokenizer = XLNetTokenizer.from_pretrained(f"{MODELS_DIR}/saved_xlnet_model")
    xlnet_model = XLNetForSequenceClassification.from_pretrained(f"{MODELS_DIR}/saved_xlnet_model").eval()
    tokenizers.append(xlnet_tokenizer)
    models.append(xlnet_model)

    return models, tokenizers

models, tokenizers = load_model_and_tokenizer()

# Streamlit UI
st.title("Cyberbullying Detection using Transformer Ensemble")
st.write("Enter a sentence to classify it into cyberbullying categories.")

text_input = st.text_area("Enter text here")

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        total_logits = None

        for model, tokenizer in zip(models, tokenizers):
            inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                total_logits = logits if total_logits is None else total_logits + logits

        avg_logits = total_logits / len(models)
        probs = F.softmax(avg_logits, dim=1).numpy()[0]
        predicted_idx = np.argmax(probs)
        predicted_label = labels[predicted_idx]

        st.success(f"Predicted Label: **{predicted_label}**")
        st.write("### Class Probabilities:")
        for i, label in enumerate(labels):
            st.write(f"- {label}: {probs[i]*100:.2f}%")
