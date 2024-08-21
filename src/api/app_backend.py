from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import py_vncorenlp
import torch
from .underthesea_offline import underthesea_off
from sklearn.metrics.pairwise import cosine_similarity
import os
app = FastAPI()

# Load models
phobert = AutoModel.from_pretrained(
    './models/phobert-base-v2/snapshots/2b51e367d92093c9688112098510e6a58bab67cd',
    local_files_only=True
)
phobert_tokenizer = AutoTokenizer.from_pretrained(
    './models/phobert-base-v2/snapshots/2b51e367d92093c9688112098510e6a58bab67cd',
    local_files_only=True
)
simcse = AutoModel.from_pretrained(
    './models/vvp_simcse/snapshots/608779b86741a8acd8c8d38132974ff04086b138',
    local_files_only=True
)
simcse_tokenizer = AutoTokenizer.from_pretrained(
    './models/vvp_simcse/snapshots/608779b86741a8acd8c8d38132974ff04086b138',
    local_files_only=True
)

miniLM = AutoModel.from_pretrained(
    './models/multi_miniLM/snapshots/bf3bf13ab40c3157080a7ab344c831b9ad18b5eb',
    local_files_only = True
)

miniLM_tokenizer = AutoTokenizer.from_pretrained(
    './models/multi_miniLM/snapshots/bf3bf13ab40c3157080a7ab344c831b9ad18b5eb',
    local_files_only = True
)

sbert = AutoModel.from_pretrained(
    './models/bkai_visbert/snapshots/84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0',
    local_files_only = True
)
sbert_tokenizer = AutoTokenizer.from_pretrained(
    './models/bkai_visbert/snapshots/84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0',
    local_files_only = True
)

os.chdir('./src/api/vncorenlp')
# Load segmenter
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir="./")

# Define input
class TextPair(BaseModel):
    sentence1: str
    sentence2: str

class TextParagraph(BaseModel):
    sentence: str
    paragraph: str

# Define response models
class SimilarityResponse(BaseModel):
    similarity: float

class MostSimilarResponse(BaseModel):
    most_similar_sentence: str
    score: float

# Embedding Methods
def simcse_embedding(text: str):
    segmented_sent = rdrsegmenter.word_segment(text)
    with torch.no_grad():
        tokenize_sent = simcse_tokenizer.encode(segmented_sent[0], truncation=True, max_length=512, return_tensors="pt")
        embedding = simcse(tokenize_sent, output_hidden_states=True, return_dict=True).pooler_output.numpy()
    return embedding

def phobert_mean_embedding(text: str):
    segmented_sent  = rdrsegmenter.word_segment(text)
    with torch.no_grad():
        tokenize_sent   = phobert_tokenizer.encode(segmented_sent[0], truncation = True, max_length = 512, return_tensors = "pt")
        embedding       = phobert(torch.tensor([tokenize_sent]), output_hidden_states = True).last_hidden_state.mean(dim = 1).numpy()
    return embedding

def phobert_cls_embedding(text: str):
    segmented_sent = rdrsegmenter.word_segment(text)
    with torch.no_grad():
        tokenize_sent = phobert_tokenizer.encode(segmented_sent[0], truncation=True, max_length=512, return_tensors="pt")
        embedding = phobert(tokenize_sent, output_hidden_states=True).last_hidden_state[:, 0, :].numpy()
    return embedding

def miniLM_embedding(text: str):
    segmented_sent = rdrsegmenter.word_segment(text)
    with torch.no_grad():
        tokenize_sent = miniLM_tokenizer.encode(segmented_sent[0], truncation = True, max_length = 512, return_tensors= "pt")
        embedding = miniLM(tokenize_sent, output_hidden_states = True).last_hidden_state.mean(dim = 1).numpy()
    return embedding

def sbert_embedding(text: str):
    segmented_sent = rdrsegmenter.word_segment(text)
    with torch.no_grad():
        tokenize_sent = sbert_tokenizer.encode(segmented_sent[0], truncation = True, max_length = 512, return_tensors= "pt")
        embedding = sbert(tokenize_sent, output_hidden_states = True).last_hidden_state.mean(dim = 1).numpy()
    return embedding

def paragraph_splitting(paragraph):
    return underthesea_off.sent_tokenize(paragraph)

@app.post("/similarity", response_model=SimilarityResponse)
def get_similarity(data: TextPair, method: str):
    if method not in ["phobert-cls", "phobert-mean", "sup-simcse", "multi-miniLM", "vi-sbert"]:
        raise HTTPException(status_code=400, detail="Invalid method")
    
    if method == "phobert-cls":
        embedding1 = phobert_cls_embedding(data.sentence1)
        embedding2 = phobert_cls_embedding(data.sentence2)
    elif method == "phobert-mean":
        # Assuming phobert_mean_embedding is implemented similarly
        embedding1 = phobert_mean_embedding(data.sentence1)
        embedding2 = phobert_mean_embedding(data.sentence2)
    elif method == "sup-simcse":
        embedding1 = simcse_embedding(data.sentence1)
        embedding2 = simcse_embedding(data.sentence2)
    elif method == "multi-miniLM":
        embedding1 = miniLM_embedding(data.sentence1)
        embedding2 = miniLM_embedding(data.sentence2)
    elif method == 'vi-sbert':
        embedding1 = sbert_embedding(data.sentence1)
        embedding2 = sbert_embedding(data.sentence2)
    
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return SimilarityResponse(similarity=similarity)

@app.post("/most-similar", response_model=MostSimilarResponse)
def get_most_similar_sentence(data: TextParagraph, method: str):
    if method not in ["phobert-cls", "phobert-mean", "sup-simcse", "multi-miniLM", "vi-sbert"]:
        raise HTTPException(status_code=400, detail="Invalid method")
    
    try:
        assert method in ["phobert-cls", "phobert-mean", "sup-simcse", "multi-miniLM", "vi-sbert"], 'Method not found'

        sent = data.sentence
        paragraph = paragraph_splitting(data.paragraph)

        max_similarity = 0
        id = -1

        match method:
            case 'phobert-mean':
                embedding_func = phobert_mean_embedding
            case 'phobert-cls':
                embedding_func = phobert_cls_embedding
            case 'sup-simcse':
                embedding_func = simcse_embedding
            case 'multi-miniLM':
                embedding_func = miniLM_embedding
            case "vi-sbert":
                embedding_func = sbert_embedding

        sent_embedding1 = embedding_func(sent)
        for idx, sentence in enumerate(paragraph):
            sent_embedding2 = embedding_func(sentence)
            similarity = cosine_similarity(sent_embedding1, sent_embedding2)
            if max_similarity < similarity:
                max_similarity = similarity
                id = idx

        return MostSimilarResponse(most_similar_sentence = paragraph[id], score = max_similarity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
