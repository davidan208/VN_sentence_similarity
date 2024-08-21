from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize
import py_vncorenlp
from datasets import load_dataset
import os
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from tqdm import tqdm
import numpy as np
import csv

# PATH from the location of this evaluate.py file
VNCORENLP_PATH  = os.path.join(os.path.dirname(__file__),'api/vncorenlp')
DATA_PATH       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODEL_PATH      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
CURRENT_PATH    = os.path.dirname(__file__)
EVALUATE_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluate')

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir= VNCORENLP_PATH)

def evaluate_for_phobert_mean(dataset, tokenizer, model, output_path, name: str):
    labels = [df['score'] for df in dataset]

    embeddings1 = []
    embeddings2 = []
    with torch.no_grad():
        for df in tqdm(dataset):
            s_sent1 = model(torch.tensor([tokenizer.encode(rdrsegmenter.word_segment(df['sentence1'])[0], truncation = True, max_length = 512)]), output_hidden_states = True).last_hidden_state.mean(dim = 1).numpy()
            s_sent2 = model(torch.tensor([tokenizer.encode(rdrsegmenter.word_segment(df['sentence2'])[0], truncation = True, max_length = 512)]), output_hidden_states = True).last_hidden_state.mean(dim = 1).numpy()
            embeddings1.append(s_sent1)
            embeddings2.append(s_sent2)
    
    embeddings1 = np.array(embeddings1).reshape(len(labels), -1)
    embeddings2 = np.array(embeddings2).reshape(len(labels), -1)

    # Remake from EmbeddingSimilarityEvaluator
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(labels, dot_products)
    eval_spearman_dot, _ = spearmanr(labels, dot_products)
            
    if output_path is not None:
        csv_path = os.path.join(EVALUATE_PATH, f'similarity_evaluation_{name}_results.csv')

        csv_headers = [
            "epoch",
            "steps",
            "cosine_pearson",
            "cosine_spearman",
            "euclidean_pearson",
            "euclidean_spearman",
            "manhattan_pearson",
            "manhattan_spearman",
            "dot_pearson",
            "dot_spearman",
        ]

        with open(csv_path, newline="", mode="w", encoding = "utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            writer.writerow(
                [
                    -1,
                    -1,
                    eval_pearson_cosine,
                    eval_spearman_cosine,
                    eval_pearson_euclidean,
                    eval_spearman_euclidean,
                    eval_pearson_manhattan,
                    eval_spearman_manhattan,
                    eval_pearson_dot,
                    eval_spearman_dot,
                ]
            )

def evaluate_for_phobert_cls(dataset, tokenizer, model, output_path, name: str):
    labels = [df['score'] for df in dataset]

    embeddings1 = []
    embeddings2 = []
    with torch.no_grad():
        for df in tqdm(dataset):
            s_sent1 = model(torch.tensor([tokenizer.encode(rdrsegmenter.word_segment(df['sentence1'])[0], truncation = True, max_length = 512)]), output_hidden_states = True).last_hidden_state[:,0,:].numpy()
            s_sent2 = model(torch.tensor([tokenizer.encode(rdrsegmenter.word_segment(df['sentence2'])[0], truncation = True, max_length = 512)]), output_hidden_states = True).last_hidden_state[:,0,:].numpy()
            embeddings1.append(s_sent1)
            embeddings2.append(s_sent2)
    
    embeddings1 = np.array(embeddings1).reshape(len(labels), -1)
    embeddings2 = np.array(embeddings2).reshape(len(labels), -1)

    # Remake from EmbeddingSimilarityEvaluator
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(labels, dot_products)
    eval_spearman_dot, _ = spearmanr(labels, dot_products)
            
    if output_path is not None:
        csv_path = os.path.join(EVALUATE_PATH, f'similarity_evaluation_{name}_results.csv')

        with open(csv_path, newline="", mode="a", encoding = "utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    -1,
                    -1,
                    eval_pearson_cosine,
                    eval_spearman_cosine,
                    eval_pearson_euclidean,
                    eval_spearman_euclidean,
                    eval_pearson_manhattan,
                    eval_spearman_manhattan,
                    eval_pearson_dot,
                    eval_spearman_dot,
                ]
            )


# Return segmented strings
def convert_dataset(dataset):
    dataset_samples = []
    for df in dataset:
        score = float(df['score']) / 5.0
        input_example = InputExample(
                texts = [
                        tokenize(df['sentence1']), 
                        tokenize(df['sentence2'])], 
                label = score)
        
        dataset_samples.append(input_example)
    return dataset_samples


vi_sts = load_dataset(DATA_PATH)['train']
df_test = vi_sts.filter(lambda example: example['split'] == 'test')
data_name = 'vi-sts-test'



model_list = ['phobert-mean','phobert-cls','visbert', 'multi_miniLM', 'vvp_simcse']
for name in model_list:
    if name == 'phobert-mean':
        model = AutoModel.from_pretrained(os.path.join(MODEL_PATH, 'phobert-base-v2/snapshots/2b51e367d92093c9688112098510e6a58bab67cd'), local_files_only = True)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, 'phobert-base-v2/snapshots/2b51e367d92093c9688112098510e6a58bab67cd'), local_files_only = True)
        evaluate_for_phobert_mean(df_test, tokenizer, model, EVALUATE_PATH, name = data_name)
         
    elif name == 'phobert-cls':
        model = AutoModel.from_pretrained(os.path.join(MODEL_PATH, 'phobert-base-v2/snapshots/2b51e367d92093c9688112098510e6a58bab67cd'), local_files_only = True)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, 'phobert-base-v2/snapshots/2b51e367d92093c9688112098510e6a58bab67cd'), local_files_only = True)
        evaluate_for_phobert_cls(df_test, tokenizer, model, EVALUATE_PATH, name = data_name)
    elif name == 'visbert':
        model = SentenceTransformer(os.path.join(MODEL_PATH, 'bkai_visbert/snapshots/84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0'), local_files_only= True, trust_remote_code= True)
        test_samples = convert_dataset(df_test)
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size = 4,name = data_name)
        test_evaluator(model, output_path = EVALUATE_PATH)
    elif name == 'multi_miniLM':
        model = SentenceTransformer(os.path.join(MODEL_PATH, 'multi_miniLM/snapshots/bf3bf13ab40c3157080a7ab344c831b9ad18b5eb'), local_files_only= True, trust_remote_code= True)
        test_samples = convert_dataset(df_test)
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size = 4,name = data_name)
        test_evaluator(model, output_path = EVALUATE_PATH)
    elif name == 'vvp_simcse':
        model = SentenceTransformer(os.path.join(MODEL_PATH, 'vvp_simcse/snapshots/608779b86741a8acd8c8d38132974ff04086b138'), local_files_only = True, trust_remote_code = True)
        test_samples = convert_dataset(df_test)
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size = 4,name = data_name)
        test_evaluator(model, output_path = EVALUATE_PATH)

import pandas as pd
evaluator = pd.read_csv(os.path.join(EVALUATE_PATH, f'similarity_evaluation_{data_name}_results.csv'))
evaluator.insert(0, 'model', model_list)
evaluator.to_csv(os.path.join(EVALUATE_PATH, f'similarity_evaluation_{data_name}_results.csv'), index = False)

print(f"[DONE] Check your result at path: {EVALUATE_PATH}/similarity_evaluation_{data_name}_results.csv")