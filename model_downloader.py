from transformers import AutoTokenizer, AutoModel
import os

PATH = './models'

if not os.path.isdir(PATH):
    os.makedirs(PATH)

phobert = AutoModel.from_pretrained(
    'vinai/phobert-base-v2',
    cache_dir = PATH
)

phobert_tokenizer = AutoTokenizer.from_pretrained(
    'vinai/phobert-base-v2',
    cache_dir = PATH
)

os.rename('./models/models--vinai--phobert-base-v2', './models/phobert-base-v2')

simcse = AutoModel.from_pretrained(
    'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base',
    cache_dir = PATH
)
simcse_tokenizer = AutoTokenizer.from_pretrained(
    'VoVanPhuc/sup-SimCSE-VietNamese-phobert-base',
    cache_dir = PATH
)

os.rename('./models/models--VoVanPhuc--sup-SimCSE-VietNamese-phobert-base', './models/vvp_simcse')

miniLM = AutoModel.from_pretrained(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    cache_dir = PATH
)

miniLM_tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    cache_dir = PATH
)

os.rename('./models/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2', './models/multi_miniLM')

sbert = AutoModel.from_pretrained(
    'bkai-foundation-models/vietnamese-bi-encoder',
    cache_dir = PATH
)
sbert_tokenizer = AutoTokenizer.from_pretrained(
    'bkai-foundation-models/vietnamese-bi-encoder',
    cache_dir = PATH
)

os.rename('./models/models--bkai-foundation-models--vietnamese-bi-encoder', './models/bkai_visbert')

