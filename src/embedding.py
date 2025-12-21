from sentence_transformers import SentenceTransformer
import gensim.downloader as api
from transformers import BertTokenizer, BertModel
import torch
from transformers import RobertaTokenizer, RobertaModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import ElectraTokenizer, ElectraModel
import numpy as np

# Function to get embeddings for a list of texts using a specified sentence-BERT model

def get_sentence_bert_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    embedding_model = SentenceTransformer(model_name)
    return embedding_model.encode(texts)

def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def get_word2vec_embedding(text):
    try:
        # Load pre-trained Word2Vec model
        # This might take a while the first time as it downloads the model
        model = api.load("word2vec-google-news-300")
        
        # Handle both single text and list of texts
        if isinstance(text, list):
            embeddings = []
            for t in text:
                words = t.lower().split()
                vectorized_words = [model[word] for word in words if word in model.key_to_index]
                if vectorized_words:
                    embeddings.append(np.mean(vectorized_words, axis=0))
                else:
                    embeddings.append(np.zeros(model.vector_size))
            return np.array(embeddings)
        else:
            words = text.lower().split()
            # Filter out words not in the vocabulary
            vectorized_words = [model[word] for word in words if word in model.key_to_index]
            if vectorized_words:
                return np.mean(vectorized_words, axis=0) # Average word vectors for sentence embedding
            else:
                return np.zeros(model.vector_size)
    except Exception as e:
        print(f"Error loading or using Word2Vec model: {e}")
        return None

def get_fasttext_embedding(text):
    try:
        # Load pre-trained FastText model
        # This might take a while the first time as it downloads the model
        model = api.load("fasttext-wiki-news-subwords-300")
        
        # Handle both single text and list of texts
        if isinstance(text, list):
            embeddings = []
            for t in text:
                words = t.lower().split()
                vectorized_words = [model[word] for word in words if word in model.key_to_index]
                if vectorized_words:
                    embeddings.append(np.mean(vectorized_words, axis=0))
                else:
                    embeddings.append(np.zeros(model.vector_size))
            return np.array(embeddings)
        else:
            words = text.lower().split()
            # Filter out words not in the vocabulary
            vectorized_words = [model[word] for word in words if word in model.key_to_index]
            if vectorized_words:
                return np.mean(vectorized_words, axis=0) # Average word vectors for sentence embedding
            else:
                return np.zeros(model.vector_size)
    except Exception as e:
        print(f"Error loading or using FastText model: {e}")
        return None

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    encoded_input = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = bert_model(**encoded_input)
    # Get the embeddings for the [CLS] token (sentence embedding)
    embeddings = output.last_hidden_state[:, 0, :].numpy()
    return embeddings.squeeze() if isinstance(text, str) else embeddings

# Load pre-trained model tokenizer and model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

def get_roberta_embedding(text):
    encoded_input = roberta_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = roberta_model(**encoded_input)
    # Get the embeddings for the <s> token (similar to [CLS] in BERT for sentence embedding)
    embeddings = output.last_hidden_state[:, 0, :].numpy()
    return embeddings.squeeze() if isinstance(text, str) else embeddings

# Load pre-trained model tokenizer and model
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_distilbert_embedding(text):
    encoded_input = distilbert_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = distilbert_model(**encoded_input)
    # Get the embeddings for the [CLS] token
    embeddings = output.last_hidden_state[:, 0, :].numpy()
    return embeddings.squeeze() if isinstance(text, str) else embeddings


# Load pre-trained model tokenizer and model
electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
electra_model = ElectraModel.from_pretrained('google/electra-small-discriminator')

def get_electra_embedding(text):
    encoded_input = electra_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = electra_model(**encoded_input)
    # Get the embeddings for the [CLS] token
    embeddings = output.last_hidden_state[:, 0, :].numpy()
    return embeddings.squeeze() if isinstance(text, str) else embeddings

def call_embedding(text, embedding_model_name):
  #add switch case based on embedding_model_name and call the respective function
  if embedding_model_name == 'word2vec':
      return get_word2vec_embedding(text)
  elif embedding_model_name == 'fasttext':
      return get_fasttext_embedding(text)
  elif embedding_model_name == 'bert':
      return get_bert_embedding(text)
  elif embedding_model_name == 'roberta':
      return get_roberta_embedding(text)
  elif embedding_model_name == 'distilbert':
      return get_distilbert_embedding(text)
  elif embedding_model_name == 'electra':
      return get_electra_embedding(text)
  elif embedding_model_name == 'sbert':
      return get_sentence_bert_embeddings(text)
  else:
      raise ValueError(f"Unknown embedding model name: {embedding_model_name}")
  
