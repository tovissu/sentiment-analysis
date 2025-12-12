from sentence_transformers import SentenceTransformer

# Function to get embeddings for a list of texts using a specified sentence-BERT model

def get_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    embedding_model = SentenceTransformer(model_name)
    return embedding_model.encode(texts)

def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)