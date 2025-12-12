def predict_sentiment(model, embedder, text):
    # Accept either a callable (e.g., get_embeddings) or an embedder object with .encode()
    embedding = embedder.encode([text])
    prediction = model.predict(embedding)[0]
    return prediction