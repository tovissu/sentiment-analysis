from src.data_loader import load_data
from src.embedding import get_embeddings, get_embedding_model
from src.sentiment_model import train_validate_sentiment_model
from src.sentiment_prediction import predict_sentiment

def main():
    print("Starting main()")
    # Load data
    features, labels = load_data('data/AllProductReviews2.csv')

    # Get embeddings
    embeddings = get_embeddings(features)

    # Train and validate sentiment model
    model = train_validate_sentiment_model(embeddings, labels)
   

    # Example predictions
    sample_texts = [
        "I love using this product!",
        "Today is Sunday and I am attending AI ML training."
    ]
    
    for sample_text in sample_texts:
        prediction = predict_sentiment(model, get_embedding_model(), sample_text)
        print(f'\nText: "{sample_text}-->{prediction}"')

if __name__ == "__main__":
    main()