import pandas as pd

# Function to load data from a CSV file
def load_data(file_path):
    try:
        df =  pd.read_csv(file_path)
        print("Data loaded successfully.")
        print("Shape of the data frame:", df.shape)

        # prepare the feature columns and target/label column
        df['features'] = df['ReviewTitle'].fillna('') + " " + df['ReviewBody'].fillna('')

        df['target'] = df['division'].fillna('neutral')

        # Return features and labels as lists so the caller can unpack them
        features = df['features'].tolist()
        labels = df['target'].tolist()
        return features, labels
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return [], []
    
# Example usage
# data = load_data("data/AllProductReviews2.csv")
