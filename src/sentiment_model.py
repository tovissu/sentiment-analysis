from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def train_validate_sentiment_model(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create a logistic regression model
    model = LogisticRegression(max_iter=1000)

    # train the model
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Validation Accuracy: {accuracy}')

    report = classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])
    print(f'Classification Report:\n{report}')

    return model