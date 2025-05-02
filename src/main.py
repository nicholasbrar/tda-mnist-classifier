import numpy as np
from sklearn.model_selection import train_test_split
from load_data import load_mnist
from features import get_all_features
from model import create_pipeline, train_model
from eval import evaluate_model

def main():
    # Load data
    X, y = load_mnist()

    # Extract features
    print("\nExtracting features...")
    X_features = get_all_features(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    # Create and train pipeline
    pipeline = create_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)

    # Evaluate model
    evaluate_model(pipeline, X_test, y_test)

if __name__ == "__main__":
    main()