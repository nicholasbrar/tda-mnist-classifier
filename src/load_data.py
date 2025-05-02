import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist(train_size=5000):
    print("Loading MNIST data...")
    X, y = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True)
    # Convert '1' - '9' to integers 1 - 9
    y = y.astype(int)
    # Stratify to ensure equal class disttribution as original dataset
    X, _, y, _ = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)
    return X, y