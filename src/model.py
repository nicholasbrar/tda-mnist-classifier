from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def create_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=100,
            kernel='rbf',
            gamma='auto',
            class_weight='balanced',
            random_state=42
        ))
    ])
    return pipeline

def train_model(pipeline, X_train, y_train):
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    return pipeline