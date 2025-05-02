# src/eval.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluates the trained pipeline and visualizes results.

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
    """
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nFinal Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix - now dynamically sized
    cm = confusion_matrix(y_test, y_pred)
    n_classes = cm.shape[0]
    
    plt.figure(figsize=(max(6, n_classes), max(6, n_classes)))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, range(n_classes))
    plt.yticks(tick_marks, range(n_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()

    # Feature importance plot with automatic feature names
    try:
        from features import get_feature_names
        feature_names = get_feature_names()
    except:
        feature_names = [f"Feature {i+1}" for i in range(X_test.shape[1])]

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=10, random_state=42
    )

    sorted_idx = result.importances_mean.argsort()[::-1]

    plt.figure(figsize=(12, max(6, len(feature_names)//2)))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False, labels=np.array(feature_names)[sorted_idx]
    )
    plt.title("Permutation Importances")
    plt.tight_layout()
    plt.show()