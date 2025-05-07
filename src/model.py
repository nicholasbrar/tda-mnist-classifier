from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import make_pipeline, make_union
from gtda.images import Binarizer, RadialFiltration, HeightFiltration
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, PersistenceEntropy, Amplitude
from sklearn import set_config
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.to_numpy().reshape((-1, 28, 28))
y = y.to_numpy()  

train_size, test_size = 100, 10
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=666
)

direction_list = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
center_list = [
    [13, 6], [6, 13], [13, 13], [20, 13], [13, 20], 
    [6, 6], [6, 20], [20, 6], [20, 20]
]

filtration_list = (
    [HeightFiltration(direction=np.array(direction), n_jobs=-1) for direction in direction_list]
    + [RadialFiltration(center=np.array(center), n_jobs=-1) for center in center_list]
)

diagram_steps = [
    [
        Binarizer(threshold=0.4, n_jobs=-1),
        filtration,
        CubicalPersistence(n_jobs=-1),
        Scaler(n_jobs=-1),
    ]
    for filtration in filtration_list
]

metric_list = [
    {"metric": "bottleneck", "metric_params": {}},
    {"metric": "wasserstein", "metric_params": {"p": 1}},
    {"metric": "wasserstein", "metric_params": {"p": 2}},
    {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 2, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 1, "n_bins": 100}},
    {"metric": "landscape", "metric_params": {"p": 2, "n_layers": 2, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}},
    {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 1, "sigma": 3.2, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 1.6, "n_bins": 100}},
    {"metric": "heat", "metric_params": {"p": 2, "sigma": 3.2, "n_bins": 100}},
]

feature_union = make_union(
    *[PersistenceEntropy(nan_fill_value=-1)]
    + [Amplitude(**metric, n_jobs=-1) for metric in metric_list]
)

tda_union = make_union(
    *[make_pipeline(*diagram_step, feature_union) for diagram_step in diagram_steps],
    n_jobs=-1
)

from sklearn.feature_selection import SelectKBest, mutual_info_classif

tda_union = make_pipeline(
    tda_union,
    SelectKBest(score_func=mutual_info_classif, k='all')  
)

set_config(display='diagram')


full_pipeline = Pipeline([
    ('tda_features', tda_union),  
    ('classifier', RandomForestClassifier(random_state=42))  
], verbose=True)

full_pipeline.fit(X_train, y_train)

accuracy = full_pipeline.score(X_test, y_test)
print(f"\nTest accuracy: {accuracy:.2%}")

print("\nPredictions vs Actual:")
for pred, true in zip(full_pipeline.predict(X_test), y_test):
    print(f"Predicted: {pred}, Actual: {true}")