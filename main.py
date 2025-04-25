import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

audio_folder = r"C:\Users\14123\Downloads\archive\Data\genres_original"
genres = ['hiphop', 'classical']

def load_audio(audio_path, sampling_rate=100, duration=30):
    audio_waveform, _ = librosa.load(audio_path, sr=sampling_rate, duration = duration)
    return audio_waveform

def compute_embedding(audio_waveform, time_delay=1, dimension=5, stride=10):
    takens_embedding = TakensEmbedding(time_delay=time_delay, dimension=dimension, stride=stride)
    return takens_embedding.fit_transform([audio_waveform])

def compute_persistence_diagram(point_cloud, metric="euclidean", homology_dimensions=[0, 1, 2]):
    VR = VietorisRipsPersistence(metric=metric, homology_dimensions=homology_dimensions)
    return VR.fit_transform(point_cloud)

def compute_persistence_image(diagram, sigma=0.5, n_bins=10, weight_function=None):
    p_image = PersistenceImage(sigma=sigma, n_bins=n_bins, weight_function=weight_function)
    return p_image.fit_transform(diagram)

def compute_persistent_homology(audio_path):
    y = load_audio(audio_path)
    point_cloud = compute_embedding(y)
    diagram = compute_persistence_diagram(point_cloud)
    img = compute_persistence_image(diagram)
    return img.reshape(-1)

X = []
y = []

for genre in genres:
    genre_path = os.path.join(audio_folder, genre)
    if os.path.isdir(genre_path):
        for filename in os.listdir(genre_path):
            if filename.endswith('.wav'):
                audio_path = os.path.join(genre_path, filename)
                print(f"Processing file: {audio_path}")
                try:
                    features = compute_persistent_homology(audio_path)
                    X.append(features)
                    y.append(genre)  
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")

X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5-fold validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = [] 

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # TODO: retune hyperparameters to new dimensionality of dataset
    clf = SVC(C=10, kernel='rbf', gamma='scale', random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)


mean_accuracy = np.mean(accuracies)
print(f'Mean Accuracy over K-Fold Cross-Validation: {mean_accuracy:.2%}')
