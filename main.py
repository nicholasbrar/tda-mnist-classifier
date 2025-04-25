import os
import librosa
import numpy as np

from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Define path to training data
audio_folder = r"C:\Users\14123\Downloads\archive\Data\genres_original"

# Only use 'hiphop' and 'classical' genres
genres = ['hiphop', 'classical']

# Function to preprocess and compute persistent homology features for a single audio file
def compute_persistent_homology(audio_path):
    y, sr = librosa.load(audio_path, sr=100, duration=30)  # Load the audio file
    takens = TakensEmbedding(time_delay=1, dimension=2, stride=10, ensure_last_value=True)
    point_cloud = takens.fit_transform([y])  # Time-Delay embedding
    VR = VietorisRipsPersistence(metric="euclidean", homology_dimensions=[0, 1])
    diagram = VR.fit_transform(point_cloud)
    p_image = PersistenceImage(sigma=0.5, n_bins=5, weight_function=None)
    img = p_image.fit_transform(diagram)  # Get the persistence image
    return img.reshape(-1)  # Flatten the image for ML use
