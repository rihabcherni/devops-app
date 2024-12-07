from flask import Flask, request, jsonify
import base64
import pickle
import numpy as np
import librosa
from io import BytesIO
import xmlrunner 
import os

app = Flask(__name__)

# Charger le modèle SVM
# with open("model/svm_model.pkl", "rb") as f:
#     svm_model = pickle.load(f)


# Définir le chemin absolu du modèle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "svm_model.pkl")

# Charger le modèle SVM
with open(MODEL_PATH, "rb") as f:
    svm_model = pickle.load(f)


# Genres list (same order as in your training data)
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def predict_genre(audio_data, clf):
    # Load and preprocess the audio file
    signal, rate = librosa.load(BytesIO(audio_data))
    hop_length = 512
    n_fft = 2048
    n_mels = 128
    S = librosa.feature.melspectrogram(y=signal, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB = S_DB.flatten()[:1200]

    # Predict the genre
    genre_label = clf.predict([S_DB])[0]
    return genres[genre_label]

# Classe pour l'erreur de traitement
class AudioProcessingError(Exception):
    pass

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    try:
        # Get the audio data from the request
        data = request.json.get('wav_music')
        if not data:
            return jsonify({'error': 'Aucun fichier audio fourni'}), 400
        
        # Decode the Base64 audio data
        audio_data = base64.b64decode(data)
        
        # Call the predict_genre function with the decoded audio data
        predicted_genre = predict_genre(audio_data, svm_model)
        
        return jsonify({'genre': predicted_genre})
    
    except AudioProcessingError:
        return jsonify({'error': 'Erreur de traitement du fichier audio'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
 