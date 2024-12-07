from flask import Flask, request, jsonify
import base64
import pickle
import numpy as np
import librosa
from io import BytesIO
import xmlrunner 

app = Flask(__name__)

with open("../model/vgg_model.pkl", "rb") as f:
    vgg_model = pickle.load(f)

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def predict_genre(audio_data, clf):
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

@app.route('/predict_vgg', methods=['POST'])
def predict_vgg():
    try:
        # Get the audio data from the request
        data = request.json.get('wav_music')
        if not data:
            return jsonify({'error': 'Aucun fichier audio fourni'}), 400
        
        # Decode the Base64 audio data
        audio_data = base64.b64decode(data)
        
        # Call the predict_genre function with the decoded audio data
        predicted_genre = predict_genre(audio_data, vgg_model)
        
        return jsonify({'genre': predicted_genre})
    
    except AudioProcessingError:
        return jsonify({'error': 'Erreur de traitement du fichier audio'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)





# # app.py
# from flask import Flask, request, jsonify
# import base64
# import numpy as np
# # from tensorflow.keras.models import load_model
# from io import BytesIO
# import librosa

# app = Flask(__name__)

# # Charger le modèle VGG19
# vgg19_model = load_model("model/vgg.pkl")

# class AudioProcessingError(Exception):
#     pass

# def preprocess_audio(audio_data):
#     try:
#         y, sr = librosa.load(BytesIO(audio_data), sr=22050)
#         features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
#         return np.array([features])
#     except Exception:
#         raise AudioProcessingError("Erreur de traitement audio")

# @app.route('/predict_vgg19', methods=['POST'])
# def predict_vgg19():
#     try:
#         data = request.json.get('wav_music')
#         if not data:
#             return jsonify({'error': 'Aucun fichier audio fourni'}), 400

#         audio_data = base64.b64decode(data)
#         features = preprocess_audio(audio_data)
#         prediction = vgg19_model.predict(features)
#         genre = np.argmax(prediction)
#         return jsonify({'genre': int(genre)})
#     except AudioProcessingError:
#         return jsonify({'error': 'Erreur de traitement du fichier audio'}), 500
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5002)
