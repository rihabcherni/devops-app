# app.py
from flask import Flask, request, jsonify
import base64
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import librosa

app = Flask(__name__)

with open("model/vgg.pkl", "rb") as f:
    vgg_model = pickle.load(f)

scaler = StandardScaler()

# Classe pour l'erreur de traitement
class AudioProcessingError(Exception):
    pass 

def preprocess_audio(audio_data):
    try:
        y, sr = librosa.load(BytesIO(audio_data), sr=None)
        features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        return scaler.transform([features])
    except Exception as e:
        raise AudioProcessingError("Erreur de traitement audio")

@app.route('/predict_vgg', methods=['POST'])
def predict_vgg():
    try:
        data = request.json.get('wav_music')
        if not data:
            return jsonify({'error': 'Aucun fichier audio fourni'}), 400
        
        audio_data = base64.b64decode(data)
        features = preprocess_audio(audio_data)
        prediction = vgg_model.predict(features)
        return jsonify({'genre': prediction[0]})
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
