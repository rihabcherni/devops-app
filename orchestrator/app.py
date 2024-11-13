from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

SVM_URL = "http://svm_service:5001/predict_svm"
VGG19_URL = "http://vgg19_service:5002/predict_vgg19"

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json.get('wav_music')
    if not data:
        return jsonify({'error': 'Aucun fichier audio fourni'}), 400

    response = {}

    try:
        svm_response = requests.post(SVM_URL, json={'wav_music': data})
        svm_response.raise_for_status()
        response['svm_genre'] = svm_response.json().get('genre')
    except requests.exceptions.RequestException:
        response['svm_genre'] = 'Erreur de classification SVM'

    try:
        vgg19_response = requests.post(VGG19_URL, json={'wav_music': data})
        vgg19_response.raise_for_status()
        response['vgg19_genre'] = vgg19_response.json().get('genre')
    except requests.exceptions.RequestException:
        response['vgg19_genre'] = 'Erreur de classification VGG19'

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
