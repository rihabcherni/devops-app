from flask import Flask, render_template, request, jsonify
import base64
import requests
import io
from pydub import AudioSegment

app = Flask(__name__)

# Define the URLs for your SVM and VGG19 services
SVM_SERVICE_URL = "http://svm_service:5001/classify"
VGG19_SERVICE_URL = "http://vgg19_service:5002/classify" 

@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded WAV file from the form
    wav_file = request.files['wav_file']
    
    if wav_file:
        # Convert the WAV file to base64
        audio = wav_file.read()
        base64_audio = base64.b64encode(audio).decode('utf-8')
        
        # Send the audio to the SVM service for classification
        response_svm = requests.post(SVM_SERVICE_URL, json={"wav_music": base64_audio})
        
        # Send the audio to the VGG19 service for classification
        response_vgg19 = requests.post(VGG19_SERVICE_URL, json={"wav_music": base64_audio})
        
        if response_svm.status_code == 200 and response_vgg19.status_code == 200:
            svm_result = response_svm.json()
            vgg19_result = response_vgg19.json()
            return jsonify({
                "svm_genre": svm_result.get("genre"),
                "vgg19_genre": vgg19_result.get("genre")
            })
        else:
            return jsonify({"error": "Error classifying the music"}), 400
    return jsonify({"error": "No file uploaded"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
