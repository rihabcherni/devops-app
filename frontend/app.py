from flask import Flask, render_template, request, jsonify
import base64
import requests

app = Flask(__name__)

# Define the URLs for your SVM and VGG19 services
SVM_SERVICE_URL = "http://svm_service:5001/predict_svm"
VGG19_SERVICE_URL = "http://vgg19_service:5002/predict_vgg"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_svm', methods=['POST'])
def classify_svm():
    # Handle SVM classification
    wav_file = request.files.get('wav_file')

    if not wav_file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Convert file to base64
        audio = wav_file.read()
        base64_audio = base64.b64encode(audio).decode('utf-8')

        # Send to SVM service
        response = requests.post(SVM_SERVICE_URL, json={"wav_music": base64_audio})

        if response.status_code == 200:
            svm_result = response.json()
            return jsonify({"svm_genre": svm_result.get("genre", "Unknown")})

        return jsonify({"error": f"SVM service error: {response.status_code}"}), 400

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/classify_vgg19', methods=['POST'])
def classify_vgg19():
    # Handle VGG19 classification
    wav_file = request.files.get('wav_file')

    if not wav_file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Convert file to base64
        audio = wav_file.read()
        base64_audio = base64.b64encode(audio).decode('utf-8')

        # Send to VGG19 service
        response = requests.post(VGG19_SERVICE_URL, json={"wav_music": base64_audio})

        if response.status_code == 200:
            vgg19_result = response.json()
            return jsonify({"vgg19_genre": vgg19_result.get("genre", "Unknown")})

        return jsonify({"error": f"VGG19 service error: {response.status_code}"}), 400

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
