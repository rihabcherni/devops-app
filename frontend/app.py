from flask import Flask, render_template, request, jsonify
import base64
import requests

app = Flask(__name__)

# Define the URLs for your SVM and VGG19 services
SVM_SERVICE_URL = "http://svm_service:5001/predict_svm"
VGG19_SERVICE_URL = "http://svm_service:5001/predict_svm"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded WAV file from the form
    wav_file = request.files.get('wav_file')

    if not wav_file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Read the file and convert it to base64
        audio = wav_file.read()
        base64_audio = base64.b64encode(audio).decode('utf-8')

        # Send the audio to the SVM service for classification
        response_svm = requests.post(SVM_SERVICE_URL, json={"wav_music": base64_audio})
        response_vgg19 = requests.post(VGG19_SERVICE_URL, json={"wav_music": base64_audio})

        # Check if both responses are valid
        if response_svm.status_code == 200 and response_vgg19.status_code == 200:
            svm_result = response_svm.json()
            vgg19_result = response_vgg19.json()
            return jsonify({
                "svm_genre": svm_result.get("genre", "Unknown"),
                "vgg19_genre": vgg19_result.get("genre", "Unknown")
            })

        # Handle HTTP errors from the services
        error_message = f"SVM status: {response_svm.status_code}, VGG19 status: {response_vgg19.status_code}"
        return jsonify({"error": f"Error classifying the music. {error_message}"}), 400

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
