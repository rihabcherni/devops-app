import unittest
import base64
import json
from app import app
from io import BytesIO
import xmlrunner


class TestVGGService(unittest.TestCase): 
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_predict_vgg_valid_audio(self):
        # Charger un fichier audio de test en base64
        with open("tests/test_audio.wav", "rb") as f:
            audio_data = f.read()
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        
        # Envoyer une requête POST avec le fichier audio
        response = self.client.post(
            '/predict_vgg',
            data=json.dumps({'wav_music': base64_audio}),
            content_type='application/json'
        )
        
        # Vérifier la réponse
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('genre', data)  # Vérifier si le genre est dans la réponse
        self.assertIn(data['genre'], ["blues", "classical", "country", "disco", 
                                      "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
    
    def test_predict_vgg_no_audio(self):
        # Tester l'absence de fichier audio
        response = self.client.post(
            '/predict_vgg',
            data=json.dumps({}),  # Pas de fichier audio fourni
            content_type='application/json'
        )
        
        # Vérifier la réponse
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Aucun fichier audio fourni')

    def test_predict_vgg_invalid_audio(self):
        # Envoyer des données non valides
        response = self.client.post(
            '/predict_vgg',
            data=json.dumps({'wav_music': 'invalid_base64'}),
            content_type='application/json'
        )
        
        # Vérifier la réponse
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_vgg_service_error_handling(self):
        # Simuler un cas où une exception interne se produit
        with app.test_request_context('/predict_vgg', method='POST'):
            response = self.client.post(
                '/predict_vgg',
                data=json.dumps({'wav_music': base64.b64encode(b"invalid data").decode()}),
                content_type='application/json'
            )
            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)
            self.assertIn('error', data)

if __name__ == "__main__":
    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output='/reports'),
        failfast=False,
        buffer=False,
        catchbreak=False,
    )


