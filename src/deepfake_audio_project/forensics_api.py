import requests


class DeepfakeForensicsIntegration:
    def __init__(self, elevenlabs_key=None, resemble_key=None):
        self.elevenlabs_api_key = elevenlabs_key
        self.resemble_api_key = resemble_key
        self.apis_available = elevenlabs_key is not None or resemble_key is not None

    def check_elevenlabs_classifier(self, audio_file_path):
        if not self.elevenlabs_api_key:
            return None
        url = "https://api.elevenlabs.io/v1/audio-native"
        headers = {"xi-api-key": self.elevenlabs_api_key}
        try:
            with open(audio_file_path, "rb") as file_obj:
                response = requests.post(url, headers=headers, files={"audio": file_obj}, timeout=10)
                result = response.json()
            return {
                "is_ai_generated": result.get("is_likely_ai_generated"),
                "confidence": result.get("probability", 0.0),
                "source": "elevenlabs",
            }
        except Exception as exc:
            print(f"ElevenLabs API Error: {exc}")
            return None

    def check_resemble_ai(self, audio_file_path):
        if not self.resemble_api_key:
            return None
        url = "https://api.resemble.ai/v2/detect"
        headers = {"Authorization": f"Bearer {self.resemble_api_key}"}
        try:
            with open(audio_file_path, "rb") as file_obj:
                response = requests.post(url, headers=headers, files={"audio": file_obj}, timeout=10)
                result = response.json()
            return {
                "is_synthetic": result.get("is_synthetic"),
                "confidence": result.get("confidence_score", 0.0),
                "source": "resemble_ai",
            }
        except Exception as exc:
            print(f"Resemble AI API Error: {exc}")
            return None

    def ensemble_with_external_apis(self, audio_file_path, model_prediction, model_confidence):
        if not self.apis_available:
            return {
                "ensemble_prediction": model_prediction,
                "ensemble_confidence": model_confidence,
                "your_model": {"prediction": model_prediction, "confidence": model_confidence},
                "external_apis": "Not configured",
            }

        elevenlabs_result = self.check_elevenlabs_classifier(audio_file_path)
        resemble_result = self.check_resemble_ai(audio_file_path)

        predictions = [model_confidence]
        weights = [0.6]
        if elevenlabs_result:
            predictions.append(elevenlabs_result["confidence"])
            weights.append(0.2)
        if resemble_result:
            predictions.append(resemble_result["confidence"])
            weights.append(0.2)

        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
        ensemble_score = sum(p * w for p, w in zip(predictions, norm_weights))

        return {
            "ensemble_prediction": "fake" if ensemble_score > 0.5 else "real",
            "ensemble_confidence": ensemble_score,
            "your_model": {"prediction": model_prediction, "confidence": model_confidence},
            "elevenlabs": elevenlabs_result,
            "resemble_ai": resemble_result,
        }

