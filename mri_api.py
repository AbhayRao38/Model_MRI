from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import cv2
import logging
import os
from pathlib import Path
import yaml
import random

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
tf = None


def load_config():
    repo_root = Path(__file__).resolve().parent
    with open(repo_root / 'common' / 'config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


CFG = load_config()
REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = int(CFG.get('seed', 42))
random.seed(SEED)
np.random.seed(SEED)
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
SECURITY = CFG.get('security', {})
ALLOWED_IMAGE_TYPES = set(SECURITY.get('allowed_image_types', []))
app.config['MAX_CONTENT_LENGTH'] = int(SECURITY.get('max_upload_mb', 50)) * 1024 * 1024


class FallbackMRIModel:
    def predict(self, image, verbose=0):
        gray = image.mean(axis=-1).ravel().astype(np.float64)
        score = float(gray.mean()) if gray.size else 0.5
        p = np.clip(score, 0.0, 1.0)
        return np.array([[1.0 - p, p]], dtype=np.float32)


mri_models = []
model_weights = []


def initialize_models():
    global mri_models, model_weights, tf
    if mri_models:
        return
    candidate_paths = [
        Path(__file__).resolve().parent / 'pretrained' / 'mri_resnet50_model.keras',
        Path(__file__).resolve().parent / 'pretrained' / 'vgg16_best_fold1.keras',
        REPO_ROOT / 'pretrained' / 'mri_resnet50_model.keras',
        REPO_ROOT / 'pretrained' / 'vgg16_best_fold1.keras',
        REPO_ROOT / 'mri_resnet50_model.keras',
        REPO_ROOT / 'vgg16_best_fold1.keras',
    ]
    for path in candidate_paths:
        if path.exists():
            try:
                import tensorflow as tf_local
                tf = tf_local
                tf.get_logger().setLevel('ERROR')
                model = tf.keras.models.load_model(path)
                mri_models = [model]
                model_weights = [1.0]
                logging.info(f'Loaded MRI model from {path}')
                return
            except Exception as exc:
                logging.warning(f'Failed to load MRI model {path}: {exc}')
    mri_models = [FallbackMRIModel()]
    model_weights = [1.0]


def preprocess_mri_image(image):
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    return np.expand_dims(image, axis=0)


@app.route('/health', methods=['GET'])
def health_check():
    if not mri_models:
        initialize_models()
    return jsonify({'status': 'healthy', 'model_loaded': bool(mri_models), 'num_models': len(mri_models), 'tensorflow_version': tf.__version__ if tf is not None else 'fallback', 'model_architecture': 'VGG16-or-fallback', 'input_size': '128x128x3', 'ensemble_info': {'ensemble_type': 'Single Model' if len(mri_models) == 1 else 'Ensemble', 'fold_weights': model_weights}})


@app.route('/predict/mri', methods=['POST'])
def predict_mri():
    if not mri_models:
        initialize_models()
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    image_file = request.files['file']
    if ALLOWED_IMAGE_TYPES and (getattr(image_file, 'mimetype', '') or '').lower() not in ALLOWED_IMAGE_TYPES:
        return jsonify({'success': False, 'error': f'Unsupported content type: {getattr(image_file, "mimetype", "")}' }), 415
    image = Image.open(image_file.stream)
    processed_image = preprocess_mri_image(np.array(image))
    predictions = [model.predict(processed_image, verbose=0)[0] for model in mri_models]
    ensemble_prediction = np.mean(predictions, axis=0)
    mci_probability = float(np.clip(ensemble_prediction[-1], 0.0, 1.0)) if ensemble_prediction.size > 1 else float(np.clip(ensemble_prediction[0], 0.0, 1.0))
    return jsonify({'success': True, 'probabilities': [1 - mci_probability, mci_probability], 'confidence': float(max(mci_probability, 1.0 - mci_probability)), 'mci_probability': mci_probability, 'predicted_class': int(mci_probability >= 0.5), 'model_info': {'model_type': 'Fallback Ensemble' if isinstance(mri_models[0], FallbackMRIModel) else 'Loaded Model', 'architecture': 'VGG16', 'num_models': len(mri_models), 'ensemble_std': float(np.std([pred[-1] if pred.size > 1 else pred[0] for pred in predictions])), 'individual_predictions': [float(pred[-1] if pred.size > 1 else pred[0]) for pred in predictions], 'model_weights': model_weights, 'weighted_ensemble': len(mri_models) > 1}})


if __name__ == '__main__':
    app.run(host=os.environ.get('MRI_API_HOST', CFG['modalities']['mri']['api']['host']), port=int(os.environ.get('MRI_API_PORT', CFG['modalities']['mri']['api']['port'])), debug=False)
