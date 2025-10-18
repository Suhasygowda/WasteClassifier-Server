# ============================================================================
# WASTE CLASSIFIER - FLASK API WITH CORS FOR FRONTEND CONNECTION
# ============================================================================

# Install: pip install flask flask-cors tensorflow opencv-python pillow

from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import cv2
import numpy as np
from PIL import Image
import json
import os
from werkzeug.utils import secure_filename
import traceback
from datetime import datetime

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MODEL_PATH = 'waste_classifier.keras'
CLASS_MAPPING_PATH = 'class_mapping.json'
IMG_SIZE = (224, 224)

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

print("\n" + "="*70)
print("🚀 WASTE CLASSIFIER API - INITIALIZATION")
print("="*70)

# ============================================================================
# LOAD MODEL AND CLASS MAPPING
# ============================================================================

model = None
class_mapping = None

try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found: {MODEL_PATH}")
    print("   Please ensure waste_classifier.keras exists in the project directory")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")

try:
    with open(CLASS_MAPPING_PATH, 'r') as f:
        class_mapping = json.load(f)
    print(f"✅ Class mapping loaded: {list(class_mapping.values())}")
except FileNotFoundError:
    print(f"⚠️  Class mapping file not found: {CLASS_MAPPING_PATH}")
    class_mapping = {
        "0": "cardboard",
        "1": "compost",
        "2": "glass",
        "3": "metal",
        "4": "paper",
        "5": "plastic",
        "6": "trash"
    }
    print(f"   Using default mapping: {list(class_mapping.values())}")
except Exception as e:
    print(f"⚠️  Error loading class mapping: {str(e)}")

print("="*70 + "\n")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(image_path):
    """
    Load image and convert to RGB if necessary
    Handles grayscale, RGBA, and RGB images
    """
    try:
        # Read image using PIL for better color handling
        img = Image.open(image_path)
        
        # Convert to RGB (handles grayscale, RGBA, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise Exception(f"Image preprocessing error: {str(e)}")

def predict_waste_class(image_path):
    """
    Predict waste class from image
    Returns class name, confidence, and all probabilities
    """
    try:
        if model is None:
            raise Exception("Model not loaded. Ensure waste_classifier.keras exists.")
        
        # Preprocess image
        img_array = load_and_preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        confidence_scores = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(confidence_scores)
        confidence = float(confidence_scores[predicted_class_idx])
        
        # Get class name
        predicted_class = class_mapping.get(str(predicted_class_idx), f"Class_{predicted_class_idx}")
        
        # Get all probabilities
        all_predictions = {
            class_mapping.get(str(i), f"Class_{i}"): float(confidence_scores[i])
            for i in range(len(confidence_scores))
        }
        
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "all_predictions": {k: round(v, 4) for k, v in all_predictions.items()},
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API documentation"""
    return jsonify({
        "app": "WasteClassifier API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "GET /info": "Model information",
            "POST /predict": "Predict from file upload",
            "POST /predict-url": "Predict from URL",
            "POST /batch-predict": "Batch predictions"
        },
        "model_status": "loaded" if model is not None else "not_loaded",
        "supported_classes": list(class_mapping.values()) if class_mapping else []
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "api_online": True,
        "model_loaded": model is not None,
        "supported_classes": list(class_mapping.values()) if class_mapping else []
    }), 200

@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "status": "offline"
        }), 500
    
    return jsonify({
        "model_name": "Waste Classifier CNN",
        "version": "1.0.0",
        "input_shape": {
            "height": IMG_SIZE[0],
            "width": IMG_SIZE[1],
            "channels": 3
        },
        "num_classes": len(class_mapping),
        "classes": list(class_mapping.values()),
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": 16,
        "model_type": "Custom CNN (4 Convolutional Blocks)",
        "framework": "TensorFlow/Keras"
    }), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Predict waste class from uploaded image
    
    Expected: POST request with file upload (key: 'file')
    Returns: JSON with prediction result
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded. Please train and save the model first."
            }), 503
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided. Send multipart form data with 'file' field"
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_waste_class(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/predict-url', methods=['POST', 'OPTIONS'])
def predict_url():
    """
    Predict waste class from image URL
    
    Expected: JSON with 'url' field
    Returns: JSON with prediction result
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded"
            }), 503
        
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'url' field in JSON body"
            }), 400
        
        url = data['url']
        
        # Download image from URL
        import urllib.request
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_url_image.jpg')
        urllib.request.urlretrieve(url, filepath)
        
        # Make prediction
        result = predict_waste_class(filepath)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error processing URL: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/batch-predict', methods=['POST', 'OPTIONS'])
def batch_predict():
    """
    Batch prediction from multiple file uploads
    
    Expected: POST request with multiple files
    Returns: JSON array with predictions
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded"
            }), 503
        
        if 'files' not in request.files:
            return jsonify({
                "success": False,
                "error": "No files provided. Send 'files' field"
            }), 400
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                result = predict_waste_class(filepath)
                result['filename'] = filename
                results.append(result)
                
                try:
                    os.remove(filepath)
                except:
                    pass
        
        return jsonify({
            "success": True,
            "total_processed": len(results),
            "predictions": results
        }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Batch prediction error: {str(e)}"
        }), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "File too large (max 16MB)"
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

@app.before_request
def before_request():
    """Handle CORS preflight"""
    if request.method == 'OPTIONS':
        return ''

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n📝 API Configuration:")
    print(f"  Host: 0.0.0.0")
    print(f"  Port: 5000")
    print(f"  Debug: True")
    print(f"  CORS: Enabled for all origins")
    print(f"\n🌐 Access API at: http://localhost:5000")
    print(f"📊 Documentation: http://localhost:5000")
    print(f"\n⚠️  Make sure:")
    print(f"  - waste_classifier.keras exists in project folder")
    print(f"  - class_mapping.json exists in project folder")
    print(f"  - Flask server is running BEFORE using the frontend")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)