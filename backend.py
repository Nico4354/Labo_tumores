import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='frontend-angular/dist/frontend-angular/browser')
CORS(app)

# 1. Manejo de errores al cargar el modelo
try:
    interpreter = tf.lite.Interpreter(model_path='brain_tumor_cnn.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error crítico: No se pudo cargar el modelo TFLite. Detalles: {e}")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Carpeta de uploads
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 2. Validación de extensiones permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_with_tflite(img_array):
    img_array = img_array.astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]


@app.route('/api/clasificar', methods=['POST'])
def clasificar_api():
    # 3. Validaciones más robustas para el archivo recibido
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró la imagen en la petición'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Tipo de archivo no permitido. Usa JPG o PNG.'}), 400

    try:
        # 4. Nombres únicos para evitar colisiones (concurrencia)
        filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex
        unique_filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Procesar imagen
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        prediction = predict_with_tflite(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        probabilities = {class_names[i]: float(f"{prob:.4f}") for i, prob in enumerate(prediction)}

        # 5. Gráfico con nombre único
        plt.figure(figsize=(6, 4))
        plt.bar(probabilities.keys(), probabilities.values(), color='skyblue')
        plt.title('Probabilidades por clase')
        plt.ylabel('Confianza')
        plt.tight_layout()

        graph_filename = f"probabilidades_{unique_id}.png"
        graph_path = os.path.join(app.config['UPLOAD_FOLDER'], graph_filename)
        plt.savefig(graph_path)
        plt.close()

        return jsonify({
            'prediction': f'Predicción: {predicted_class.upper()}',
            'image_name': unique_filename,
            'graph_name': graph_filename,
            'probs': probabilities
        })
        
    except Exception as e:
        # 6. Captura de errores de procesamiento
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500


# 7. Ruta explícita para servir los archivos subidos
@app.route('/uploads/<filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_angular(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)