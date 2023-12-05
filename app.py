from flask import Flask, render_template, request, send_from_directory
import os
import torch
from PIL import Image, ImageDraw
import uuid  # Added import for generating unique identifiers

app = Flask(__name__)

base_dir = '/home/andres/Documents/Flask/ProyectoFInal/'
uploads_dir = 'static/uploads/'
outputs_dir = 'static/outputs/'

# Ruta al modelo entrenado (asegúrate de que la ruta sea correcta)
model_path = base_dir + "yolo/best.pt"
# Cargar el modelo
model = torch.hub.load("ultralytics/yolov5:master", "custom", path=model_path)

def process_image(file_path):
    results = model(file_path)

    original_image = Image.open(file_path)
    draw = ImageDraw.Draw(original_image)

    predictions = results.xyxy[0].cpu().numpy()
    for pred in predictions:
        box = pred[:4]
        label = int(pred[5])
        draw.rectangle(box, outline='red', width=10)
        draw.text((box[0], box[1]), f'Class {label}', fill='red')

    unique_identifier = str(uuid.uuid4().hex)
    output_filename = f'{unique_identifier}_processed.jpg'
    output_path = os.path.join(base_dir, outputs_dir, output_filename)
    original_image.save(output_path)

    return output_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No se seleccionó ningún archivo')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No se seleccionó ningún archivo')

    # Guardar el archivo en la carpeta 'uploads'
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Procesar la imagen y obtener el nombre del archivo procesado
    output_filename = process_image(file_path)

    # Devolver la ruta de la imagen procesada
    return render_template('index.html', filename=file.filename, output_filename=output_filename)

# Agregar una ruta para acceder a la imagen procesada
@app.route('/output_image/<filename>')
def get_output(filename):
    return send_from_directory(outputs_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
