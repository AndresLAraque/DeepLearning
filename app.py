from flask import Flask, render_template, request, send_from_directory
import os
import torch 
from PIL import Image, ImageDraw

app = Flask(__name__)

base_dir = '/home/andres/Documents/Flask/ProyectoFInal/'
uploads_dir = 'static/uploads/'
outputs_dir = 'static/outputs/'

# Configurar los directorios
# os.makedirs(uploads_dir, exist_ok=True)
# os.makedirs(outputs_dir, exist_ok=True)

# Ruta al modelo entrenado (asegúrate de que la ruta sea correcta)
model_path = base_dir + "yolo/best.pt"
# Ruta imagen de prueba
path_to_image = base_dir + uploads_dir + 'test.jpg'
# Cargar el modelo
model = torch.hub.load("ultralytics/yolov5:master", "custom", path=model_path)

results = model(path_to_image)

original_image = Image.open(path_to_image)

# Crear un objeto ImageDraw para su
draw = ImageDraw.Draw(original_image)

predictions = results.xyxy[0].cpu().numpy()  # Convertir a array de NumPy
for pred in predictions:
    box = pred[:4]  # Coordenadas del cuadro delimitador (x_min, y_min, x_max, y_max)
    label = int(pred[5])  # Etiqueta de la clase (convertir a entero)

    # Superponer el cuadro delimitador en la imagen
    draw.rectangle(box, outline='red', width=10)
    draw.text((box[0], box[1]), f'Class {label}', fill='red')

# Guardar la imagen generada por el modelo
output_filename = 'output_test.jpg'
output_path = base_dir + outputs_dir + output_filename
original_image.save(output_path)

@app.route('/')
def index():
    return render_template('index.html', output_filename=output_filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No se seleccionó ningún archivo')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No se seleccionó ningún archivo')

    # Guardar el archivo en la carpeta 'uploads'
    file.save(uploads_dir + file.filename)

    # Devolver la ruta de la imagen cargada
    return render_template('index.html', filename=file.filename, output_filename=output_filename)

# Agregar una ruta para acceder a la imagen generada por el modelo
@app.route('/output_image')
def get_output():
    return send_from_directory(outputs_dir, output_filename)

if __name__ == '__main__':
    app.run(debug=True)