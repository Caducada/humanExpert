import os
from werkzeug.utils import secure_filename
from important_scripts import confidence, generate_image, convert_hub
from flask import Flask, render_template, request, redirect, url_for



app = Flask(__name__, static_folder='static', static_url_path='/static')

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}
MAX_FILE_SIZE = 50 * 1024 * 1024  

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/info', methods=['GET', 'POST'])
def info():
    return render_template('info.html', page="info", category="null")


@app.route('/', methods=['GET', 'POST'])
def index():
    process = request.form.get('processFile')
    if process == 'True':
        predictor = confidence.ConfidencePredictor()
        generator = generate_image.SpectrogramGenerator()
        file = request.files['file']
        if file.filename.endswith('.m4a'):
            file = convert_hub.converter.convert_m4a_to_wav(file)
        grayscale = generator.filestorage_to_grayscale_spectrogram(file)
        if type(grayscale) == str:
            return render_template('index.html', page="index", category="error")
        predictor.predict_confidence(grayscale)
        predictor.get_top_result(predictor.results)
        return render_template('index.html', page="index", category=predictor.get_top_result(predictor.results))
    return render_template('index.html', page="index", category="null")


if __name__ == '__main__':
    app.run(debug=True)
