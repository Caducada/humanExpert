from flask import Flask, render_template, request, redirect, url_for
from important_scripts import confidence, generate_image
from werkzeug.utils import secure_filename
import os



app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    process = request.form.get('processFile')
    if process == 'True':
        predictor = confidence.ConfidencePredictor()
        generator = generate_image.SpectrogramGenerator()
        file = request.files['file']
        grayscale = generator.filestorage_to_grayscale_spectrogram(file)
        predictor.predict_confidence(grayscale)
        report = predictor.print_confidence_report(predictor.results)
        return render_template('index.html', report=report)
    return render_template('index.html')


if __name__ == '__main__':
    while True:
        app.run(debug=True)
