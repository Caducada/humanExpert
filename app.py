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


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return f'File {filename} uploaded successfully!'
    
    return 'Invalid file. Please upload an MP3 file.'


if __name__ == '__main__':
    app.run(debug=True)
