import io
import cv2
import librosa
import numpy as np
from werkzeug.datastructures import FileStorage

class SpectrogramGenerator:

    def filestorage_to_grayscale_spectrogram(
        self,
        file: FileStorage,
        target_duration: float = 30.0,
        n_mels: int = 256,
        n_fft: int = 4096,
        hop_length: int = 256
    ):

        audio_bytes = file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        try:
            y, sr = librosa.load(audio_buffer, sr=None, mono=True)
        
        except Exception as e:
            return "error"  

        target_samples = int(target_duration * sr)

        if len(y) > target_samples:
            y = y[:target_samples]
        else:
            repeats = int(np.ceil(target_samples / len(y)))
            y = np.tile(y, repeats)[:target_samples]

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=sr // 2,
            power=2.0
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        grayscale = (mel_norm * 255).astype(np.uint8)

        return cv2.resize(grayscale, (775, 308))
