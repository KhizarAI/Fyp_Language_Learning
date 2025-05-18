from flask import Flask, request, jsonify, render_template
from audio_preprocessor import preprocess_audio
from model_loader import WhisperModel
from transcription_service import transcribe_audio
import io

app = Flask(__name__)

# Load Whisper model once at startup
whisper_model = WhisperModel()
processor = whisper_model.get_processor()
model = whisper_model.get_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'transcript': "No audio received."})

    audio_file = request.files['audio']

    try:
        audio_tensor = preprocess_audio(io.BytesIO(audio_file.read()))
        transcription = transcribe_audio(audio_tensor, processor, model)
        return jsonify({'transcript': transcription})

    except Exception as e:
        print("Error:", e)
        return jsonify({'transcript': "Error during transcription."})

if __name__ == '__main__':
    app.run(debug=True)
