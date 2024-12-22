from flask import Flask, request, jsonify, render_template
from audio_preprocessor import preprocess_audio
from model_loader import WhisperModel
from transcription_service import transcribe_audio
import io

# Initialize Flask app
app = Flask(__name__)

# Initialize the model and processor once at startup
whisper_model = WhisperModel()
processor = whisper_model.get_processor()
model = whisper_model.get_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files['audio']

    try:
        audio_bytes = audio_file.read()
        audio_data = preprocess_audio(io.BytesIO(audio_bytes))

        # Transcribe the audio
        transcription = transcribe_audio(audio_data, processor, model)

        # Render the transcription on the same page
        return render_template('index.html', Transcribe=transcription)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred while processing the audio."}), 500

if __name__ == '__main__':
    app.run(debug=True)
