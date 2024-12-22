import librosa

def preprocess_audio(audio_path, target_sample_rate=16000):
    """
    Preprocess audio for Whisper. Truncates audio longer than max_duration.
    
    Parameters:
        audio_path (str): Path to the audio file.
        target_sample_rate (int): Sampling rate for the model.
    
    Returns:
        array: Preprocessed audio as a array.
    """
    audio, sr = librosa.load(audio_path, sr=target_sample_rate)
    print(audio)
    print(type(audio))
    return audio

preprocess_audio("recod.wav")
