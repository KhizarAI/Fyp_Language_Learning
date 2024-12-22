import torch

def transcribe_audio(audio, processor, model):
    """
    Transcribe audio and return the transcription with timestamps.
    
    Parameters:
        audio_tensor (torch.Tensor): Preprocessed audio tensor.
        processor (WhisperProcessor): Loaded processor.
        model (WhisperForConditionalGeneration): Loaded Whisper model.
    
    Returns:
        String: Transcription text
    """
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], return_dict_in_generate=True, output_scores=False, attention_mask=1)
    
    # Decode transcription
    transcription = processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]

    return transcription