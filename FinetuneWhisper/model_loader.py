from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperModel:
    def __init__(self, model_path="./FinetuneWhisper"):
        """
        Initialize the Whisper model and processor.
        
        Parameters:
            model_path (str): Path to the locally saved model.
        """
        print("Loading Whisper model and processor...")
        self.processor = WhisperProcessor.from_pretrained("processor")
        self.model = WhisperForConditionalGeneration.from_pretrained("model")
        print("Model and processor loaded successfully.")
    
    def get_processor(self):
        return self.processor
    
    def get_model(self):
        return self.model
