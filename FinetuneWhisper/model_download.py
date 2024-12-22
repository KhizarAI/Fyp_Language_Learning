from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "khizarAI/finetune-whisper-base.en"  # Replace with your desired model name
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

processor.save_pretrained("./processor")
model.save_pretrained("./model")
