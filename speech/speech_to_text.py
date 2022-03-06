from .klaam import SpeechRecognition, TextToSpeech
import librosa
import torch

def get_arabic_model():
    return SpeechRecognition()

def arabic_text(model):
    t = model.transcribe('test.wav')
    return t


def english_text(processor, model):
    speech_array, sampling_rate = librosa.load('test.wav', sr=16_000)
    inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    print(predicted_sentences)
    return predicted_sentences


# model = TextToSpeech()
# model.synthesize("وَسادْيُو مَانِي فِي الدَّقِيقَتَيْنِ السّابِعَةَ عَشْرَةَ - وَ الْخَامِسَةَ وَالْأَرْبعينَ مِنَ الشَّوْطِ الْأَوَّلِ لِلْمُبَارَاةِ")
# Audio("sample.wav")