import nltk
nltk.download('words')
nltk.download('punkt')
from flask import Flask, request, jsonify, send_from_directory, send_file
from speech.speech_to_text import arabic_text, english_text, get_arabic_model
from speech.text_to_speech import english_speech, get_ts_model
# from IPython.display import Audio
# from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
# import matplotlib.pyplot as plt
from keras.models import load_model
import json
from bot import get_response
import pickle


# import IPython
app = Flask(__name__)


ar_chatbot_model = load_model('ar_chatbot_model.h5')
en_chatbot_model = load_model('en_chatbot_model.h5')
en_intents = json.loads(open('intent.json').read())
ar_intents = json.loads(open('ar_intent.json', encoding="utf8").read())
ar_words = pickle.load(open('ar_words.pkl','rb'))
ar_classes = pickle.load(open('ar_classes.pkl','rb'))
en_words = pickle.load(open('en_words.pkl','rb'))
en_classes = pickle.load(open('en_classes.pkl','rb'))
# #speech english
processor_sr_en = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
model_sr_en = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
#
# #speech ar
model_sr = get_arabic_model()
#
# #text
model_tts_ar= get_ts_model()
#
# #text
device = "cuda" if torch.cuda.is_available() else "cpu"

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)


# @app.route("/speech", methods=["POST"])
# def get_speech():
#     speech_arr = request.form.get('array')
#     speech_arr = np.array(speech_arr)
#     print(type(speech_arr))
#     print(speech_arr)
#     data, samplerate = sf.read('piano.wav')  # mono signal
#     sf.write(data, 'new_file.ogg', samplerate=samplerate)
#     sf.write('sound.wav', speech_arr, 16000)
#
#     # samplerate = 16000
#     # write("example.wav", samplerate, speech_arr)
#
#     lang = request.headers['lang']
#
#
#     # transcript = model_sr.transcribe('test.wav')
#     # print(transcript)
#     # file.save('test.wav')
#     # return send_from_directory('', 'test.wav', as_attachment=True)
#     return jsonify({"message": "doner"})



@app.route('/files/<path:path>')
def send_file(path):
    return send_from_directory('', path)

@app.route("/speech", methods=["POST"])
def get_speech():
    file = request.files['file']
    lang = request.headers['lang']
    file.save('demo.wav')
    if lang == "ar":
        trans = arabic_text(model_sr)
        bot_response = get_response(trans, model=ar_chatbot_model, intents=ar_intents, words=ar_words,
                                    classes=ar_classes)
        model_tts_ar.synthesize(bot_response)


    elif lang == "en":
        trans = english_text()
        bot_response = get_response(trans, model=en_chatbot_model, intents=en_intents, words=en_words,
                                    classes=en_classes)
        waveforms = english_speech(device, processor, tacotron2, vocoder, bot_response)
        torchaudio.save("demo.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)

    return jsonify({"message":"http://192.168.1.20:5000/files/demo.wav"})


@app.route("/text", methods=["POST"])
def get_text():
    text = request.form.get('text')
    lang = request.headers['lang']
    bot_response = get_response(text, model=ar_chatbot_model, intents=ar_intents, words=ar_words, classes=ar_classes)
    if lang == "en":
        bot_response = get_response(text, model=en_chatbot_model, intents=en_intents, words=en_words, classes=en_classes)
    return jsonify({"message":bot_response})

# #beeb beeb
# @app.route("/speech_en", methods=["POST"])
# def get_speech_en():
#     file = request.files['file']
#     file.save('./test.wav')
#
#     speech_array, sampling_rate = librosa.load('./test.wav', sr=16_000)
#     inputs = processor_sr_en(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         logits = model_sr_en(inputs.input_values, attention_mask=inputs.attention_mask).logits
#     predicted_ids = torch.argmax(logits, dim=-1)
#     predicted_sentences = processor_sr_en.batch_decode(predicted_ids)
#
#     print(predicted_sentences)
#     return jsonify({"message":predicted_sentences})
#
# @app.route("/tts_ar", methods=["POST"])
# def get_text_ar():
#     text = request.form.get('data')
#     print(text)
#     model_tts_ar.synthesize(text)
#
#     return jsonify({"message":"done"})
#
#
# @app.route("/tts_en", methods=["POST"])
# def get_text_en():
#     file = request.form.get('data')
#
#     with torch.inference_mode():
#         processed, lengths = processor(file)
#         processed = processed.to(device)
#         lengths = lengths.to(device)
#         spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
#         waveforms, lengths = vocoder(spec, spec_lengths)
#
#     torchaudio.save("output_wavernn.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)
#     IPython.display.display(IPython.display.Audio("output_wavernn.wav"))
#
#     return jsonify({"message":'done'})
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')