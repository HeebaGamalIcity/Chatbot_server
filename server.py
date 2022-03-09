import nltk
# nltk.download('words')
# nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
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


reversed_tag = ""
ref_tag = False

finail_resp = { "سؤال_الرصيد": "رصيد حسابك الحالي 10 ألاف جنيه",
                "question_credit" : "your account contain 10 thousand Egyptian pound",
                "شكوي توقف":"لقد تم إعادة تفعيل الكارت الخاص بك",
                "complain_stopped" : "Your card has been reactivated",
                "تفعيل": "لقد تم تفعيل الكارت الخاص بك",
                "activation_card":"Your card has been activated"}



@app.route('/files/<path:path>')
def send_file(path):
    return send_from_directory('', path)


@app.route("/speech", methods=["POST"])
def get_speech():
    file = request.files['file']
    lang = request.headers['lang']
    file.save('demo.wav')
    global reversed_tag, ref_tag


    if lang == "ar":
        trans = arabic_text(model_sr)

        bot_response, reversed_tag, ref_tag = get_response(trans, model=ar_chatbot_model, intents=ar_intents, words=ar_words,
                                    classes=ar_classes  , reversed_tag=reversed_tag, final_tag=ref_tag, T = True)
        model_tts_ar.synthesize(bot_response)#"أَسَفٌ لَمْ أَسْتَطِيعَ فَهْمُكَ")


    elif lang == "en":
        trans = english_text(model=model_sr_en, processor=processor_sr_en)
        bot_response, reversed_tag, ref_tag = get_response(trans, model=en_chatbot_model, intents=en_intents, words=en_words,
                                    classes=en_classes , reversed_tag=reversed_tag, final_tag=ref_tag,T = False)
        waveforms = english_speech(device, processor, tacotron2, vocoder, bot_response)
        torchaudio.save("demo.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)

    return jsonify({"message":"http://209.51.170.248:5000/files/sample.wav"})


@app.route("/text", methods=["POST"])
def get_text():
    global reversed_tag, ref_tag
    text = request.form.get('text')
    lang = request.headers['lang']
    bot_response = ""
    print(reversed_tag)
    if lang == 'ar':
        bot_response, reversed_tag, ref_tag = get_response(text, model=ar_chatbot_model, intents=ar_intents, words=ar_words,
                                                           classes=ar_classes, reversed_tag=reversed_tag, final_tag=ref_tag, T = False)
    if lang == "en":
        bot_response, reversed_tag, ref_tag  = get_response(text, model=en_chatbot_model, intents=en_intents, words=en_words,
                                    classes=en_classes , reversed_tag=reversed_tag, final_tag=ref_tag, T = False)
    return jsonify({"message":bot_response})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
