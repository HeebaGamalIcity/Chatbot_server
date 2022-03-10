import nltk
# nltk.download('words')
# nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from flask import Flask, request, jsonify, send_from_directory, send_file
from speech.speech_to_text import arabic_text, english_text, get_arabic_model
from speech.text_to_speech import english_speech, get_ts_model
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
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

# bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
# processor = bundle.get_text_processor()
# tacotron2 = bundle.get_tacotron2().to(device)
# vocoder = bundle.get_vocoder().to(device)


lang = 'English'
tag = 'kan-bayashi/ljspeech_vits'
vocoder_tag = "none"
text2speech = Text2Speech.from_pretrained(
    model_tag=str_or_none(tag),
    vocoder_tag=str_or_none(vocoder_tag),
    device="cpu",
    threshold=0.5,
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    speed_control_alpha=1.0,
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

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
    file.save('test.wav')
    global reversed_tag, ref_tag


    if lang == "ar":
        trans = arabic_text(model_sr)

        bot_response, reversed_tag, ref_tag = get_response(trans, model=ar_chatbot_model, intents=ar_intents, words=ar_words,
                                    classes=ar_classes  , reversed_tag=reversed_tag, final_tag=ref_tag, T = True)
        model_tts_ar.synthesize(bot_response)#"أَسَفٌ لَمْ أَسْتَطِيعَ فَهْمُكَ")


    elif lang == "en":
        trans = english_text(model=model_sr_en, processor=processor_sr_en)[0]
        bot_response, reversed_tag, ref_tag = get_response(trans, model=en_chatbot_model, intents=en_intents, words=en_words,
                                    classes=en_classes , reversed_tag=reversed_tag, final_tag=ref_tag,T = False)
        #waveforms = english_speech(device, processor, tacotron2, vocoder, bot_response)
        with torch.no_grad():
                wav = text2speech(bot_response)["wav"]

        torchaudio.save("sample.wav", wav.unsqueeze(0).cpu(), sample_rate=19000)

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
