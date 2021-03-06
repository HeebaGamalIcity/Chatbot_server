import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from nltk.corpus import words
from nltk.metrics.distance import edit_distance

correct_words = words.words()


lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence):
    print('test')
    print(sentence)
    print('test')
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model, words, classes):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

finail_resp = { "سؤال_الرصيد": "رصيد حسابك الحالي 10 ألاف جنيه",
                "question_credit" : "your account contain 10 thousand Egyptian pound",
                "شكوي توقف":"لقد تم إعادة تفعيل الكارت الخاص بك",
                "complain_stopped" : "Your card has been reactivated",
                "تفعيل": "لقد تم تفعيل الكارت الخاص بك",
                "activation_card":"Your card has been activated"}

def get_response(sentence, model, intents, words, classes, reversed_tag, final_tag, T):
    tag = predict_class(sentence, model, words, classes)[0]['intent']
    for key in intents:
        for intent in intents[key]:
            if final_tag:
                return finail_resp[reversed_tag], " ", False

            if intent['tag'] == tag and T == False:
                return random.choice(intent['responses']), intent['tag'] , (intent['tag']  in finail_resp)
            elif intent['tag'] == tag and T == True:
                return random.choice(intent['responses_T']), intent['tag'] , (intent['tag']  in finail_resp)
