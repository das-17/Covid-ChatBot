#Creating GUI with tkinter
import tkinter
from tkinter import *
from PIL import Image, ImageTk
import nltk
from nltk.stem import WordNetLemmatizer
import gpt_2_simple as gpt2
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import json
import random

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    print(p)
    print(res)
    ERROR_THRESHOLD = 0.5
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list



def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text, gpt2, sess):
    ints = predict_class(text, model)
    print(ints)
    if len(ints) == 0:
        return "unrecognized"
    res = getResponse(ints, intents)
    print(res)
    return res

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(fg='#fdc700', font=("Verdana", 12 ))
        res = chatbot_response(msg, gpt2, sess)
        if res == "unrecognized":
            ChatLog.insert(END, "I didn't understand that, can you clarify?" + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
        if res == "generative":
            ChatLog.insert(END, "Bot: " + "I think you're asking about COVID symptoms. Let me consult my SlugDoc technology!" + '\n\n')
            pre = 'Patient:\n {} \n Doctor:\n '.format(msg)
            res_txt = gpt2.generate(sess,
                                    length=50,
                                    temperature=0.1,
                                    prefix=pre,
                                    nsamples=1,
                                    batch_size=1, 
                                    return_as_list=True
                                    )
            res_txt = res_txt[0].split('Doctor:\n ')[1].split('.')[0]
            ChatLog.insert(END, "Here's what SlugDoc thinks: " + res_txt + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
        else:
            ChatLog.insert(END, "Bot: " + res + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)

if __name__ == "__main__":
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('data/intents_v3.json', errors = 'ignore').read(), strict=False)
    words = pickle.load(open('artifacts/words_v3.pkl','rb'))
    classes = pickle.load(open('artifacts/classes_v3.pkl','rb'))

    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)

    model = load_model('chatbot_model_v3.h5')
    model.call = tf.function(model.call)

    base = Tk()
    base.title("SlugDoc COVID Information Bot")
    base.geometry("400x600")
    base.resizable(width=FALSE, height=FALSE)
    img = Image.open("healthslug.jpg")
    render = ImageTk.PhotoImage(img)
    logo = Label(base,image=render)
    #Create Chat window
    ChatLog = Text(base, bd=0,bg="#003c6c", height="6", width="50", font="Arial", wrap=WORD)
    ChatLog.config(state=DISABLED)
    #Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
    ChatLog['yscrollcommand'] = scrollbar.set
    #Create Button to send message
    SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height="5",
                        bd=0, bg="#003c6c",fg='#fdc700',
                        command= send )
    #Create the box to enter message
    EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial", wrap=WORD)
    EntryBox.bind("<Return>", send)
    #Place all components on the screen
    logo.place(x=6,y=6 , height = 90, width = 93)
    scrollbar.place(x=376,y=106, height=386)
    ChatLog.place(x=6,y=106, height=386, width=370)
    EntryBox.place(x=128, y=501, height=90, width=265)
    SendButton.place(x=6, y=501, height=90)
    
    base.mainloop()

