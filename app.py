
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import json 
import random
import pickle 
import logging
#import theano 
from keras.models import load_model


from flask import Flask,jsonify,request
#from flask_cors import CORS 

with open("intents.json") as file:
    data = json.load(file)


with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i]=(1)
    
    return np.array(bag)






def dialog(sen):
    res = model.predict([[bag_of_words(sen,words)]])
    res_index = np.argmax(res)
    tag = labels[res_index]
    for tg in data["intents"]:
             if tg['tag'] == tag:
                 responses = tg['responses']
    return random.choice(responses)



app = Flask(__name__)
#CORS(app)
model  = load_model('newmodel.h5')


@app.route('/chat',methods=['POST'])
def chat():
    user_input = request.json['msg']
    
    return jsonify({'msg':str(dialog(user_input))})


app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

if __name__ == '__main__':
     app.run(debug=True)  
      
   
 


