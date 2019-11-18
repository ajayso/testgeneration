# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import keras
import numpy as np
from keras import layers
import random
import sys

class TextGenerator:
    
    def load_data(self):
        path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        self.text = open(path).read().lower()
        
    def sample(self,preds, temperature = 1.0):
        preds = np.array(preds).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds,1)
        return(np.argmax(probas))
    

    def process_data(self):
        maxlen = 100
        step = 3
        sentences = []
        next_chars =[]
        
        # populating the current sentences along with the expected next character in the sequence
        # as we need to predict the next char
        for i in range(0, len(self.text) - maxlen, step):
            sentences.append(self.text[i: i + maxlen])
            next_chars.append(self.text[i + maxlen])
        
        chars = sorted(list(set(self.text))) # similar concept of a set in mathematics will give unique unordered collection of data type that is iterable, mutable with no duplicates
        #Dictionary that maps unique characters to there indexes in the list chars
        char_indices = dict((char, chars.index(char)) for char in chars)
        #vectorization
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype = np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        
        #One Hot Encoding the characters into binary
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
            
        model = keras.models.Sequential()
        model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(layers.Dense(len(chars), activation='softmax'))
        optimizer = keras.optimizers.RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer = optimizer)
        
        #Training the language model and sampling for it
        # Given the training model and a seed text snippet , you can generate the next text  by going the following steps repeteadly
        # 1. Draw from the model a probability distribution for the next character, given the generated text so far
        # 2. Reweight the distribution to a certain temperature 
        # 3. Sample the new character at random according to the reweighted distribution
        # 4. Add the new character at the end of the available text
        # Finally the following loop will repeatedly train and generate text. 
        # You will begin generating text using a range of different temperature after every epoch.
        # This allows you to see how the generated text evolves as the model begins to converge as well as impact 
        # of the temperature on the sampling strategy
        
        for epoch in range(1,60):
            print('epoch', epoch)
            model.fit(x, y , batch_size = 128 , epochs = 1)
            start_index = random.randint(0, len(self.text) - maxlen -1)
            generated_text = self.text[start_index: start_index + maxlen]
            print('--- Generating with seed: "' + generated_text + '"')
            
            for temperature in [0.2, 0.5,1.0, 1.2]:
                print('------ temperature:', temperature)
                sys.stdout.write(generated_text)
                for i in range(400):
                    sampled = np.zeros((1, maxlen, len(chars)))
                    for t, char in enumerate(generated_text):
                        sampled[0, t, char_indices[char]] = 1
                    
                    preds = model.predict(sampled, verbose=0)[0]
                    next_index= self.sample(preds, temperature)
                    next_char = chars[next_index]
                    
                    generated_text += next_char
                    generated_text= generated_text[1:]
                    
                    sys.stdout.write(next_char)
                    
                        
        
        
        
        
        
        
text_generator = TextGenerator()
text_generator.load_data()
text_generator.process_data()
      