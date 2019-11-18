# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:00:04 2019

@author: Ajay Solanki
"""
import os
os.chdir('src')

import json
import numpy as np
import tensorflow as tf
import model, sample, encoder

#  model, sample and encoder are the Python files present in the src subfolder of the main GPT-2 folder:
class GPT_Test:
    def interact_model(self,
        model_name, # model_name: This indicates which model we are using. In our case, we are using the GPT-2 model with 345 million parameters or weights
        seed, # Integer seed for random number generators, fix seed to reproduce results
        nsamples, # This represents the number of sample texts generated in our output
        batch_size, #This only affects speed/memory. This must also divide nsamples
        length, # It represents the number of tokens in the generated text. If the length is None, then the number of tokens is decided by model hyperparameters
        temperature, # is controls randomness in Boltzmann distribution. Lower temperature results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive. Higher temperature results in more random completions
        top_k, #This parameter controls diversity. If the value of top_k is set to 1, this means that only 1 word is considered for each step (token). If top_k is set to 40, that means 40 words are considered at each step. 0 (default) is a special setting meaning no restrictions. top_k = 40 generally is a good value
        models_dir # It represents the path to parent folder containing model subfolders (contains the <model_name> folder)
    ):
        models_dir = "E:\\workdirectory\\Code Name Val Halen\\DS Sup\\DL\\Chapter A9\\gpt\\gpt-2\\models\\" #os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0
    
        enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
    
        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    
        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [batch_size, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k
            )
    
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(sess, ckpt)
    
            while True:
                raw_text = input("Model prompt >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("Model prompt >>> ")
                context_tokens = enc.encode(raw_text)
                generated = 0
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
                        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                        print(text)
                print("=" * 80)
                


gpt_test = GPT_Test()
gpt_test.interact_model( '345M',
    None,
    1,
    1,
    300,
    1,
    0,
    '/content/gpt-2/models')