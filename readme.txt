Text generation using RNN
Recurrent Neural Networks Brief Introduction
Coming to think to a neural network we generally start off with off information flowing in one direction (feed forward) and all the neuron in a layer are connected to all the neurons in the previous layer and so on which are typically called as Dense network. Now Dense Networks are good at solving some basic problems. The first look at a dense network and its pretty clear we are dealing too many neurons at every layer and the calculation involve grows with the number of neurons at each layer.
 
Coming to think of scenarios whereby 
A.	There is no need for every neuron in a specific network layer to be connected to the neurons in the previous layer. There is a need to only connect a small set of neurons in one layer to another. 
B.	There may be a need to for a set of neurons in a specific layer to be connected to other than previous layer.
If you go think about any network design, there could be a need for neurons or a set of neurons to be connected differently then standard feed forward neural network.
Deriving in the way we think about data and problems and considering each network layer as a processing unit, the need to connect processing units among themselves and to data differently to solve the overall the large problem gives to the push to the idea of having different deep neural network types. 
A lot of the data pattern around us seem to follow a sequence for example text data or time series data there is method to the madness as what to expect as the next character or number in the sequence. 
Recurrent Neural Network is type of deep neural network which has the capability to maintain state or sequence of the data been processed, 
For example, let’s say we need to predict next word in a sentence.
The key message here the words in the text are related unlike the general assumption the data fed to neural networks are not related, 
 For argument sake an MLP can be used to do the job at hand. A simple MLP with 1 hidden layer accepts the input, applies the activation function to arrive at the output. A Dense network layer will have a series of hidden layers to do the same job.  With series of hidden layer comes multiple activation functions and weights and biases. These weights will be independent in nature. The idea here is identify the relationship of the sequence of words assuming we can feed inputs to hidden layers and each of the hidden layer use the same weights and bias, this becomes the solution.
 
These layers are combined and weights, biases are same for all layers.  If we roll up these layers into a single layer this becomes the recurrent layer.
The idea behind RNN is to make use of sequential information, The traditional neural network the assumption is the inputs and outputs are independent, In sequential data input and output are related  Like for example we need to predict next word in a sentence, its important to know which words came before RNNs perform the same task for every element of a sequence with the output being dependent on the previous computation. Another way to think about RNN is they have a memory which captures information about what has been calculated so far.  RNNs are expected to make use of information in arbitrarily long sentences but in practice they can look back a few steps, this is where LSTM comes in.

 
Using RNN to generate sequence data

Having understood the capability of RNN storing sequence of data, its pretty straightforward to generate a sequence of data if we have a trained RNN network with all the required data.  This is where things get more interesting. 
For instance, given the input “The dog is b…” The network is trained to predict a as the next character.  The data input for neural network the text or characters are tokenized and converted to vectors and fed to the RNN.  Any network which can model the probability of the next token given the previous one is called a Language Model.  A complete pretrained LSTM English language model are available which can be readily used find the link here.
A language model captures the latent space of languages it’s a statistical structure. 
How to use a trained language model for text generation.
1.	Feed a sample text (conditioning) to the trained model. 
2.	The trained model will predict probabilities of possible next character based on the previous one. 
3.	The generated output chosen is fed back to step 1
The loop allows to generate an arbitrary length that reflects the structure of data on which the model is trained, sequences that are almost human written sequences.  In the example in the code section this in action we take an LSTM layer, feed it strings of N character extracted from a text corpus and train to predict N + 1. The output of the model will be a SoftMax over all possible characters, a probability distribution for the next character. This LSTM is called character level neural language model.
When generating text, the choice of next character is important, there are multiple approaches on how to choose the next character.
-	Greedy Sampling:  Always choose the character with the highest probability, such an approach results in repetitive, predictable strings that don’t look like coherent language.
-	Stochastic Sampling: A more interesting approach makes slightly more surprising choices, it introduces randomness in the process, by sampling from probability distribution for the next character.
Sampling probabilistically from SoftMax output of the model is neat, it allows even unlikely characters to be sampled some of the time, generating more interesting looking sentences and sometimes showing creativity by coming up with new, realistic sounding words that didn’t occur in the training data.

There is a need to have method to control the randomness or the temperature. When sampling from generative models it’s always good to explore different amount of randomness in the generation process. At the end of the day it’s us humans are ultimate judges of how interesting the generated data is, interestingness is highly subjective and there is no telling in advance where the point of optimal entropy lies.
            In order to control amount of stochasticity in the sampling process. we'll introduce the parameter called SoftMax temperature that characterizes the entropy of the probability distribution used for sampling.  It characterizes how surprising or predictable the choice of the next character will be. Given a temperature value a new probability distribution is calculated from the original one (the SoftMax output of the model) by reweighting the following way.


Sample Code -I 
The sample code uses sufficiently large text of Lord of the Rings from Wikipedia. In this text you will see the writing of Nietzsche, late nineteenth century German philosopher translated into English. The language model we learn will be specifically Nietzsche writing style and topics of choice rather than a more generic model of English language.
The network used for the sample is straightforward LSTM and SoftMax layer
 








The loop which keeps generating the next character (which is eventual input for next iteration)
 
The sample function which generates the new probabilities based on the temperature decided.
The working sample code can be found here.
Sample Code -II
OpenAI released generative pre-training model (GPT) which achieved the state-of-the-art result in many NLP task in 2018. GPT is leveraged transformer to perform both unsupervised learning and supervised learning to learn text representation for NLP downstream tasks.
To demonstrate the success of this model, OpenAI enhanced it and released a GPT-2 in Feb 2019. GPT-2 is trained to predict next word based on 40GB text. Unlike other model and practice, OpenAI does not publish the full version model but a lightweight version. They mentioned it in their blog:
Sample code uses GPT-2 light for text generation, To run the code you will need 
1.	Download GPT-2 git clone https://github.com/openai/gpt-2.git
2.	Download the small pretrained model - python3 download_model.py 117M
3.	Run the sample code GPT_Trail.py for text generation.
On executing the code it will prompt for an input below is a sample of “Today is my birthday and
 

A full blown GPT -2 can be tested at this link - https://talktotransformer.com/




