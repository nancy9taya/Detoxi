# Overview DETOXI
Toxicity has always been an issue. We can find toxicity in  almost all aspects of our lives. Unfortunately, people tend to have aggressive behavior towards each other, because of several reasons like competition. With the appearance of social media platforms like Facebook, Twitter, Instagram , etc; people around the world became close to each other and are now able to communicate and express their opinions at any time. However, this was not the only side of it. There were other issues that came along. Some people use social media as a platform for being rude to one another rather than politely expressing their opinions.
This is where our project “Detoxi” idea came from. We aim to limit the toxicity that surrounds us as much as possible. 


## Table of contents:
* [DataSets](#datasets)
* [Cleanning Code](#cleanning-code)
* [GRU Model](#gru-model)
* [BERT Model](#bert-model)
* [XGBOOST Model](#xgboost-model)
* [Author](#author)

## DataSets
We used [Kaggle's Civil Comments dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data), 
[Hate Speech and Offensive dataset](https://www.kaggle.com/mrmorj/hate-speech-and-offensive-language-dataset)
and [Insulting Tweets during the 2019 Federal Election in Canada](https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/9VSRHU)
datasets to train, validate and test our models. 




## Cleanning Code
* [Cleanning Code](https://github.com/nancy9taya/Detoxi/blob/main/CleaningCode.ipynb)

We aim to handle text cleaning in the preparation stage to minimize computing time without losing any information. We accomplished it by following these steps:

* Remove empty rows
* Remove @username from tweets
* Contractions: We use the contractions package to expand the contraction in English such as: 
we'll -> we will  
we shouldn't've -> we should not have
* Repeating the capitals words: Here we repeat the Capital words to confirm the meaning of this word as the user tries to focus on it.
* Remove Noise: Text data could include various unnecessary characters or punctuation such as URLs, HTML tags, non-ASCII characters, or other special characters (symbols, emojis, and other graphic characters).
* Remove Numbers.
* Replace the Typos, slang, acronyms or informal abbreviations: We attempt to replace abbreviations and slang with a hash table that replaces abbreviations with their meaning.
* Convert to lower case: The most common approach in text cleaning is capitalization or lower case due to the diversity of capitalization to form a sentence. This technique will project all words in text and document into the same feature space. However, it would also cause problems with exceptional cases such as the USA or UK, which could be solved by replacing typos, slang, acronyms or informal abbreviations technique.
* Handling Negation: Understanding was one of our model's most difficult problems. As a result, we attempt to deal with it in the following manner: 
 Finding ["not", "n't", "no", "none", "neither", "nor", "never"]. 
 Using “?.,!:;” text punctuation, locate the conclusion of the negated phrase to detect the negation scoop. 
Then look for synonyms and antonyms for each synonym, returning the negated term if one is discovered.
 And finally remove the negation words as not, nor ,...etc and return the negated  words.
* Remove the punctuation
* Tokenization: This is a common technique that splits a sentence into tokens, where a token could be characters, words, phrases, symbols, or other meaningful elements. By breaking sentences into smaller chunks, that would help to investigate the words in a sentence and also the subsequent steps in the NLP pipeline, such as stemming.
* Remove the Stop Words using nltk.
* POS tagging: Part of speech tagging (POS tagging) distinguishes the part of speech (noun, verb, adjective, and etc.) of each word in the text.
* Lemmatization: This is the task of determining that two words have the same root, despite their surface differences. The words am, are, and is have the shared lemma be; the words dinner and dinners both have the lemma dinner. Lemmatizing each of these forms to the same lemma will let us ﬁnd all mentions of words in Russian like Moscow.
* Remove the empty rows after lemmatization

## GRU Model
* [GRU Model Code](https://github.com/nancy9taya/Detoxi/blob/main/Model_GRU_2Emb.ipynb)
* ### Bidirectional GRU:

Bidirectional GRU are really just putting two independent GRUs together. This structure allows the networks to have both backward and forward information about the sequence at every time step.
Using bidirectional will run your inputs in two directions, one from past to future and the other from future to past. What distinguishes this approach from unidirectional is that in the Simple GRU that runs backward, information from the future is preserved, whereas using the two hidden states combined, you can preserve information from both past and future at any point in time.

* ### Features extraction:

In our model, each little detail about the user's tweet might help the model learn more about the user's state and the emotion expressed in the text.
The features collected from text are useful information that is passed to a neural network.

|Feature  | Usage|
| ------------- | ------------- |
| Counting the capitals  | The capital letters are meant to emphasize the importance of the words, and many people have employed them to express strong feelings about words |
| Counting the unique words  | The unique words show how distinctive the phrases are and how long they are, rather than merely repeating words |
| Counting the punctuations  | It indicates that the user wants to pause, stop, or emphasize specific sections of the text |
| Counting the exclamation_marks  | The number of exclamation marks so times depend on personal attacks |
| Counting the question_marks | The number of question marks so times depend on personal attacks  |
| Counting the you_count | To figure out how the tweet contains personal information for a specific person |
| Counting the mentions  | The mentions demonstrate how the user wants to draw attention to himself or herself from certain individuals or organizations |
| Counting the smilies  | The smilies may indicate either a friendly tweet or sarcastic |
| Counting the symbols | The high number of symbols usually used on insults **&#$%“”¨«»®´·º½¾¿¡§£₤‘’* |


* ### Tokenizing and Texts_to_sequences:

Tokenization is a common technique that splits a sentence into tokens, where a token could be characters, words, phrases, symbols, or other meaningful elements. By breaking sentences into smaller chunks, that would help to investigate the words in a sentence and also the subsequent steps in the NLP pipeline, such as lemmatization.
Then “texts_to_sequences” transforms each text into a sequence of integers. So it basically takes each word in the text and replaces it with its corresponding integer value from the word_index dictionary.

* ### Custom Padding:

We want all of the texts to have the same length, so we use padding to add zeros to the end of the vector, however this does not work properly.
As a result, we created customized padding that repeats the word sequences until the desired length is reached. Our model's performance is improved because of the custom padding.

* ### Combining Two Different Word Embedding:

We decide to use word embedding GloVe which is a powerful word embedding technique that has been used for text classiﬁcation The approach is used as each 0word is presented by a high dimension vector and trained based on the surrounding words over a huge corpus. The pre-trained word embedding used in many works is based on 400,000 vocabularies trained over Wikipedia 2014 and Gigaword 5 as the corpus and 50 dimensions for word presentation. GloVe also provides other pre-trained word vectorizations with 100, 200, 300 dimensions which are trained over even bigger corpora, including Twitter content.
Combining the GloVe vector with FastText that improves on Word2Vec by taking word parts into account, too. This trick enables training of embeddings on smaller datasets and generalization to unknown words.
We also replaced the words that aren't in formal english with the embedding GloVe and FastText representation of the word “something”.

* ###  Attention Layer:

Although an GRU  is supposed to capture the long-range dependency better than the RNN, it tends to become forgetful in specific cases. Another problem is that there is no way to give more importance to some of the input words compared to others while translating the sentence.
As a result, whenever the suggested model creates a phrase, it looks for a collection of hidden states in the encoder where the most relevant information is available. This concept is known as 'Attention.' .In Bahdanau's work, he proposed an attention mechanism that learns to align and translate jointly. It is also known as Additive attention as it performs a linear combination of encoder states and the decoder states. 


* ### Evaluation:

We split the data set into 80% for train and validation , 20% for  test
The 80% we split it to training and validation using K-folds with K=10.

| Training Accuracy | Valdition Accuracy| Test Accuracy |
| --- | --- | --- |
| 95% | 92.8% | 92.8% | 


## BERT Model
* [BERT Model Code](https://github.com/nancy9taya/Detoxi/blob/main/finalbert.ipynb)

Transformer, an attention mechanism that learns contextual relationships between words (or sub-words) in a text, is used by BERT. Transformer contains two distinct processes in its basic form: an encoder that reads the text input and a decoder that generates a job prediction. Only the encoder mechanism is required because BERT's objective is to create a language model.
The detailed workings of Transformer are described in a paper by Google called "Attention is all you need".
This property enables the model to deduce the context of a word from its surroundings (left and right of the word). This property enables the model to deduce the context of a word from its surroundings (left and right of the word).

* ### We use Small BERT
Small BERTS feature the same fundamental design as large BERTS but include fewer and/or smaller Transformer pieces, allowing you to experiment with performance, size, and quality tradeoffs while also being easier to fine-tune. As a result, we chose Small BERTS to leverage Transformer's cutting-edge design to boost performance in a timely manner.

* ### The Preprocessing Model
Before being entered into BERT, text inputs must be converted to numeric token ids and organized in multiple Tensors.

A BERT model would employ three outputs from the preprocessing (input words id, input mask, and input type ids):
* The input has been trimmed to a maximum of 128 tokens.
* Because it is a single phrase input, the input type ids only contain one value (0).
* It would have one number for each input for a multiple phrase input.

* ## Using BERT Model
The BERT models produce a map with three crucial keys: pooled output, sequence output, and encoder outputs: pooled output, sequence output, and encoder outputs:
Each input sequence is represented by pooled output. [batch size, H] is its shape. This may be thought of as a movie review embedded.
Each token in the context is represented by sequence output. [batch size, seq length, H], for example, is the shape. Consider it a contextual embedding for each token in the film review.
The encoder outputs are the L Transformer blocks' intermediate activations. outputs["encoder outputs"] For I L, I is a Tensor of shape [batch size, seq length, 1024] that contains the outputs of the i-th Transformer block. sequence output is the final value in the list.
We'll utilize the pooled output array for fine-tuning purposes.

* ## Evaluation

We Split the Data to 80% Train and 20% Valdition

| Training Accuracy | Valdition Accuracy| Test Accuracy |
| --- | --- | --- |
| 93.5% | 95% | 91% | 


## XGBOOST Model
* [XGBOOST Model Code](https://github.com/nancy9taya/Detoxi/blob/main/xgboost-bert.ipynb)

Our model is simply taking a text from GUI and passing it through our 3 sub models to extract  finest features  from text and combining their predictions to be the input to the XGboost to find the best response either to hide the comment or leave it.
The figure shows how the Xgboost architecture works by combining the submodel predictions to the final one
![Xgboost](https://github.com/nancy9taya/Detoxi/blob/main/XGBOOST.png)

## Author
[Nancy Hassan](https://github.com/nancy9taya)
Email: (Nancy.hassan99@eng-st.cu.edu.eg)
