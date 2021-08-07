# Overview DETOXI
Toxicity has always been an issue. We can find toxicity in  almost all aspects of our lives. Unfortunately, people tend to have aggressive behavior towards each other, because of several reasons like competition. With the appearance of social media platforms like Facebook, Twitter, Instagram , etc; people around the world became close to each other and are now able to communicate and express their opinions at any time. However, this was not the only side of it. There were other issues that came along. Some people use social media as a platform for being rude to one another rather than politely expressing their opinions.
This is where our project “Detoxi” idea came from. We aim to limit the toxicity that surrounds us as much as possible. 


## Table of contents
* [Cleanning Code](#general-info)
* [GRU Model](#general-info)
* [BERT Model](#technologies)
* [XGBOOST Model](#setup)

## Cleanning Code
We aim to handle text cleaning in the preparation stage to minimize computing time without losing any information. We accomplished it by following these steps:

1- Remove empty rows
2- Remove @username from tweets
3- Contractions: We use the contractions package to expand the contraction in English such as: 
we'll -> we will  
we shouldn't've -> we should not have
4- Repeating the capitals words: Here we repeat the Capital words to confirm the meaning of this word as the user tries to focus on it.
5- Remove Noise: Text data could include various unnecessary characters or punctuation such as URLs, HTML tags, non-ASCII characters, or other special characters (symbols, emojis, and other graphic characters).
6- Remove Numbers.
7- Replace the Typos, slang, acronyms or informal abbreviations: We attempt to replace abbreviations and slang with a hash table that replaces abbreviations with their meaning.
8- Convert to lower case: The most common approach in text cleaning is capitalization or lower case due to the diversity of capitalization to form a sentence. This technique will project all words in text and document into the same feature space. However, it would also cause problems with exceptional cases such as the USA or UK, which could be solved by replacing typos, slang, acronyms or informal abbreviations technique.
9- Handling Negation: Understanding was one of our model's most difficult problems. As a result, we attempt to deal with it in the following manner: 
 Finding ["not", "n't", "no", "none", "neither", "nor", "never"]. 
 Using “?.,!:;” text punctuation, locate the conclusion of the negated phrase to detect the negation scoop. 
Then look for synonyms and antonyms for each synonym, returning the negated term if one is discovered.
 And finally remove the negation words as not, nor ,...etc and return the negated  words.
10- Remove the punctuation
11- Tokenization: This is a common technique that splits a sentence into tokens, where a token could be characters, words, phrases, symbols, or other meaningful elements. By breaking sentences into smaller chunks, that would help to investigate the words in a sentence and also the subsequent steps in the NLP pipeline, such as stemming.
12- Remove the Stop Words using nltk.
13- POS tagging: Part of speech tagging (POS tagging) distinguishes the part of speech (noun, verb, adjective, and etc.) of each word in the text.
14- Lemmatization: This is the task of determining that two words have the same root, despite their surface differences. The words am, are, and is have the shared lemma be; the words dinner and dinners both have the lemma dinner. Lemmatizing each of these forms to the same lemma will let us ﬁnd all mentions of words in Russian like Moscow.
15- Remove the empty rows after lemmatization

## GRU Model

* Bidirectional GRU 
Bidirectional GRU are really just putting two independent GRUs together. This structure allows the networks to have both backward and forward information about the sequence at every time step.
Using bidirectional will run your inputs in two directions, one from past to future and the other from future to past. What distinguishes this approach from unidirectional is that in the Simple GRU that runs backward, information from the future is preserved, whereas using the two hidden states combined, you can preserve information from both past and future at any point in time.

* Features extraction
