# Sentiment-Analysis
This is used to detect the sentiment of text

Sentiment Analysis is the most common text classification tool that analyses an incoming message and tells whether the underlying sentiment is positive, negative our neutral.
Sentiment analysis is a powerful tool that allows computers to understand the underlying subjective tone of a piece of writing. This is something that humans have difficulty with, and as you might imagine, it isn’t always so easy for computers, either. But with the right tools and Python, you can use sentiment analysis to better understand the sentiment of a piece of writing.
![sentiment analysis](https://user-images.githubusercontent.com/77839791/108054505-be877e00-7074-11eb-9170-aabe45ee22af.png)

# Preprocessing text in order to understand and clean our data:

The sentiment analysis begins with the process of loading out data, once we have loaded our data we pass our data through an NLP pipeline, which does the following :
1. Tokenizes sentences : This process involves breaking down the sentences into words , or any other form of simpler text.
2. Removing stop words : This process involves removing words like "if","but" or "or" etc.
3. Normalizing words : This process involves consiedering or reducing all of the words into a single word.
4. Vectorizing text : This process involves the converting of text to numerical forms for the ease of use for our classifier.

All these steps serve to reduce the noise inherent in any human-readable text and improve the accuracy of your classifier’s results. There are lots of great tools to help with this, such as the Natural Language Toolkit, TextBlob, and spaCy.



# Results:
We could classify a sentence of our own and get positive results.

