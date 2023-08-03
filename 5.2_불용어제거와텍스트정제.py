sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?',
    'Tom and Jerry is a good cartoon',
]

from bs4 import BeautifulSoup

soup = BeautifulSoup(sentences)
stopwords = ['and', 'is', 'are', 'the', 'a', 'an']
words = sentences.split()
filtered_sentence = ''
for word in words:
    if word not in stopwords:
        filtered_sentence = filtered_sentence + ' ' + word
sentences.append(filtered_sentence)

