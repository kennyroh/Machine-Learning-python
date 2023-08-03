import csv
import string

from bs4 import BeautifulSoup
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


sentences = []

labels = []

# stopwords = ["a", "an", "and", "are", "as", "at", "be", "but"]
stopwords = []

table = str.maketrans('', '', string.punctuation)

with open('business-employment-data-march-2023-quarter-industry-revisions.csv', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        labels.append(row[1])
        sentence = row[3].lower()
        sentence = sentence.replace(',', ' , ')
        sentence = sentence.replace('.', ' . ')
        sentence = sentence.replace('-', ' - ')
        sentence = sentence.replace('/', ' / ')
        soup = BeautifulSoup(sentence, features="html.parser")  # 'sentences'를 'sentence'로 변경
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = ""
        for word in words:
            word = word.translate(table)
            if word not in stopwords:
                filtered_sentence += word + " "
        sentences.append(filtered_sentence)  # 문장 리스트에 필터링된 문장 추가

print(sentences)
training_size = 120

training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[:training_size]
testing_labels = labels[training_size:]

vocab_size = 200
max_length = 10
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_padded = pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type)

print(training_sequences[0])
print(training_padded[0])
