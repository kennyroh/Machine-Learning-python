import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
    'Tom and Jerry is a good cartoon',
    ]
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

test_data = [
    'i really love my dog',
    'my dog loves my manatee'

    ]

test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)
