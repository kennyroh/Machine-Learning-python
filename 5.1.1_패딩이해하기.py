from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow import keras

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?',
    'Tom and Jerry is a good cartoon',
    ]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)



padded = pad_sequences(sequences)
print(padded)

padded = pad_sequences(sequences, padding='post')
print(padded)

padded = pad_sequences(sequences, padding='post', maxlen=5)
print(padded)

padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)


