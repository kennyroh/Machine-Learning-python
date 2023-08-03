import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))

for review in train_data:
    imdb_sentences.append(review['text'].decode('utf-8'))

tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)

sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(tokenizer.word_index)
