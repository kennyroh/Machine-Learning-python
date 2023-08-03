from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np
import tensorflow as tf

tokenizer = Tokenizer()
data="Sure! Here's a random paragraph for you: The sun slowly descended below the horizon, casting a warm golden glow across the tranquil landscape. The air was filled with a gentle breeze, rustling the leaves of the tall oak trees that lined the path. As dusk settled in, the stars began to emerge, dotting the darkening sky like tiny diamonds. The sound of crickets chirping in the distance added to the symphony of nature, creating a peaceful ambiance. It was a moment of quiet serenity, where worries and troubles seemed to melt away, leaving only a sense of calm and contentment."
corpus = data.lower().split('\n')

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(tokenizer)
print(total_words)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# print(input_sequences[1:])

max_sequence_len = max([len(x) for x in input_sequences])
print(max_sequence_len)

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
