import tensorflow as tf
from tensorflow import keras

tv = keras.layers.TextVectorization(max_tokens=10000)


sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
    'Tom and Jerry is a good cartoon',
]

tv.adapt(sentences)

print(tv.get_vocabulary())

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tv(test_data)
print(test_seq.numpy())
