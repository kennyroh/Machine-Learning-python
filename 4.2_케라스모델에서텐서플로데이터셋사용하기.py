import tensorflow as tf
import tensorflow_datasets as tfds

(training_images, training_labels), (test_images, test_labels) = tfds.load('fashion_mnist',
                                                                           split=['train', 'test'],
                                                                           batch_size=-1,
                                                                           as_supervised=True)

training_images = tf.cast(training_images, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

