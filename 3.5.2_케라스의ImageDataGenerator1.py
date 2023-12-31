import urllib.request
import zipfile

from tensorflow.keras.optimizers import RMSprop
from tensorflow_core.python.training import learning_rate_decay

# url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
# file_name = 'horse-or-human.zip'
train_dir = 'training/'
# urllib.request.urlretrieve(url, file_name)

# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(train_dir)
# zip_ref.close()
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

model.fit(train_generator, epochs=10)
