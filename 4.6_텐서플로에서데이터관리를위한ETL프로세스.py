import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# 모델정의 시작#
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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 모델정의 끝#

# 추출 단계 시작 #
data = tfds.load('mnist', split='train', as_supervised=True)
val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)


# 추출 단계 끝 #

# 변환 단계 시작 #
def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255)
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation='nearest')
    return image, label


train = data.map(augmentimages)
train_batches = train.shuffle(seed=100).batch(32)
val_batches = val_data.batch(32)
# 변환 단계 끝 #

# 로드 단계 시작 #
history = model.fit(train_batches, epochs=10, validation_data=val_batches)
# 로드 단계 끝 #


