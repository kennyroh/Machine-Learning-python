import tensorflow as tf
import tensorflow_datasets as tfds
# mnist_data = tfds.load('fashion_mnist')
# for item in mnist_data:
#     print(item)

# mnist_data = tfds.load('fashion_mnist', split='train')
# assert isinstance(mnist_data, tfds.core.Dataset)
# print(type(mnist_data))
# for item in mnist_data.take(1):
    # print(type(item))
    # print(item.keys())
    # print(item['image'])
    # print(item['label'])


mnist_data, info = tfds.load('fashion_mnist', split='train', with_info=True)
print(info)

