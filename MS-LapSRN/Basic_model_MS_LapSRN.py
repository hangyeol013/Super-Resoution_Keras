from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

initializer = tf.keras.initializers.he_uniform(seed=1)


def initConv_model():
    img_input = layers.Input(shape=(None, None, 3))
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer=initializer)(img_input)

    model = Model(inputs=img_input, outputs=x, name='init_conv')

    return model


def embedding_block(layer_num):
    img_input = layers.Input(shape=(None, None, 64))
    x = layers.LeakyReLU(alpha=0.2)(img_input)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer=initializer)(x)

    for _ in range(layer_num - 1):
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer=initializer)(x)
        
    model = Model(inputs=img_input, outputs=x, name='embedding_block')

    return model


def embedding_model(layer_num, recursive_num):
    embedding = embedding_block(layer_num)
    img_input = layers.Input(shape=(None, None, 64))

    x = embedding(img_input)
    x = layers.add([x, img_input])

    for _ in range(recursive_num - 1):
        x = embedding(x)
        x = layers.add([x, img_input])

    x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                               kernel_initializer=initializer)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    model = Model(inputs=img_input, outputs=x, name='Embedding')

    return model


def res_sub():
    img_input = layers.Input(shape=(None, None, 64))

    res = layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same', kernel_initializer=initializer)(img_input)

    model = Model(inputs=img_input, outputs=res, name='res_sub')

    return model


def upsample_model():
    img_input = layers.Input(shape=(None, None, 3))

    upsample = layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2),
                                      padding='same', kernel_initializer=initializer)(img_input)

    model = Model(inputs=img_input, outputs=upsample, name='upsample')

    return model


def MS_LapSRN_model(D, R):
    initConv = initConv_model()
    embedding = embedding_model(layer_num=D, recursive_num=R)
    upsample = upsample_model()
    res = res_sub()

    img_input = layers.Input(shape=(None, None, 3))

    initConv_x = initConv(img_input)

    upsample_1 = upsample(img_input)
    embedding_1 = embedding(initConv_x)
    res_1 = res(embedding_1)

    hr_x2 = layers.add([upsample_1, res_1])

    upsample_2 = upsample(hr_x2)
    embedding_2 = embedding(embedding_1)
    res_2 = res(embedding_2)

    hr_x4 = layers.add([upsample_2, res_2])

    model = Model(inputs=img_input, outputs=[hr_x2, hr_x4])

    return model