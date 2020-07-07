import warnings
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Dense, Lambda, Softmax
from keras.layers import Input, ZeroPadding2D, Add, GlobalAveragePooling2D, Concatenate, Conv2DTranspose, Multiply
from keras.models import Model
from keras.utils.data_utils import get_file

warnings.filterwarnings('ignore')


def masking(X, f, stage, block, depth):
    filters = [f, f * 2, f * 4, f * 8, f * 16]

    if depth == 1:
        X = conv_block(X, filters[1], stage, block + 1, strides=(1, 1))
        X = conv_block(X, filters[0], stage, block + 2, strides=(1, 1))
    elif depth == 2:
        X1 = conv_block(X, filters[1], stage, block + 1, strides=(1, 1))
        P1 = MaxPooling2D((2, 2))(X1)
        X2 = conv_block(P1, filters[2], stage, block + 2, strides=(1, 1))

        X3 = Conv2DTranspose(filters[1], kernel_size=(2, 2), strides=(2, 2))(X2)
        X3 = BatchNormalization(axis=3)(X3)
        X3 = Activation('relu')(X3)
        X3 = Concatenate(axis=-1)([X3, X1])
        X3 = conv_block(X3, filters[1], stage, block + 3, strides=(1, 1))

        X = conv_block(X3, filters[0], stage, block + 4, strides=(1, 1))
    elif depth == 3:
        X1 = conv_block(X, filters[1], stage, block + 1, strides=(1, 1))
        P1 = MaxPooling2D((2, 2))(X1)
        X2 = conv_block(P1, filters[2], stage, block + 2, strides=(1, 1))
        P2 = MaxPooling2D((2, 2))(X2)
        X3 = conv_block(P2, filters[3], stage, block + 3, strides=(1, 1))

        X4 = Conv2DTranspose(filters[2], kernel_size=(2, 2), strides=(2, 2))(X3)
        X4 = BatchNormalization(axis=3)(X4)
        X4 = Activation('relu')(X4)
        X4 = Concatenate(axis=-1)([X4, X2])
        X4 = conv_block(X4, filters[2], stage, block + 4, strides=(1, 1))

        X5 = Conv2DTranspose(filters[1], kernel_size=(2, 2), strides=(2, 2))(X4)
        X5 = BatchNormalization(axis=3)(X5)
        X5 = Activation('relu')(X5)
        X5 = Concatenate(axis=-1)([X5, X1])
        X5 = conv_block(X5, filters[1], stage, block + 5, strides=(1, 1))

        X = conv_block(X5, filters[0], stage, block + 6, strides=(1, 1))
    elif depth == 4:
        X1 = conv_block(X, filters[1], stage, block + 1, strides=(1, 1))
        P1 = MaxPooling2D((2, 2))(X1)
        X2 = conv_block(P1, filters[2], stage, block + 2, strides=(1, 1))
        P2 = MaxPooling2D((2, 2))(X2)
        X3 = conv_block(P2, filters[3], stage, block + 3, strides=(1, 1))
        P3 = MaxPooling2D((2, 2))(X3)
        X4 = conv_block(P3, filters[4], stage, block + 4, strides=(1, 1))

        X5 = Conv2DTranspose(filters[3], kernel_size=(2, 2), strides=(2, 2), padding='same')(X4)
        X5 = BatchNormalization(axis=3)(X5)
        X5 = Activation('relu')(X5)
        X5 = Concatenate(axis=-1)([X5, X3])
        X5 = conv_block(X5, filters[3], stage, block + 5, strides=(1, 1))

        X6 = Conv2DTranspose(filters[2], kernel_size=(2, 2), strides=(2, 2), padding='same')(X5)
        X6 = BatchNormalization(axis=3)(X6)
        X6 = Activation('relu')(X6)
        X6 = Concatenate(axis=-1)([X6, X2])
        X6 = conv_block(X6, filters[2], stage, block + 6, strides=(1, 1))

        X7 = Conv2DTranspose(filters[1], kernel_size=(2, 2), strides=(2, 2), padding='same')(X6)
        X7 = BatchNormalization(axis=3)(X7)
        X7 = Activation('relu')(X7)
        X7 = Concatenate(axis=-1)([X7, X1])
        X7 = conv_block(X7, filters[1], stage, block + 7, strides=(1, 1))

        X = conv_block(X7, filters[0], stage, block + 8, strides=(1, 1))

    X = Softmax(axis=-1)(X)
    return X


def identity_block(X, f, stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'

    X_shortcut = X

    X = BatchNormalization(axis=3, name=bn_name + '1')(X)
    X = Activation('relu', name=relu_name + '1')(X)
    X = Conv2D(filters=f, kernel_size=(3, 3), strides=(1, 1), padding='same', bias=False, name=conv_name + '1')(X)

    X = BatchNormalization(axis=3, name=bn_name + '2')(X)
    X = Activation('relu', name=relu_name + '2')(X)
    X = Conv2D(filters=f, kernel_size=(3, 3), strides=(1, 1), padding='same', bias=False, name=conv_name + '2')(X)

    X = Add()([X, X_shortcut])
    return X


def conv_block(X, f, stage, block, strides=(2, 2)):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'

    X = BatchNormalization(axis=3, name=bn_name + '1')(X)
    X = Activation('relu', name=relu_name + '1')(X)

    X_shortcut = Conv2D(filters=f, kernel_size=(1, 1), strides=strides, padding='same', bias=False, name=sc_name)(X)

    X = Conv2D(filters=f, kernel_size=(3, 3), strides=strides, padding='same', bias=False, name=conv_name + '1')(X)

    X = BatchNormalization(axis=3, name=bn_name + '2')(X)
    X = Activation('relu', name=relu_name + '2')(X)
    X = Conv2D(filters=f, kernel_size=(3, 3), strides=(1, 1), padding='same', bias=False, name=conv_name + '2')(X)

    X = Add()([X, X_shortcut])
    return X


def ResMaskingNet(input_shape=None,
                  classes=7):
    img_input = Input(shape=input_shape)

    bn_axis = 3
    x = Lambda(lambda img: tf.image.resize(img, (224, 224)))(img_input)
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = Conv2D(64, (7, 7),
               strides=(2, 2),
               padding='valid',
               bias=False,
               kernel_initializer='he_normal',
               name='conv0')(x)
    x = BatchNormalization(axis=bn_axis, name='bn0')(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pooling0')(x)

    x = conv_block(x, 64, stage=0, block=0, strides=(1, 1))
    x = identity_block(x, 64, stage=0, block=1)
    x = identity_block(x, 64, stage=0, block=2)
    M = masking(x, 64, stage=0, block=92, depth=4)
    M = Lambda(lambda m: 1 + m)(M)
    x = Multiply()([x, M])

    x = conv_block(x, 128, stage=1, block=0)
    x = identity_block(x, 128, stage=1, block=1)
    x = identity_block(x, 128, stage=1, block=2)
    x = identity_block(x, 128, stage=1, block=3)
    M = masking(x, 128, stage=1, block=93, depth=3)
    M = Lambda(lambda m: 1 + m)(M)
    x = Multiply()([x, M])

    x = conv_block(x, 256, stage=2, block=0)
    x = identity_block(x, 256, stage=2, block=1)
    x = identity_block(x, 256, stage=2, block=2)
    x = identity_block(x, 256, stage=2, block=3)
    x = identity_block(x, 256, stage=2, block=4)
    x = identity_block(x, 256, stage=2, block=5)
    M = masking(x, 256, stage=2, block=95, depth=2)
    M = Lambda(lambda m: 1 + m)(M)
    x = Multiply()([x, M])

    x = conv_block(x, 512, stage=3, block=0)
    x = identity_block(x, 512, stage=3, block=1)
    x = identity_block(x, 512, stage=3, block=2)
    M = masking(x, 512, stage=3, block=92, depth=1)
    M = Lambda(lambda m: 1 + m)(M)
    x = Multiply()([x, M])

    x = BatchNormalization(axis=3, name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    # x = AveragePooling2D(pool_size=(2,2), padding='same', name='avg_pool')(x)
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(classes, activation='softmax', name='fc7')(x)

    # Create model.
    model = Model(img_input, x, name='resnet')

    weights_path = get_file(
        'resnet34_imagenet_1000_no_top.h5',
        'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet34_imagenet_1000_no_top.h5',
        cache_subdir='models',
        md5_hash='8caaa0ad39d927cb8ba5385bf945d582'
    )
    model.load_weights(weights_path, by_name=True)

    return model
