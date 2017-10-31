import tensorflow as tf


def flatten(name, input_tensor):
    shape = input_tensor.get_shape().as_list()
    (batch_size, *dimensions) = shape
    flattened_size = 1
    for dimension in dimensions:
        flattened_size *= dimension
    return tf.reshape(tensor=input_tensor, shape=[-1, flattened_size])


def get_layers1_model(x, classes_count, keep_prob):
    conv_kernel_size = (3, 3)
    conv_strides = (1, 1)
    pool_kernel_size = (2, 2)
    pool_strides = (2, 2)

    conv1_1 = tf.layers.conv2d(name='conv1_1', inputs=x, filters=32, kernel_size=conv_kernel_size, strides=conv_strides, padding='same',
                               kernel_initializer=tf.contrib.keras.initializers.he_normal(),  bias_initializer=tf.contrib.keras.initializers.he_normal(),
                               activation=None)
    conv1_1 = tf.maximum(conv1_1, 0.01 * conv1_1)

    conv1_2 = tf.layers.conv2d(name='conv1_2', inputs=conv1_1, filters=32, kernel_size=conv_kernel_size, strides=conv_strides, padding='same',
                               kernel_initializer=tf.contrib.keras.initializers.he_normal(), bias_initializer=tf.contrib.keras.initializers.he_normal(),
                               activation=None)
    conv1_2 = tf.maximum(conv1_2, 0.01 * conv1_2)

    maxpool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=pool_kernel_size, strides=pool_strides, name='pool1')

    conv2_1 = tf.layers.conv2d(name='conv2_1', inputs=maxpool1, filters=64, kernel_size=conv_kernel_size, strides=conv_strides, padding='same',
                               kernel_initializer=tf.contrib.keras.initializers.he_normal(), bias_initializer=tf.contrib.keras.initializers.he_normal(),
                               activation=None)
    conv2_1 = tf.maximum(conv2_1, 0.01 * conv2_1)

    conv2_2 = tf.layers.conv2d(name='conv2_2', inputs=conv2_1, filters=64, kernel_size=conv_kernel_size, strides=conv_strides, padding='same',
                               kernel_initializer=tf.contrib.keras.initializers.he_normal(), bias_initializer=tf.contrib.keras.initializers.he_normal(),
                               activation=None)
    conv2_2 = tf.maximum(conv2_2, 0.01 * conv2_2)

    maxpool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=pool_kernel_size, strides=pool_strides, name='pool2')

    conv3_1 = tf.layers.conv2d(name='conv3_1', inputs=maxpool2, filters=128, kernel_size=conv_kernel_size, strides=conv_strides, padding='same',
                               kernel_initializer=tf.contrib.keras.initializers.he_normal(), bias_initializer=tf.contrib.keras.initializers.he_normal(),
                               activation=None)
    conv3_1 = tf.maximum(conv3_1, 0.01 * conv3_1)

    conv3_2 = tf.layers.conv2d(name='conv3_2', inputs=conv3_1, filters=128, kernel_size=conv_kernel_size, strides=conv_strides, padding='same',
                               kernel_initializer=tf.contrib.keras.initializers.he_normal(), bias_initializer=tf.contrib.keras.initializers.he_normal(),
                               activation=None)
    conv3_2 = tf.maximum(conv3_2, 0.01 * conv3_2)

    maxpool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=pool_kernel_size, strides=pool_strides, name='pool3')

    conv4_1 = tf.layers.conv2d(name='conv4_1', inputs=maxpool3, filters=256, kernel_size=conv_kernel_size, strides=conv_strides, padding='same',
                               kernel_initializer=tf.contrib.keras.initializers.he_normal(), bias_initializer=tf.contrib.keras.initializers.he_normal(),
                               activation=None)
    conv4_1 = tf.maximum(conv4_1, 0.01 * conv4_1)

    conv4_2 = tf.layers.conv2d(name='conv4_2', inputs=conv4_1, filters=256, kernel_size=conv_kernel_size, strides=conv_strides, padding='same',
                               kernel_initializer=tf.contrib.keras.initializers.he_normal(), bias_initializer=tf.contrib.keras.initializers.he_normal(),
                               activation=None)
    conv4_2 = tf.maximum(conv4_2, 0.01 * conv4_2)

    maxpool4 = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=pool_kernel_size, strides=pool_strides, name='pool3')

    flat = flatten('flat', maxpool4)

    dense1 = tf.layers.dense(
        name='dense1',
        inputs=flat,
        units=512,
        kernel_initializer=tf.contrib.keras.initializers.he_normal(),
        bias_initializer=tf.contrib.keras.initializers.he_normal(),
        activation=None)
    dense1 = tf.maximum(dense1, 0.01 * dense1)
    drop1 = tf.nn.dropout(dense1, keep_prob=keep_prob)

    dense2 = tf.layers.dense(
        name='dense2',
        inputs=drop1,
        units=512,
        kernel_initializer=tf.contrib.keras.initializers.he_normal(),
        bias_initializer=tf.contrib.keras.initializers.he_normal(),
        activation=None)
    dense2 = tf.maximum(dense2, 0.01 * dense2)
    drop2 = tf.nn.dropout(dense2, keep_prob=keep_prob)

    logits = tf.layers.dense(
        name='logits',
        inputs=drop2,
        units=classes_count,
        kernel_initializer=tf.contrib.keras.initializers.he_normal(),
        bias_initializer=tf.contrib.keras.initializers.he_normal(),
        activation=None)

    return logits
