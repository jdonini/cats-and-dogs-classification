import numpy as np
import tensorflow as tf
from build_model import get_layers1_model as get_model
import cv2


def build_parse_sample(sample_shape, classes_count):
    def parse_sample(sample):
        features_schema = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)}
        features = tf.parse_single_example(sample, features_schema)
        image_flat = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(tf.reshape(image_flat, [*sample_shape]), tf.float32) / 255.0
        label_flat = tf.decode_raw(features['label'], tf.uint8)
        label = tf.cast(tf.reshape(label_flat, [classes_count]), tf.float32)
        return (image, label)
    return parse_sample


def main():
    learning_rate = 0.001
    epochs = 64
    batch_size = 128

    model_filepath = './model1/model'
    dataset = './dataset'
    image_shape = (128, 128, 3)
    classes_count = 37
    keep_probability = tf.placeholder(dtype=tf.float32, name='keep_prob')

    filepath_train = '{}/train.tfrecords'.format(dataset)
    dataset_train = tf.contrib.data.TFRecordDataset([filepath_train])
    dataset_train = dataset_train.map(
        map_func=build_parse_sample(image_shape, classes_count),
        num_threads=tf.constant(2, tf.int32),
        output_buffer_size=tf.constant(batch_size, tf.int64))
    dataset_train = dataset_train.batch(batch_size)

    filepath_test = '{}/test.tfrecords'.format(dataset)
    dataset_test = tf.contrib.data.TFRecordDataset([filepath_test])
    dataset_test = dataset_test.map(
        map_func=build_parse_sample(image_shape, classes_count),
        num_threads=tf.constant(2, tf.int32),
        output_buffer_size=tf.constant(batch_size, tf.int64))
    dataset_test = dataset_test.batch(batch_size)

    handle = tf.placeholder(tf.string, shape=[], name='handle')
    iterator = tf.contrib.data.Iterator.from_string_handle(
        string_handle=handle,
        output_types=dataset_train.output_types,
        output_shapes=dataset_train.output_shapes)
    (x, y) = iterator.get_next()
    x = tf.identity(input=x, name='input')
    y = tf.identity(input=y, name='label')

    print(y.shape)

    iterator_train = dataset_train.make_initializable_iterator(
        shared_name='iterator_train')
    iterator_test = dataset_test.make_initializable_iterator(
        shared_name='iterator_test')

    logits = get_model(
        x=x,
        classes_count=classes_count,
        keep_prob=keep_probability)
    softmax = tf.nn.softmax(logits=logits, name='output')
    cost = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y),
        name='cost')

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    corrects = tf.equal(tf.argmax(softmax, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32), name='accuracy')

    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()))
        handle_train = session.run(iterator_train.string_handle())
        handle_test = session.run(iterator_test.string_handle())

        print('\nTRAINING --------------------------------', end='\n')
        for epoch in range(epochs):
            steps = 0
            cost_acc = 0
            session.run(iterator_train.initializer)
            while True:
                try:
                    # images = session.run(x, feed_dict={handle: handle_train})
                    # for image in images:
                    #     cv2.imshow('Oi', np.uint8(image * 255))
                    #     cv2.waitKey(0)
                    feed_dict = {keep_probability: 0.4, handle: handle_train}
                    fetches = [optimizer]
                    fetches_result = session.run(
                        fetches=fetches, feed_dict=feed_dict)
                    steps += 1
                    print('.', end='')
                    if steps % 32 == 0:
                        print('\n')
                except tf.errors.OutOfRangeError:
                    break
            steps = 0
            cost_acc = 0
            accuracy_acc = 0
            session.run(iterator_train.initializer)
            while True:
                try:
                    feed_dict = {keep_probability: 1.0, handle: handle_train}
                    fetches = [cost, accuracy]
                    fetches_result = session.run(
                        fetches=fetches,
                        feed_dict=feed_dict)
                    cost_acc += fetches_result[0]
                    accuracy_acc += fetches_result[1]
                    steps += 1
                except tf.errors.OutOfRangeError:
                    break
            accuracy_avg = accuracy_acc / steps
            cost_avg = cost_acc / steps
            print('\n----------------------------')
            print('Epoch {:04}/{:04}:    Accuracy: {:0.4f}    Cost: {:012.4f}'.format(epoch + 1, epochs, accuracy_avg, cost_avg), end='\n')
            if epoch % 2 == 0:
                saver.save(session, model_filepath)

        steps = 0
        accuracy_acc = 0
        session.run(iterator_test.initializer)
        while True:
            try:
                feed_dict = {keep_probability: 1.0, handle: handle_test}
                fetches = [accuracy]
                fetches_result = session.run(
                    fetches=fetches,
                    feed_dict=feed_dict)
                accuracy_acc += fetches_result[0]
                steps += 1
            except tf.errors.OutOfRangeError:
                break
        accuracy_avg = accuracy_acc / steps
        print('TESTING ----------------------------')
        print('Accuracy: {:0.4f}'.format(accuracy_avg), end='\n')

        saver.save(session, model_filepath)


if __name__ == '__main__':
    main()
