#Tfrecords
import tensorflow as tf
def write_tfrecord(filename, images, labels):
    writer = tf.io.TFRecordWriter(filename)
    
    for image, label in zip(images, labels):
        tf_example = create_tf_example(image, label)
        writer.write(tf_example.SerializeToString())

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tf_example(image, label, classes):
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/data':
                _bytes_feature(image.tostring()),
            'one_hot_label':
                _int64_feature(label),
            'image/shape':
                _int64_feature(image.shape),
            'classes':
                _int64_feature(classes)                    
        }))

    # Serialize to string and write to file
    return example

def _parse_fn(example):
    # Define features
    features = {
        'image/data': tf.io.FixedLenFeature([], tf.float32),
        "label": tf.FixedLenFeature([], tf.int64),
        "image/shape": tf.FixedLenFeature([], tf.int64),                
        "classes": tf.FixedLenFeature([], tf.int64)        
    }

    # Load one example and parse
    example = tf.io.parse_single_example(example, features)

    # Load image from file
    image = tf.cast(example["image/data"], tf.float32)
    target_shape = tf.cast(example["image/shape"], tf.int64)

    # Reshape to known shape
    loaded_image = tf.reshape(image, target_shape, name="cast_loaded_image")
    
    #one hot
    one_hot_labels = tf.one_hot(labels, classes, dtype=tf.int32)
    classes = tf.one_hot(classes, dtype=tf.int32)
    
    return loaded_image, one_hot_labels