#Tfrecords
import tensorflow as tf
def write_tfrecord(filename, images, labels, classes):
    writer = tf.io.TFRecordWriter(filename)
    
    for image, label in zip(images, labels):
        tf_example = create_tf_example(image, label, classes)
        writer.write(tf_example.SerializeToString())
    writer.close()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tf_example(image, label, classes):
    print("Classes : {}".format(classes))
    """Generate one record"""
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]
    
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/data':_bytes_feature(image.tostring()),
            'label': _int64_feature(label),
            'image/height': _int64_feature(rows),
            'image/width': _int64_feature(cols),
            'image/depth': _int64_feature(depth),
            'classes': _int64_feature(classes)                    
        }))

    # Serialize to string and write to file
    return example

def _parse_fn(tfrecord):
    # Define features
    features = {
        'image/data': tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),                
        "image/depth": tf.io.FixedLenFeature([], tf.int64),                        
        "classes": tf.io.FixedLenFeature([], tf.int64)        
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)
    classes = tf.cast(example['classes'], tf.int32)
    
    height = tf.cast(example['image/height'], tf.int64)
    width = tf.cast(example['image/width'], tf.int64)
    depth = tf.cast(example['image/depth'], tf.int64)
    label = tf.cast(example['label'], tf.int64)
    
    # Load image from file, what dtype?
    image = tf.io.decode_raw(example['image/data'], tf.int64)
    image_shape = tf.stack([height, width, depth])
    
    # Reshape to known shape
    loaded_image = tf.reshape(image, image_shape, name="cast_loaded_image")
    
    #one hot
    one_hot_labels = tf.one_hot(label, classes)
    
    return loaded_image, one_hot_labels