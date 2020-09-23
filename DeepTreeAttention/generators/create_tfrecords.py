#Tfrecords
import tensorflow as tf

def write_tfrecord(filename, sites, images, labels=None, classes=21, train=True, x=None, y=None):
    """Write a training or prediction tfrecord
        Args:
            train: True -> create a training record with labels. False -> a prediciton record with raster indices
        """
    writer = tf.io.TFRecordWriter(filename)

    #Write parser
    if train:
        for image, site, label in zip(images, sites, labels):
            tf_example = create_training_record(image=image, label=label, site=site, classes=classes)
            writer.write(tf_example.SerializeToString())
    else:
        if x is None:
            raise ValueError(
                "x and y raster indices are required when creating prediction records")
        for image, site, x, y in zip(images, sites, x, y):
            tf_example = create_prediction_record(image=image, classes=classes, site=site, x=x, y=y)
            writer.write(tf_example.SerializeToString())

    writer.close()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_training_record(image, site, label, classes):
    """Generate one record"""
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]

    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/data': _bytes_feature(image.tostring()),
            'label': _int64_feature(label),
            'site': _int64_feature(site),            
            'image/height': _int64_feature(rows),
            'image/width': _int64_feature(cols),
            'image/depth': _int64_feature(depth),
            'classes': _int64_feature(classes),
        }))

    # Serialize to string and write to file
    return example


def create_prediction_record(image, site, classes, x, y):
    """Generate one record"""
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]

    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/data': _bytes_feature(image.tostring()),
            'image/height': _int64_feature(rows),
            'image/width': _int64_feature(cols),
            'image/depth': _int64_feature(depth),
            'classes': _int64_feature(classes),
            'site': _int64_feature(site),            
            'x': _int64_feature(x),
            'y': _int64_feature(y),
        }))

    # Serialize to string and write to file
    return example


def _train_parse_(tfrecord):
    # Define features
    features = {
        'image/data': tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "site": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/depth": tf.io.FixedLenFeature([], tf.int64),
        "classes": tf.io.FixedLenFeature([], tf.int64),
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)
    classes = tf.cast(example['classes'], tf.int32)

    height = tf.cast(example['image/height'], tf.int64)
    width = tf.cast(example['image/width'], tf.int64)
    depth = tf.cast(example['image/depth'], tf.int64)
    site = tf.cast(example['site'], tf.int64)

    #recast
    label = tf.cast(example['label'], tf.uint16)
    label = tf.cast(example['label'], tf.uint8)

    # Load image from file
    image = tf.io.decode_raw(example['image/data'], tf.uint16)
    image_shape = tf.stack([height, width, depth])

    # Reshape to known shape
    loaded_image = tf.reshape(image, image_shape, name="cast_loaded_image")
    loaded_image = tf.cast(loaded_image, dtype=tf.float32)

    #one hot
    one_hot_labels = tf.one_hot(label, classes)

    return (loaded_image, site),  one_hot_labels


def _train_submodel_parse_(tfrecord):
    # Define features
    features = {
        'image/data': tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/depth": tf.io.FixedLenFeature([], tf.int64),
        "classes": tf.io.FixedLenFeature([], tf.int64),
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)
    classes = tf.cast(example['classes'], tf.int32)

    height = tf.cast(example['image/height'], tf.int64)
    width = tf.cast(example['image/width'], tf.int64)
    depth = tf.cast(example['image/depth'], tf.int64)

    #recast
    label = tf.cast(example['label'], tf.uint16)
    label = tf.cast(example['label'], tf.uint8)

    # Load image from file
    image = tf.io.decode_raw(example['image/data'], tf.uint16)
    image_shape = tf.stack([height, width, depth])

    # Reshape to known shape
    loaded_image = tf.reshape(image, image_shape, name="cast_loaded_image")
    loaded_image = tf.cast(loaded_image, dtype=tf.float32)

    #one hot
    one_hot_labels = tf.one_hot(label, classes)

    return loaded_image, (one_hot_labels, one_hot_labels, one_hot_labels)


def _predict_parse_(tfrecord):
    """Tfrecord parser for prediction. No labels available
        Args:
            tfrecord: path to tfrecord
        Returns:
            indices: x,y index of row, col position in original raster
            loaded_image: image data crop
        """
    # Define features
    features = {
        'image/data': tf.io.FixedLenFeature([], tf.string),
        'site': tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/depth": tf.io.FixedLenFeature([], tf.int64),
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "x": tf.io.FixedLenFeature([], tf.int64),
        "y": tf.io.FixedLenFeature([], tf.int64),
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)

    height = tf.cast(example['image/height'], tf.int64)
    width = tf.cast(example['image/width'], tf.int64)
    depth = tf.cast(example['image/depth'], tf.int64)
    site = tf.cast(example['site'], tf.int64)

    # Load image from file
    image = tf.io.decode_raw(example['image/data'], tf.uint16)
    image_shape = tf.stack([height, width, depth])

    # Reshape to known shape
    loaded_image = tf.reshape(image, image_shape, name="cast_loaded_image")
    loaded_image = tf.cast(loaded_image, dtype=tf.float32)

    raster_rows = tf.cast(example['x'], tf.int64)
    raster_cols = tf.cast(example['y'], tf.int64)

    return (loaded_image, site), raster_rows, raster_cols
