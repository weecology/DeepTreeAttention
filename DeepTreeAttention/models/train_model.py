"""Train a DeepTreeAttention model using a directory of hyperspectral .tif image crops or a set of tfrecords."""
import tensorflow as tf
import warnings

def _check_args(train_dataset, generator):
    if train_dataset is None:
        if generator is None:
            raise ValueError("train_dataset and generator are both None. Atleast one data input must be specific")
        else:
            warnings.warn("The generator input is mostly suited for small scale testing and debugging of tfrecords. For more robust and scalable training use tf.data, see https://www.tensorflow.org/guide/data#consuming_python_generators")
    
def train(model, config, train_dataset=None, test_dataset=None, generator=None, callbacks=None):
    """Train a DeepTreeAttention model. Models can be trained on either a keras-sequence generator or a tf.data dataset
    Args:
        model: a tensorflow graph for training
        config: a training dictionary config, see utils.config
        train_dataset: a tf.data dataset
        test_dataset: a tf.data.dataset
        generator:  A keras sequence generator
    Returns:
        model: a trained model graph
    """
    _check_args(train_dataset, generator)
    
    train_config = config["train"]
    
    #compile
    model.compile(
            loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(lr=float(train_config['learning_rate']))
        )
    
    epochs = train_config["epochs"]
    
    if generator:
        steps_per_epoch = train_config["steps_per_epoch"]
        model.fit(generator, callbacks=callbacks,epochs=epochs)
    else:
        #TODO does test_dataset go in a callback or as "validation dataset" https://www.tensorflow.org/guide/keras/train_and_evaluate#using_a_validation_dataset
        model.fit(train_dataset, callbacks=callbacks, epochs=epochs)
    
    return model
