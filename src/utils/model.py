import os
import io
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.python.ops.gen_array_ops import unique
from src.utils.common import get_timestamp

def _get_model_summary(model):
    with io.StringIO() as stream:
        model.summary(
            print_fn=lambda x: stream.write(f"{x}\n")
        )
        summary_str = stream.getvalue()
    return summary_str

def get_base_model(input_shape:list = [224,224],model_path:str = "basemodel.h5")->tf.keras.models.Model:
    base_model = VGG16(weights="imagenet",\
                input_shape=input_shape,\
                include_top=False)

    logging.info(f"base model summary{_get_model_summary(base_model)}")
    base_model.save(model_path)
    logging.info(f"base model has been saved at {model_path}")

    return base_model

def prepare_fullmodel(base_model,learning_rate=0.0005,classes=2,freeze_all=False,freeze_till=None):

    if freeze_all:
        for layer in base_model.layers[:]:
            layer.trainable = False
    elif freeze_till is not None and freeze_till>0:
        for layer in base_model.layers[:-freeze_till]:
            layer.trainable = False

    flatten = Flatten()(base_model.output)
    output_layer = Dense(units=classes,activation="softmax")(flatten)

    full_model = Model(base_model.input,output_layer)

    optimizer = Adam(learning_rate=learning_rate)
    
    full_model.compile(optimizer = optimizer,loss="categorical_cross_entropy",metrics=["accuracy"])
    logging.info(f"cutom model is built and ready to be trained")
    logging.info(f"full model summary{_get_model_summary(full_model)}")


    return full_model

def load_model(model_path:str):
    model = tf.keras.models.load_model(model_path)
    logging.info(f"model has been loaded from {model_path}")
    logging.info(f"loaded model summary {_get_model_summary(model)}")
    return model

def get_unique_modelpath(model_dir,model_name):
    modelstr = get_timestamp(model_name)
    unique_modelname = modelstr + ".h5"
    model_path = os.path.join(model_dir,unique_modelname)
    return model_path
