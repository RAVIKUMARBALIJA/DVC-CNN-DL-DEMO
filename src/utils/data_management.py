from tkinter import Image
import tensorflow as tf
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_valid_generator(data_dir:str="data",
    IMAGE_SIZE: tuple=(224,224),
    BATCH_SIZE: int=32,
    do_data_augmentation: bool=True) -> tuple:

    datagenerator_kwargs = dict(rescale = 1./255,
        validation_split=0.2)
    
    dataflow_kwargs = dict(
        target_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        interpolation = "bilinear"
    )
    
    valid_datagenerator = ImageDataGenerator(**datagenerator_kwargs)

    valid_generator = valid_datagenerator.flow_from_directory(
        directory=data_dir,
        subset="test",
        shuffle=False,
        **dataflow_kwargs
    )

    if do_data_augmentation:
        train_datagenerator = ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            vertical_flip=True,
            **datagenerator_kwargs
        )
        logging.info(f"data augmentation is used for training")
    else:
        train_datagenerator = valid_datagenerator
        logging.info(f"data augmentation is not used for training")

    train_generator = train_datagenerator.flow_from_directory(
        directory=data_dir,
        subset="train",
        shuffle=True,
        **dataflow_kwargs)

    logging.info("train and valid generators are created")
    return train_generator, valid_generator