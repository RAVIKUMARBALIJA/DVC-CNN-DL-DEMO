import io
import os
import json
import pickle
import logging
import sys

from tensorflow.keras import callbacks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import get_timestamp
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping

def create_and_save_tensorboard_callbacks(callbacks_dir: str, tensorboard_logdir:str) -> None:
    unique_name = get_timestamp("tb_logs")
    tb_running_log_dir = os.path.join(tensorboard_logdir, unique_name)
    tb_callaback = TensorBoard(log_dir = tb_running_log_dir)

    tb_callaback_filepath = os.path.join(callbacks_dir, "tensorboard_cb.cb")
    pickle.dump(tb_callaback, open(tb_callaback_filepath,"wb"))

    logging.info(f"tensorflow callback is saved at {tb_callaback_filepath}")

def create_and_save_checkpointing_callback(callbacks_dir, model_checkpoint_dir):
    checkpoint_file = os.path.join(model_checkpoint_dir, "ckpt_model.h5")
    checkpoint_callback = ModelCheckpoint(checkpoint_file, save_best_only=True)

    ckpt_callback_filepath = os.path.join(callbacks_dir, "checkpoint_cb.cb")
    pickle.dump(checkpoint_callback, open(ckpt_callback_filepath,"wb"))

    logging.info(f"checkpoint callback is saved at {ckpt_callback_filepath}")

def get_callbacks(callback_dir_path):

    callback_paths = [
        os.path.join(callback_dir_path,file)for file in os.listdir(callback_dir_path) if str(file).ends_with("cb")
    ]

    callbacks = [
        pickle.load(path) for path in callback_paths
    ]

    logging.info(f"saved callbacks are now saved and ready to be saved")

    return callbacks


