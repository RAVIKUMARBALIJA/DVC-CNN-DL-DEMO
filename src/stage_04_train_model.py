import argparse
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.common import read_yaml, create_directories
from utils.model import load_model, get_unique_modelpath
from utils.callbacks import get_callbacks
from utils.data_management import train_valid_generator

STAGE = "TRAIN_MODEL" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    train_model_dir_path = os.path.join(artifacts_dir,artifacts["TRAINED_MODEL_DIR"])
    create_directories(train_model_dir_path)

    untrained_model_path = os.path.join(artifacts_dir,artifacts["BASE_MODEL_DIR"],artifacts["UPDATED_BASE_MODEL_NAME"])

    model = load_model(untrained_model_path)
    callback_dir_path = os.path.join(artifacts_dir,artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)

    train_generator, valid_generator = train_valid_generator(
        data_dir=artifacts["DATA_DIR"],
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmentation=params["AUGMENTATION"]
    )

    trian_steps_per_epoch = train_generator.samples/train_generator.batch_size
    valid_steps_per_epoch = valid_generator.samples/valid_generator.batch_size
    model.fit(train_generator,
    validation_data=valid_generator,
    epochs=params["EPOCHS"],
    stpes_per_epoch=trian_steps_per_epoch,
    validation_steps=valid_steps_per_epoch,
    callbacks=callbacks)

    create_directories([train_model_dir_path])
    model_unique_name = get_unique_modelpath(train_model_dir_path)
    model.save(model_unique_name)
    logging.info(f"trained model saved at \n {model_unique_name}")





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e