import sys
import os
import argparse
import shutil
from tensorflow.python.keras.backend import update
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.model import get_base_model,get_unique_modelpath,prepare_fullmodel


STAGE = "PREPARE_BASEMODEL" ## <<< change stage name 

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
    artifacts_dir = config["ARTIFACTS_DIR"]

    base_model_dir = artifacts["BASE_MODEL"]
    base_model_name = artifacts["BASE_MODEL_NAME"]

    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)

    create_directories([base_model_dir_path])

    base_model_path = os.path.join(base_model_dir_path, base_model_name)

    base_model = get_base_model(input_shape=params["IMAGE_SIZE"],model_path=base_model_path)

    full_model = prepare_fullmodel(base_model, learning_rate=params["LEARNING_RATE"],classes=2,
                freeze_all=True, freeze_till=None)
    
    updated_model_path = os.path.name(base_model_dir_path,config["UPDATED_BASE_MODEL_NAME"])

    full_model.save(updated_model_path)
    logging.info(f"full model has been saved at {updated_model_path}")


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