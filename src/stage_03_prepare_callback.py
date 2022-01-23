import argparse
import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.callbacks import create_and_save_checkpointing_callback, create_and_save_tensorboard_callbacks
from utils.common import read_yaml, create_directories


STAGE = "PREPARE_CALLBACKS" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    #params = read_yaml(params_path)
    

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    tb_log_dir = os.path.join(artifacts_dir, artifacts["TENSORBOARD_ROOT_LOG_DIR"])
    callbacks_dir = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    checkpoint_dir = os.path.join(artifacts_dir, artifacts["CHECKPOINT_DIR"])

    create_directories([artifacts_dir,
    callbacks_dir, checkpoint_dir])

    create_and_save_tensorboard_callbacks(callbacks_dir,tb_log_dir)
    create_and_save_checkpointing_callback(callbacks_dir, checkpoint_dir)




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    #args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e