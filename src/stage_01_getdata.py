import argparse
import os
import sys
import shutil
from tqdm import tqdm
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.common import read_yaml,create_directories,clear_existing_dirs

STAGE = "get_data" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    
    source_path = config["source_download_dirs"]
    train_source = source_path["train"]
    test_source = source_path["test"]


    destination_path = config["local_data_dirs"]
    destination_data = destination_path["data"]
    train_dest_path = destination_path["train"]
    train_dog = destination_path["train_dog"]
    train_cat = destination_path["train_cat"]

    test_dest_path = destination_path["test"]

    clear_existing_dirs(destination_data)
    create_directories([destination_data,train_dest_path,train_dog,train_cat,test_dest_path])

    for file in tqdm(os.listdir(train_source)):
        if "dog" in file.lower():
            shutil.copy(os.path.join(train_source,file),train_dest_path+"/dog")
        elif "cat" in file.lower():
            shutil.copy(os.path.join(train_source,file),train_dest_path+"/cat")
    
    for file in tqdm(os.listdir(test_source)):
        shutil.copy(os.path.join(test_source,file),test_dest_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e