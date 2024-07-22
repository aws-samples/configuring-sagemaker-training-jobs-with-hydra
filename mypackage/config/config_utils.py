import json
import logging
from pathlib import Path

import boto3
from omegaconf import DictConfig, OmegaConf


def load_config_yaml_from_module(module_name, config="default.yaml"):
    """Load config file from a module in the ioptimize.config directory."""
    config_path = Path(module_name.__file__).parent / config
    return OmegaConf.load(config_path)


def save_json_to_s3(data: dict, bucket_name: str, s3_key: str):
    """
    Save a JSON file to an S3 bucket.

    Args:
        data (dict): The data to be saved as JSON.
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The file key in the S3 bucket.

    Returns:
        bool: True if file was uploaded successfully, False otherwise.
    """
    # Initialize a boto3 S3 client
    s3 = boto3.client("s3")

    # Convert the data to a JSON string
    json_data = json.dumps(data)

    # Upload the JSON string to S3
    s3.put_object(Body=json_data, Bucket=bucket_name, Key=s3_key)


def load_sm_config_if_exists(cfg: DictConfig) -> DictConfig:
    """
    Load the SageMaker configuration from a JSON file if it exists.

    This function attempts to read a JSON configuration file located at
    '/opt/ml/input/data/config/config.json'. If the file exists, it is opened,
    read, and the JSON content is parsed into a Python dictionary. If the file
    does not exist, we return the default config.

    Note:
    This exact functionality is needed to check if configurations were passed
    in from Sagemaker, or if the default configurations should be used.

    Args:
        cfg (DictConfig): Hydra default config.

    Returns:
        dict or None: A dictionary containing the parsed JSON data if the file
        exists and is valid JSON or the default configuration.
    """
    sm_config_path = Path("/opt/ml/input/data/config/config.json")
    if Path(sm_config_path).exists():
        logging.info("Using config provided through SageMaker channel.")
        with open(sm_config_path, "r") as file:
            return OmegaConf.create(json.load(file))
    else:
        logging.info("Using config provided to entrypoint.")
        return cfg
