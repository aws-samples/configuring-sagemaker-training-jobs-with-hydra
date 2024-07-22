import logging
from datetime import datetime
from pathlib import Path

import boto3
import hydra
from omegaconf import OmegaConf

from mypackage import config
from mypackage.config.config_utils import save_json_to_s3

config_path = Path(config.__file__).parent
sm_client = boto3.client("sagemaker")
sts_client = boto3.client("sts")
session = boto3.session.Session()


def create_ecr_image_uri(repo_name: str, image_tag: str) -> str:
    """Create ecr image uri from repo_name and image_tag"""
    account = sts_client.get_caller_identity()["Account"]
    region = session.region_name

    # Create ECR image URI
    ecr_image = f"{account}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{image_tag}"

    return ecr_image


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg):
    """Launch optimization on SageMaker training job."""
    logging.info(OmegaConf.to_yaml(cfg))

    # SageMaker configs
    sm_cfg = cfg["sagemaker"]
    repo_name = sm_cfg["repo_name"]
    image_tag = sm_cfg["image_tag"]
    role_arn = sm_cfg["role_arn"]
    instance_type = sm_cfg["instance_type"]
    instance_count = sm_cfg["instance_count"]
    volume_size_in_gb = sm_cfg["volume_size_in_gb"]
    max_runtime_in_seconds = sm_cfg["max_runtime_in_seconds"]

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    training_job_name = f"my-training-{current_time}"
    # set SM algorithm parameters
    algorithm_image_uri = create_ecr_image_uri(repo_name, image_tag)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    s3_config_key = f"{sm_cfg['config_s3_prefix']}/{training_job_name}/config.json"
    logging.info(
        f"Saving config file to S3 bucket {sm_cfg['config_s3_bucket']}, key {s3_config_key}."
    )
    save_json_to_s3(cfg, sm_cfg["config_s3_bucket"], s3_config_key)

    # assemble training params for SM training job
    training_params = {
        "AlgorithmSpecification": {
            "TrainingImage": algorithm_image_uri,
            "TrainingInputMode": "File",
        },
        "RoleArn": role_arn,
        "ResourceConfig": {
            "InstanceCount": instance_count,
            "InstanceType": instance_type,
            "VolumeSizeInGB": volume_size_in_gb,
        },
        "TrainingJobName": training_job_name,
        "StoppingCondition": {
            "MaxRuntimeInSeconds": max_runtime_in_seconds,
        },
        "HyperParameters": {},  # Not using hyperparams as they are supplied through config.
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": sm_cfg["input_data_s3_path"],
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
            },
            {
                "ChannelName": "config",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{sm_cfg['config_s3_bucket']}/{s3_config_key}",
                        "S3DataDistributionType": "FullyReplicated",
                    },
                },
            },
        ],
        "OutputDataConfig": {
            "S3OutputPath": sm_cfg["output_data_s3_path"],
        },
    }

    logging.info(f"Submitting training job with params: {training_params}")
    sm_client.create_training_job(**training_params)


if __name__ == "__main__":
    main()
