import logging
from datetime import datetime
from pathlib import Path

import boto3
import hydra
from omegaconf import OmegaConf

from mypackage import config

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

    hpo_job_name = f"my-hpo-job-{current_time}"
    # set SM algorithm parameters
    algorithm_image_uri = create_ecr_image_uri(repo_name, image_tag)

    # assemble training params for SM training job
    hpo_cfg = OmegaConf.to_container(
        cfg.sagemaker.hyper_parameter_optimization, resolve=True, throw_on_missing=True
    )
    hpo_params = {
        "Strategy": hpo_cfg["strategy"],
        "HyperParameterTuningJobObjective": {
            "Type": hpo_cfg["objective"]["type"],
            "MetricName": hpo_cfg["objective"]["metric_name"],
        },
        "ResourceLimits": {
            "MaxNumberOfTrainingJobs": hpo_cfg["resource_limits"][
                "max_number_ob_training_jobs"
            ],
            "MaxParallelTrainingJobs": hpo_cfg["resource_limits"][
                "max_parallel_training_jobs"
            ],
        },
        "ParameterRanges": {
            "IntegerParameterRanges": hpo_cfg["parameter_ranges"][
                "input_parameter_ranges"
            ]
        },
        "TrainingJobEarlyStoppingType": hpo_cfg["training_job_early_stopping_type"],
        "TuningJobCompletionCriteria": {
            "TargetObjectiveMetricValue": hpo_cfg["tuning_job_completion_criteria"][
                "target_objective_metric_value"
            ],
        },
    }

    algorithm_specification = {
        "TrainingImage": algorithm_image_uri,
        "TrainingInputMode": hpo_cfg["algorithm_specification"]["training_input_mode"],
        "MetricDefinitions": hpo_cfg["algorithm_specification"]["metric_definitions"],
    }

    input_data_config = [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": sm_cfg["input_data_s3_path"],
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }
    ]

    output_data_config = {"S3OutputPath": sm_cfg["output_data_s3_path"]}
    # Logging the submission
    logging.info(
        f"Submitting HPO job with name {hpo_job_name} and params: {hpo_params}"
    )
    # Create Hyperparameter Tuning Job
    sm_client.create_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=hpo_job_name,
        HyperParameterTuningJobConfig=hpo_params,
        TrainingJobDefinition={
            "AlgorithmSpecification": algorithm_specification,
            "RoleArn": role_arn,
            "ResourceConfig": {
                "InstanceCount": instance_count,
                "InstanceType": instance_type,
                "VolumeSizeInGB": volume_size_in_gb,
            },
            "InputDataConfig": input_data_config,
            "OutputDataConfig": output_data_config,
            "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_in_seconds},
        },
    )


if __name__ == "__main__":
    main()
