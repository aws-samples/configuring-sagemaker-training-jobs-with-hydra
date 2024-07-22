#!/bin/bash
# Default values

# Containerization tool
DEFAULT_DOCKER_CMD="docker"  # Change this to "finch" or "docker"

# Account id and region from where the docker base image is pulled
DEFAULT_AWS_ACCOUNT_ID_DOCKER_BASE_IMAGE="763104351884"

# Variables from command line arguments
REPO_NAME=$1
image_tag=$2
AWS_REGION=$3
ACCOUNT_ID=$4
DOCKER_CMD=${5:-$DEFAULT_DOCKER_CMD}
AWS_ACCOUNT_ID_DOCKER_BASE_IMAGE=${6:-$DEFAULT_AWS_ACCOUNT_ID_DOCKER_BASE_IMAGE}

# Check if the first two arguments are provided
if [ -z "$REPO_NAME" ] || [ -z "$image_tag" ]
then
    echo "Usage: $0 <REPO_NAME> <image_tag> [AWS_REGION] [ACCOUNT_ID]"
    echo "REPO_NAME and image_tag are required."
    exit 1
fi

# Log into Docker base image ECR
# NOTE: The account ID has to be aligned with the ID in the Dockerfile
aws ecr get-login-password --region ${AWS_REGION} | $DOCKER_CMD login --username AWS --password-stdin ${AWS_ACCOUNT_ID_DOCKER_BASE_IMAGE}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo "-----"

# Build the Docker image
$DOCKER_CMD build --build-arg REGION=$AWS_REGION -t ${REPO_NAME}:${image_tag} .

echo "-----"

# Tag the Docker image
TAG_LOCAL=${REPO_NAME}:${image_tag}
TAG_REMOTE=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${TAG_LOCAL}

echo $TAG_LOCAL
echo $TAG_REMOTE
$DOCKER_CMD tag $TAG_LOCAL $TAG_REMOTE

echo "-----"

# Log into workload ECR
aws ecr get-login-password --region ${AWS_REGION} | $DOCKER_CMD login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
echo "-----"

# Push the Docker image to the ECR repository
PUSH_COMMAND=${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${image_tag}
echo $PUSH_COMMAND
$DOCKER_CMD push $PUSH_COMMAND
