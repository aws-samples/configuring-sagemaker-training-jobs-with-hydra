# Shell script ensures that docker is logged into the correct account

# Default region
ARG REGION=us-east-1
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.0.0-cpu-py310

# Echo selected region
ARG REGION
RUN echo "The build is using the region: ${REGION}"


ARG PACKAGE=mypackage

ENV PATH="/opt/ml/code:${PATH}"

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
WORKDIR /opt/ml/code

COPY ${PACKAGE}/. /opt/ml/code/${PACKAGE}/

COPY poetry.lock pyproject.toml README.md /opt/ml/code/
COPY scripts/. /opt/ml/code/
COPY ${PACKAGE}/config/. /opt/ml/code/

ENV SAGEMAKER_PROGRAM sagemaker_entrypoint.py

RUN pip install --no-cache /opt/ml/code/
