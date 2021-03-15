#!/bin/bash

DOCKER_IMG_NAME=xd:$1
DOCKERFILE_PATH=aws/Dockerfile1.7.1
MY_URL=212469337107.dkr.ecr.us-east-1.amazonaws.com

# Assume that the script runs from the root directory.
# May need --no-cache
#docker build --build-arg NAS_VER=$(date +%Y%m%d-%H%M%S) -t ${DOCKER_IMG_NAME} -f ${DOCKERFILE_PATH} .
docker build --build-arg NAS_VER=$(date +%Y%m%d-%H%M%S) -t ${DOCKER_IMG_NAME} -f ${DOCKERFILE_PATH} .

if [[ $2 == push ]]; then
    echo "Pushing to AWS..."
    if [[ $3 == renbo ]]
    then
        # Login into ECR.
        docker tag ${DOCKER_IMG_NAME} ${MY_URL}/${DOCKER_IMG_NAME}
        aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 212469337107.dkr.ecr.us-east-1.amazonaws.com
        docker push ${MY_URL}/${DOCKER_IMG_NAME}
    fi
fi