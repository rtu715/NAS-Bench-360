#!/bin/bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 212469337107.dkr.ecr.us-east-1.amazonaws.com
DOCKER_IMG_NAME=xd:$1
ECR_URL=212469337107.dkr.ecr.us-east-1.amazonaws.com
AWS_ACCESS_KEY_ID=AKIAIUU44JQI6IVWK6CQ
docker pull ${ECR_URL}/${DOCKER_IMG_NAME}

#need to modify -v volume parameters
docker run -it --runtime nvidia -v /home/renbot/workspace:/workspace --rm -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} ${ECR_URL}/${DOCKER_IMG_NAME}