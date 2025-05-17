#!/bin/bash

image="juliussin/algo_ultralytics:1.0.1"
container="algo_ultralytics"

# Define the base directory for dashboard resources
dashboard_resources="$(pwd)/../../dashboard2"
src="$(pwd)/../"
resources="$dashboard_resources/dockerDeployment/resources"

# Print the dashboard resources path
echo "Using dashboard resources from: $dashboard_resources"

# Stop and remove the existing container if it's running
docker stop $container 2>/dev/null
docker rm $container 2>/dev/null

# Run the new container with specified configurations
docker run --gpus all \
    --restart always \
    -v $src:/home/src/ \
    -v $resources:/home/resources/ \
    -v $resources:/home/videos/ \
    --name $container \
    -w /home/src/ \
    --network host \
    $image
