image="juliussin/algo_ultralytics:0.1.0"
container="algo_ultralytics"

# Define the base directory for dashboard resources
dashboard_resources="$(pwd)/../../dashboard2"  # Change this to your actual path
src="$(pwd)/../"
resources="$dashboard_resources/dockerDeployment/resources"

# Stop and remove the existing container if it's running
docker stop $container
docker rm $container

# Run the new container with specified configurations
docker run --gpus all \
    --restart always \
    -it \
    -v $src:/home/src/ \
    -v $resources:/home/resources/ \
    -v $resources:/home/videos/ \
    --name $container \
    -w /home/src/ \
    --network host \
    --entrypoint "/bin/bash" \
    $image
    #-c "bash run.sh"