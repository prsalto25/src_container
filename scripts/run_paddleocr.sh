# Set the container name and image
image="juliussin/paddleocrapi:0.1.2"
container="paddleocrapi"

# Define the base directory for dashboard resources

# Stop and remove the existing container if it's running
docker stop $container
docker rm $container

# Run the new container with specified configurations
docker run -d --gpus all --restart always --name $container -p 28000:8000 -p 5555:5555 -e MODE=both -e USE_GPU=True $image