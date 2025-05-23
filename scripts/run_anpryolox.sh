image="graymatics1/anpr_yolox:v5"
container="anpr_yolox_server"

docker stop $container
docker rm $container
docker run --gpus all \
    --name $container \
    --network host \
    -e PORT=5004 \
    -p 5004:5004 \
    --workdir /workspace/YOLOX/tools \
    -it \
    --entrypoint='python' \
    $image yolox_s_server.py
