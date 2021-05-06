docker build -t data_training .
docker run --rm -it \
    --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1,2,3,4,5,6 \
    -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/ \
    -v ${PWD}/media:/media \
    -v ${PWD}/files:/files \
    -v ${PWD}/logs:/logs \
    -v ${PWD}/results:/results \
    data_training

