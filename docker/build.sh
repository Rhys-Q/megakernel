set -e
set -x



BUILD_SCRIPT_DIR=$(cd $(dirname $0); pwd)
echo $BUILD_SCRIPT_DIR # docker 

docker build  -f megakernel.Dockerfile -t megakernel:latest .

cd ..

# docker run --gpus all -dit --privileged  --name megakernel --restart always   -v /home/hz/qzq_work:/workspace  -v /root/docker_home:/root  megakernel:latest bash