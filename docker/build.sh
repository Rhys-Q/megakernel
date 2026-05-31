set -e
set -x



BUILD_SCRIPT_DIR=$(cd $(dirname $0); pwd)
echo $BUILD_SCRIPT_DIR # docker 

docker build  -f megakernel.Dockerfile -t megakernel:latest .

cd ..