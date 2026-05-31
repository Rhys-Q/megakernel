From pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt install apt-transport-https ca-certificates  llvm  pkg-config git tmux vim -y
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install jaxtyping einops pytest ninja numpy psutil typing_extensions  pytest cloudpickle xgboost uvicorn fastapi pydantic_settings
RUN git config --global --add safe.directory '*' && git config --global user.email "1826249828@qq.com" && git config --global user.name "qzq"
