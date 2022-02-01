FROM python:3.9-buster
COPY ./ /app
WORKDIR /app

RUN pip3 install -r requirements.txt \
    && mkdir -p /root/.cache/torch/hub/checkpoints/

ADD https://contestfiles.s3.eu-central-1.amazonaws.com/nyds/train.zip?roistat_visit=358957 ./data/train.zip

RUN mkdir ./data/train/ \
    && unzip ./data/train.zip -d ./data/train/ \
    && rm ./data/train.zip \
    && mkdir -p ./data/weight/step_1 \
    && curl https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth -o /root/.cache/torch/hub/checkpoints/efficientnet_b7_lukemelas-dcc49843.pth \
    && curl https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth -o /root/.cache/torch/hub/checkpoints/efficientnet_b6_lukemelas-c76e70fd.pth
    # && curl https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth -o /root/.cache/torch/hub/checkpoints/resnext101_32x8d-8ba56ff5.pth
    # && curl https://download.pytorch.org/models/resnet34-b627a593.pth -o /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth

CMD ["python3","run.py"]
