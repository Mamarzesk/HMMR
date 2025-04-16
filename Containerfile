FROM ubuntu:22.04

WORKDIR /code

COPY . .

WORKDIR /minc

ADD --checksum=sha256:59db5cc05e512a6a240ec2141a6db00a32c6f864079804c823c2ba46860f78b4 \ 
http://packages.bic.mni.mcgill.ca/minc-toolkit/Debian/minc-toolkit-1.9.18-20200813-Ubuntu_20.04-x86_64.deb ./toolkit.deb

RUN apt-get update && apt-get -y install python3 \
    python3-pip \
    && apt-get -y install /minc/toolkit.deb \
    && pip install numpy \
    pyminc \
    scipy \
    torch_cubic_spline_grids \
    && pip install torch \
    torchaudio \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu126 

CMD ["bash", "/code/start.sh"]
