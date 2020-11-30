
#Change this base image acccording to your convenience
# for tensorflow  v1.X
# FROM nvcr.io/nvidia/tensorflow:20.09-tf1-py3

#for tensorflow v2.X
FROM nvcr.io/nvidia/tensorflow:20.09-tf2-py3

#for pytorch 
#FROM nvcr.io/nvidia/pytorch:20.09-py3


ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN set -x && \
    apt update && \
    apt install -y --no-install-recommends \
        jupyter \
        git\
        wget\
        build-essential \
        apt-utils \
        ca-certificates \
        curl \
        software-properties-common \
        libopencv-dev \ 
        python3-dev \
        python3-pip \ 
        python3-setuptools \
        cmake \
        swig \
        wget \
        unzip

 
COPY misc/requirements.txt /tmp/
RUN pip3 install pip --upgrade
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# publish port
#EXPOSE 5656

# Example Entry point
#ENTRYPOINT ["/bin/bash","-c", "/usr/bin/python3 -m server.py"]
