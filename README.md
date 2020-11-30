# VirtualStaining using cGANs
<img src="assets/Fig 3-1.png"/>


Train and evaluate conditional generative adveserial networks to virtually stain stacks of brightfield images.

This project is powered by [DeepTrack 2.0](https://github.com/softmatterlab/DeepTrack-2.0)


## Installation

Clone the repository using `git clone https://github.com/softmatterlab/VirtualStaining`

### Installing dependencies using pip

Requires python >= 3.6

Open a terminal and run `pip install -r misc/requirements.txt`

### Installing dependencies using Docker

Requires Docker.

Open a terminal, move into VirtualStaining, and run `docker build . -t VirtualStaining`

Start the container by running `docker run --name VirtualStaining -it VirtualStaining`


