from ubuntu:21.04

# disable prompts
env DEBIAN_FRONTEND noninteractive

run apt update && \ 
    apt-get install -y wget \
                      vim \
                      tmux \
                      git \
		      build-essentials

# install ispc 
run wget https://github.com/ispc/ispc/releases/download/v1.16.1/ispc-v1.16.1-linux.tar.gz
run tar -xvf ispc-v1.16.1-linux.tar.gz
env PATH $PATH:${HOME}/Downloads/ispc-v1.16.1-linux/bin


# download assignment starter code
run git clone https://github.com/stanford-cs149/asst1.git
