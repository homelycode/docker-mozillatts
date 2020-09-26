FROM nvidia/cuda:10.1-runtime

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        espeak libsndfile1 python3 python3-pip python3-setuptools

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
	update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
	
RUN pip install -U pip

RUN pip install https://github.com/reuben/TTS/releases/download/ljspeech-fwd-attn-pwgan/TTS-0.0.1+92aea2a-py3-none-any.whl

RUN pip install numba==0.48 torch==1.6.0 torchvision

EXPOSE 5002

ENTRYPOINT [ "/usr/bin/python3", "-m", "TTS.server.server" ]
CMD [ "--use_cuda", "1"] 