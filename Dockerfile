FROM nvidia/cuda:11.0-runtime

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        git gcc g++ python3-dev espeak libsndfile1 python3 python3-pip python3-setuptools wget locales \
        && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
	update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
	
RUN pip install -U pip && pip install TTS

RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

RUN tts --list_models && \
    tts --text "Text for test." \
        --model_name "tts_models/en/ljspeech/tacotron2-DDC" \
        --out_path /tmp/output.wav && \
    rm -Rf /tmp/output.wav

# new release should be out soon with non english models working again, 21 Jun 2021

WORKDIR /app

EXPOSE 5002

ENTRYPOINT [ "/usr/bin/python3", "-m", "TTS.server.server" ]
CMD [ "--use_cuda", "1", "--tts_checkpoint", "/model/tts.pth.tar", "--tts_config", "/model/tts.json", "--vocoder_checkpoint", "/model/vocoder.pth.tar", "--vocoder_config", "/model/vocoder.json"] 

