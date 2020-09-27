FROM nvidia/cuda:11.0-runtime

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        git gcc g++ python3-dev espeak libsndfile1 python3 python3-pip python3-setuptools wget locales \
        && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
	update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
	
RUN pip install -U pip && pip install wheel

RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

# TTS Model = Tacotron2 DDC
# model: https://drive.google.com/file/d/1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos/view?usp=sharing
# config: https://drive.google.com/file/d/18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc/view?usp=sharing
# stats: https://drive.google.com/file/d/1uULLlv2J7LYNuQsSfrvGBa2C-x30e1aO/view?usp=sharing

# Vocoder Model = 
# model: https://drive.google.com/file/d/1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K/view?usp=sharing
# config: https://drive.google.com/file/d/1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu/view?usp=sharing

# Vocoder Model = ParallelWaveGAN 
# model: https://drive.google.com/file/d/13oKu33Fj0cwH661apA2_vsjn9rT1NkEb/view?usp=sharing
# config: https://drive.google.com/file/d/1qN7vQRIYkzvOX_DtiZtTajzoZ1eW1-Eg/view?usp=sharing

RUN cd /tmp && \
    wget -q https://github.com/tanaikech/goodls/releases/download/v1.2.7/goodls_linux_amd64 && \
    chmod a+x ./goodls_linux_amd64 && \
    mkdir /model && \
    cd /model && \
    echo 'https://drive.google.com/file/d/1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K/view?usp=sharing' | /tmp/goodls_linux_amd64 --np -f vocoder.pth.tar && mv checkpoint_1450000.pth.tar vocoder.pth.tar && \
    echo 'https://drive.google.com/file/d/1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu/view?usp=sharing' | /tmp/goodls_linux_amd64 --np -f vocoder.json && mv config.json vocoder.json && \
    echo 'https://drive.google.com/file/d/1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos/view?usp=sharing' | /tmp/goodls_linux_amd64 --np  -f tts.pth.tar && mv checkpoint_130000.pth.tar tts.pth.tar && \
    echo 'https://drive.google.com/file/d/18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc/view?usp=sharing' | /tmp/goodls_linux_amd64 --np -f tts.json && mv config.json tts.json && \
    echo 'https://drive.google.com/file/d/1uULLlv2J7LYNuQsSfrvGBa2C-x30e1aO/view?usp=sharing' | /tmp/goodls_linux_amd64 --np && \
    mv /tmp/goodls_linux_amd64 ./goodls

RUN cd /tmp && \
    git clone https://github.com/rahulpowar/TTS.git && \
    cd TTS/ && \
    python setup.py bdist_wheel && \
    pip install ./dist/*.whl && \
    mv TTS/server/templates /usr/local/lib/python3.8/dist-packages/TTS/server/. && \
    rm -Rf /tmp/TTS

# Patch for 'AttrDict' object has no attribute 'gst' in config
# Also turn on windowing to stop early term https://github.com/mozilla/TTS/issues/170#issuecomment-486494470
RUN echo '{"gst":{"gst_embedding_dim": null,"gst_num_heads": null,"gst_style_tokens": null}}' > /tmp/patch.json && \
	echo '{"windowing" : true}' > /tmp/windowing.json && \
    pip install demjson && \
    apt-get update && apt-get install --yes jq && rm -rf /var/lib/apt/lists/* && \ 
    mv /model/tts.json /model/_tts.json && \
    jsonlint -Sf --sort preserve /model/_tts.json > /tmp/tts.json && \
    jq -s '.[1] + .[0] + .[2]' /tmp/tts.json /tmp/patch.json /tmp/windowing.json > /tmp/merged.json && \
    jsonlint -Sf --sort preserve  --html-safe /tmp/merged.json > /model/tts.json

WORKDIR /model

#RUN pip install https://github.com/reuben/TTS/releases/download/ljspeech-fwd-attn-pwgan/TTS-0.0.1+92aea2a-py3-none-any.whl
#RUN pip install numba==0.48 torch==1.6.0 torchvision

EXPOSE 5002

ENTRYPOINT [ "/usr/bin/python3", "-m", "TTS.server.server" ]
CMD [ "--use_cuda", "1", "--tts_checkpoint", "/model/tts.pth.tar", "--tts_config", "/model/tts.json", "--vocoder_checkpoint", "/model/vocoder.pth.tar", "--vocoder_config", "/model/vocoder.json"] 

