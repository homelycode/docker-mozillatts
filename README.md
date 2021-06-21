# Docker Image for TTS

Build from mozilla TTS

docker run --runtime=nvidia -p 5002:5002 docker.pkg.github.com/homelycode/docker-mozillatts/tts:latest

# Example Playback

SDL_AUDIODRIVER="alsa" AUDIODEV="hw:Loopback,0" ffplay -nodisp -f s16le -ar 22050 -ac 1 http://192.168.1.181:5002/audio

# Notes

docker run -ti --runtime=nvidia -p 5002:5002 --entrypoint=/bin/bash -v $(pwd):/app test

Package in /usr/local/lib/python3.6/dist-packages/TTS patched

Model from https://github.com/mozilla/TTS/issues/512

python -m TTS.server.server --use_cuda 1 --tts_checkpoint /model/tts.pth.tar --tts_config /model/tts.json --vocoder_checkpoint /model/vocoder.pth.tar --vocoder_config /model/vocoder.json