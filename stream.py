#!/usr/bin/env python3

import argparse

from tts import make_synthesizer

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from asyncio import Queue, sleep
import asyncio
from threading import Thread

import numpy as np

def get_app():
    app = FastAPI()
    parser = argparse.ArgumentParser(description="""Synthesize speech on command line.""")
    parser.add_argument("--use_cuda", type=bool, help="Run model on CUDA.", default=False)
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )

    args = parser.parse_args()

    app.synthesizer = make_synthesizer(args.model_name, args.use_cuda)
    app.audio_queues = set()
    app.cancel = asyncio.Event()
    print(f"Model has sample rate {app.synthesizer.output_sample_rate}")
    return app

app = get_app()

COMMIT_MS = 300

async def drain(queue, cancel, sr):
    chunks = int((COMMIT_MS / 1000.0) * sr)
    while True:
        new_data = await queue.get()
        if cancel.is_set():
            continue

        wav = np.array(new_data)
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        
        parts = np.array_split(wav_norm, len(wav_norm) / chunks)

        for p in parts:
            if cancel.is_set():
                break
            yield p.astype(np.int16).tobytes(order='C')
            await sleep(((COMMIT_MS - 100) / 1000.0))

@app.get('/status')
async def status():
    return {'status': 'ok'}

@app.get('/audio')
async def audio(request: Request):
    synthesizer = request.app.synthesizer

    queue = Queue()
    request.app.audio_queues.add(queue)    
    return StreamingResponse(drain(queue, request.app.cancel, synthesizer.output_sample_rate), media_type="audio/wav")

@app.get('/cancel')
async def cancel(request: Request):
    cancel = request.app.cancel

    cancel.set()
    for q in request.app.audio_queues:
        for _ in range(q.qsize()):
            q.get_nowait()
            q.task_done()
    
    await sleep(((COMMIT_MS - 100) / 1000.0))
    cancel.clear()
    return {'cancel': 'ok'}

@app.get('/say')
async def say(text: str, request: Request):
    synthesizer = request.app.synthesizer
    cancel = request.app.cancel
    sens = synthesizer.split_into_sentences(text)
    finished = asyncio.Event()

    async def done():
        finished.set()

    async def post(wav):
        for q in request.app.audio_queues:
            if cancel.is_set():
                return

            q.put_nowait(wav)

    def side_thread(loop):
        try:
            for l in sens:
                wav = synthesizer.tts(l)
                if finished.is_set():
                    print('! Aborting synthesizer')
                    return

                asyncio.run_coroutine_threadsafe(post(wav), loop)
        finally:        
            asyncio.run_coroutine_threadsafe(done(), loop)
        
    
    loop = asyncio.get_event_loop()
    thread = Thread(target=side_thread, args=(loop,), daemon=True)
    thread.start()
    
    done, pending = await asyncio.wait([finished.wait(), cancel.wait()], return_when = asyncio.FIRST_COMPLETED)
    finished.set()
    
    return {'say': 'ok'}

def main():
    uvicorn.run('stream:app', host='0.0.0.0', port=5002, reload=False)


if __name__ == "__main__":
    main()    