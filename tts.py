from pathlib import Path
import TTS
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager


def make_synthesizer(model_name, use_cuda):
    # load model manager
    path = Path(TTS.__file__).parent / ".models.json"
    manager = ModelManager(path)

    model_path, config_path, model_item = manager.download_model(model_name)
    vocoder_name = model_item["default_vocoder"]
    vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

    speakers_file_path = None
    encoder_path = None
    encoder_config_path = None
    
    return Synthesizer(
        model_path,
        config_path,
        speakers_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        use_cuda,
    )
