import torch
import torchaudio
from .klaam import TextToSpeech


def get_ts_model():
  return TextToSpeech()


def english_speech(device, processor, tacotron2, vocoder, text):
  torch.random.manual_seed(0)

  with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, spec_lengths)
    return waveforms

