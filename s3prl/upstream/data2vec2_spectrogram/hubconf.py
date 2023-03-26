from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def data2vec2_spectrogram_custom(ckpt: str, refresh: bool = False, **kwargs):
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)

    return _UpstreamExpert(ckpt, **kwargs)


def data2vec2_spectrogram_local(*args, **kwargs):
    return data2vec2_spectrogram_custom(*args, **kwargs)
