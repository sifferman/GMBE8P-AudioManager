import numpy
from conftest import signal_to_noise_decibels, tone_with_noise

from smbaudio.sound_effects.dsp_adpcm import (
    FRAME_SIZE,
    EncodedDspAdpcm,
    decode_dsp_adpcm,
    decode_dsp_adpcm_streams,
    encoded_size_for_samples,
)
from smbaudio.sound_effects.dsp_adpcm_encoder import encode_dsp_adpcm


def decode_one_sample_at_a_time(encoded: EncodedDspAdpcm) -> numpy.ndarray:
    newest = oldest = 0
    decoded = []
    for frame_start in range(0, len(encoded.frame_bytes), FRAME_SIZE):
        frame = encoded.frame_bytes[frame_start : frame_start + FRAME_SIZE]
        pair_index, scale = frame[0] >> 4, frame[0] & 0xF
        for code_byte in frame[1:]:
            for code in (code_byte >> 4, code_byte & 0xF):
                signed_code = code - 16 if code >= 8 else code
                weighted = (
                    newest * encoded.coefficients[pair_index * 2] + oldest * encoded.coefficients[pair_index * 2 + 1]
                )
                sample = max(-0x8000, min(0x7FFF, (((signed_code << scale) << 11) + weighted + 1024) >> 11))
                newest, oldest = sample, newest
                decoded.append(sample)
    return numpy.array(decoded[: encoded.sample_count], dtype=numpy.int16)


def test_encoded_sound_effect_decodes_close_to_the_wanted_samples():
    wanted = tone_with_noise(20001, 22050, 700, seed=3)
    encoded = encode_dsp_adpcm(wanted)
    assert len(encoded.frame_bytes) == encoded_size_for_samples(len(wanted))
    assert signal_to_noise_decibels(wanted, decode_dsp_adpcm(encoded)) > 30


def test_segmented_decoder_matches_decoding_one_sample_at_a_time():
    encoded = encode_dsp_adpcm(tone_with_noise(9000, 22050, 300, seed=4))
    assert numpy.array_equal(decode_dsp_adpcm(encoded), decode_one_sample_at_a_time(encoded))


def test_streams_decoded_together_do_not_affect_each_other():
    encoded_streams = [
        encode_dsp_adpcm(tone_with_noise(sample_count, 22050, 500, seed=sample_count))
        for sample_count in (5000, 13, 2000)
    ]
    for encoded, decoded in zip(encoded_streams, decode_dsp_adpcm_streams(encoded_streams)):
        assert numpy.array_equal(decoded, decode_dsp_adpcm(encoded))


def test_silence_encodes_to_silence():
    assert not decode_dsp_adpcm(encode_dsp_adpcm(numpy.zeros(500, dtype=numpy.int16))).any()
