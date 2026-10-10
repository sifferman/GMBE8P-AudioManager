import numpy
from conftest import signal_to_noise_decibels, tone_with_noise

from smbaudio.music.dtk_adpcm import (
    FILE_SIZE_MULTIPLE,
    FRAME_SIZE,
    SAMPLES_PER_FRAME,
    decode_dtk_adpcm,
    encode_dtk_adpcm,
)

FIRST_COEFFICIENTS = (0, 0x3C, 0x73, 0x62)
SECOND_COEFFICIENTS = (0, 0, 0x34, 0x37)


def stereo_test_signal(sample_count: int) -> numpy.ndarray:
    return numpy.stack(
        [tone_with_noise(sample_count, 48000, 440, seed=1), tone_with_noise(sample_count, 48000, 1234, seed=2)], axis=1
    )


def decode_one_sample_at_a_time(adpcm_bytes: bytes) -> numpy.ndarray:
    histories = [[0, 0], [0, 0]]
    decoded = []
    for frame_start in range(0, len(adpcm_bytes), FRAME_SIZE):
        frame = adpcm_bytes[frame_start : frame_start + FRAME_SIZE]
        for code_byte in frame[FRAME_SIZE - SAMPLES_PER_FRAME :]:
            both_channels = []
            for channel, code in enumerate((code_byte & 0xF, code_byte >> 4)):
                filter_index, shift = frame[channel] >> 4, frame[channel] & 0xF
                newest, oldest = histories[channel]
                weighted = newest * FIRST_COEFFICIENTS[filter_index] - oldest * SECOND_COEFFICIENTS[filter_index]
                prediction = max(-0x200000, min(0x1FFFFF, (weighted + 0x20) >> 6))
                signed_code = code - 16 if code >= 8 else code
                current = (((signed_code << 12) >> shift) << 6) + prediction
                histories[channel] = [current, newest]
                both_channels.append(max(-0x8000, min(0x7FFF, current >> 6)))
            decoded.append(both_channels)
    return numpy.array(decoded, dtype=numpy.int16)


def test_encoded_music_decodes_close_to_the_wanted_samples():
    wanted = stereo_test_signal(30000)
    decoded = decode_dtk_adpcm(encode_dtk_adpcm(wanted))
    assert signal_to_noise_decibels(wanted, decoded[: len(wanted)]) > 30


def test_encoded_music_size_is_a_multiple_of_32_kilobytes():
    assert len(encode_dtk_adpcm(stereo_test_signal(30000))) % FILE_SIZE_MULTIPLE == 0


def test_segmented_decoder_matches_decoding_one_sample_at_a_time():
    encoded = encode_dtk_adpcm(stereo_test_signal(20000))
    assert numpy.array_equal(decode_dtk_adpcm(encoded), decode_one_sample_at_a_time(encoded))


def test_encoding_decoded_music_reproduces_it():
    first_generation = decode_dtk_adpcm(encode_dtk_adpcm(stereo_test_signal(30000)))
    second_generation = decode_dtk_adpcm(encode_dtk_adpcm(first_generation))
    assert signal_to_noise_decibels(first_generation, second_generation) > 80
