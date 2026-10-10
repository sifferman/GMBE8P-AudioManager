from dataclasses import dataclass

import numpy

from smbaudio.adpcm.segmented_codec import AdpcmArithmetic, EncodedFrames, SegmentedAdpcmDecoder
from smbaudio.adpcm.segments import Segmentation

FRAME_SIZE = 8
SAMPLES_PER_FRAME = 14
COEFFICIENT_PAIR_COUNT = 8
COEFFICIENT_FRACTION_BITS = 11
SEGMENTATION = Segmentation(frames_per_segment=64, samples_per_frame=SAMPLES_PER_FRAME)


def frame_count_for_samples(sample_count: int) -> int:
    return -(-sample_count // SAMPLES_PER_FRAME)


def encoded_size_for_samples(sample_count: int) -> int:
    return frame_count_for_samples(sample_count) * FRAME_SIZE


class DspAdpcmArithmetic(AdpcmArithmetic):
    prediction_fraction_bits = COEFFICIENT_FRACTION_BITS

    def predict(
        self,
        newest_history: numpy.ndarray,
        oldest_history: numpy.ndarray,
        newest_coefficient: numpy.ndarray,
        oldest_coefficient: numpy.ndarray,
    ) -> numpy.ndarray:
        return newest_history * newest_coefficient + oldest_history * oldest_coefficient

    def history_after_code(self, scaled_code: numpy.ndarray, prediction: numpy.ndarray) -> numpy.ndarray:
        rounding = 1 << (COEFFICIENT_FRACTION_BITS - 1)
        return numpy.clip((scaled_code + prediction + rounding) >> COEFFICIENT_FRACTION_BITS, -0x8000, 0x7FFF)

    def history_from_samples(self, samples: numpy.ndarray) -> numpy.ndarray:
        return samples

    def samples_from_history(self, history: numpy.ndarray) -> numpy.ndarray:
        return history.astype(numpy.int16)


DSP_ADPCM_ARITHMETIC = DspAdpcmArithmetic()


class DspAdpcmFrames:
    def __init__(self, frame_bytes: numpy.ndarray):
        self.frame_bytes = frame_bytes

    @classmethod
    def from_encoded_frames(cls, encoded: EncodedFrames) -> "DspAdpcmFrames":
        headers = ((encoded.predictor_indices << 4) | encoded.scales).astype(numpy.uint8)
        code_nibbles = encoded.codes.astype(numpy.uint8) & 0xF
        code_bytes = (code_nibbles[:, 0::2] << 4) | code_nibbles[:, 1::2]
        return cls(numpy.concatenate([headers[:, numpy.newaxis], code_bytes], axis=1))

    @property
    def coefficient_pair_indices(self) -> numpy.ndarray:
        return (self.frame_bytes[:, 0] >> 4) % COEFFICIENT_PAIR_COUNT

    @property
    def scaled_codes(self) -> numpy.ndarray:
        code_bytes = self.frame_bytes[:, 1:].astype(numpy.int64)
        codes = numpy.stack([code_bytes >> 4, code_bytes & 0xF], axis=-1).reshape(-1, SAMPLES_PER_FRAME)
        scales = (self.frame_bytes[:, :1] & 0xF).astype(numpy.int64)
        return (((codes ^ 8) - 8) << scales) << COEFFICIENT_FRACTION_BITS


@dataclass(frozen=True)
class EncodedDspAdpcm:
    frame_bytes: bytes
    coefficients: tuple[int, ...]
    sample_count: int

    @property
    def frames(self) -> DspAdpcmFrames:
        used_bytes = numpy.frombuffer(
            self.frame_bytes, dtype=numpy.uint8, count=encoded_size_for_samples(self.sample_count)
        )
        return DspAdpcmFrames(used_bytes.reshape(-1, FRAME_SIZE))

    @property
    def coefficient_pairs(self) -> numpy.ndarray:
        return numpy.array(self.coefficients, dtype=numpy.int64).reshape(COEFFICIENT_PAIR_COUNT, 2)


def _frames_in_whole_segments(encoded: EncodedDspAdpcm) -> DspAdpcmFrames:
    return DspAdpcmFrames(SEGMENTATION.frames_in_whole_segments(encoded.frames.frame_bytes))


def decode_dsp_adpcm_streams(encoded_streams: list[EncodedDspAdpcm]) -> list[numpy.ndarray]:
    if not encoded_streams:
        return []
    frames_per_stream = [_frames_in_whole_segments(encoded) for encoded in encoded_streams]
    coefficient_pair_per_frame = numpy.concatenate(
        [
            encoded.coefficient_pairs[frames.coefficient_pair_indices]
            for encoded, frames in zip(encoded_streams, frames_per_stream)
        ]
    )
    segment_counts = [len(frames.frame_bytes) // SEGMENTATION.frames_per_segment for frames in frames_per_stream]
    first_segment_of_each_stream = numpy.cumsum([0] + segment_counts)[:-1]
    decoder = SegmentedAdpcmDecoder(
        DSP_ADPCM_ARITHMETIC,
        SEGMENTATION,
        numpy.concatenate([frames.scaled_codes for frames in frames_per_stream]),
        coefficient_pair_per_frame[:, 0],
        coefficient_pair_per_frame[:, 1],
        first_segment_of_each_stream,
    )
    samples = decoder.decoded_samples()
    return [
        samples[first_segment * SEGMENTATION.samples_per_segment :][: encoded.sample_count]
        for encoded, first_segment in zip(encoded_streams, first_segment_of_each_stream)
    ]


def decode_dsp_adpcm(encoded: EncodedDspAdpcm) -> numpy.ndarray:
    return decode_dsp_adpcm_streams([encoded])[0]
