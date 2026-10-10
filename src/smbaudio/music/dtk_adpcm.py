import numpy

from smbaudio.adpcm.segmented_codec import (
    MAXIMUM_SCALE,
    AdpcmArithmetic,
    EncodedFrames,
    SegmentedAdpcmDecoder,
    SegmentedAdpcmEncoder,
)
from smbaudio.adpcm.segments import Segmentation

SAMPLE_RATE = 48000
CHANNEL_COUNT = 2
FRAME_SIZE = 32
SAMPLES_PER_FRAME = 28
FILE_SIZE_MULTIPLE = 0x8000
FILTER_COUNT = 4
HISTORY_FRACTION_BITS = 6
MINIMUM_PREDICTION = -0x200000
MAXIMUM_PREDICTION = 0x1FFFFF

NEWEST_HISTORY_COEFFICIENTS = numpy.array([0, 0x3C, 0x73, 0x62] + [0] * 12, dtype=numpy.int64)
OLDEST_HISTORY_COEFFICIENTS = numpy.array([0, 0, 0x34, 0x37] + [0] * 12, dtype=numpy.int64)
FRAME_HEADER_SIZE = FRAME_SIZE - SAMPLES_PER_FRAME
DECODING_SEGMENTATION = Segmentation(frames_per_segment=128, samples_per_frame=SAMPLES_PER_FRAME)


def frame_count_for_samples(sample_count: int) -> int:
    return -(-sample_count // SAMPLES_PER_FRAME)


def encoded_size_for_samples(sample_count: int) -> int:
    used_size = frame_count_for_samples(sample_count) * FRAME_SIZE
    return used_size + -used_size % FILE_SIZE_MULTIPLE


class DtkAdpcmArithmetic(AdpcmArithmetic):
    prediction_fraction_bits = HISTORY_FRACTION_BITS

    def predict(
        self,
        newest_history: numpy.ndarray,
        oldest_history: numpy.ndarray,
        newest_coefficient: numpy.ndarray,
        oldest_coefficient: numpy.ndarray,
    ) -> numpy.ndarray:
        weighted_history = newest_history * newest_coefficient - oldest_history * oldest_coefficient
        rounded_prediction = (weighted_history + (1 << (HISTORY_FRACTION_BITS - 1))) >> HISTORY_FRACTION_BITS
        return numpy.clip(rounded_prediction, MINIMUM_PREDICTION, MAXIMUM_PREDICTION)

    def history_after_code(self, scaled_code: numpy.ndarray, prediction: numpy.ndarray) -> numpy.ndarray:
        return scaled_code + prediction

    def history_from_samples(self, samples: numpy.ndarray) -> numpy.ndarray:
        return samples << HISTORY_FRACTION_BITS

    def samples_from_history(self, history: numpy.ndarray) -> numpy.ndarray:
        return numpy.clip(history >> HISTORY_FRACTION_BITS, -0x8000, 0x7FFF).astype(numpy.int16)


DTK_ADPCM_ARITHMETIC = DtkAdpcmArithmetic()


class DtkAdpcmFrames:
    def __init__(self, frame_bytes: numpy.ndarray):
        self.frame_bytes = frame_bytes

    @classmethod
    def from_file_bytes(cls, adpcm_bytes: bytes) -> "DtkAdpcmFrames":
        return cls(numpy.frombuffer(adpcm_bytes, dtype=numpy.uint8).reshape(-1, FRAME_SIZE))

    @classmethod
    def from_encoded_frames(cls, encoded: EncodedFrames) -> "DtkAdpcmFrames":
        shifts = MAXIMUM_SCALE - encoded.scales
        headers = ((encoded.predictor_indices << 4) | shifts).astype(numpy.uint8)
        code_nibbles = encoded.codes.astype(numpy.uint8) & 0xF
        code_bytes = code_nibbles[..., 0] | (code_nibbles[..., 1] << 4)
        return cls(numpy.concatenate([headers, headers, code_bytes], axis=1))

    @property
    def sample_count(self) -> int:
        return len(self.frame_bytes) * SAMPLES_PER_FRAME

    @property
    def filters(self) -> numpy.ndarray:
        return self.frame_bytes[:, :CHANNEL_COUNT] >> 4

    @property
    def shifts(self) -> numpy.ndarray:
        return (self.frame_bytes[:, :CHANNEL_COUNT] & 0xF).astype(numpy.int64)

    @property
    def signed_codes(self) -> numpy.ndarray:
        code_bytes = self.frame_bytes[:, FRAME_HEADER_SIZE:].astype(numpy.int64)
        left_and_right_codes = numpy.stack([code_bytes & 0xF, code_bytes >> 4], axis=-1)
        return (left_and_right_codes ^ 8) - 8

    @property
    def scaled_codes(self) -> numpy.ndarray:
        shifted_codes = (self.signed_codes << MAXIMUM_SCALE) >> self.shifts[:, numpy.newaxis, :]
        return shifted_codes << HISTORY_FRACTION_BITS

    def to_file_bytes(self, file_size: int) -> bytes:
        return self.frame_bytes.tobytes()[:file_size].ljust(file_size, b"\0")


def decode_dtk_adpcm(adpcm_bytes: bytes) -> numpy.ndarray:
    stored_frames = DtkAdpcmFrames.from_file_bytes(adpcm_bytes)
    frames = DtkAdpcmFrames(DECODING_SEGMENTATION.frames_in_whole_segments(stored_frames.frame_bytes))
    decoder = SegmentedAdpcmDecoder(
        DTK_ADPCM_ARITHMETIC,
        DECODING_SEGMENTATION,
        frames.scaled_codes,
        NEWEST_HISTORY_COEFFICIENTS[frames.filters],
        OLDEST_HISTORY_COEFFICIENTS[frames.filters],
        first_segment_of_each_stream=numpy.array([0]),
    )
    return decoder.decoded_samples()[: stored_frames.sample_count]


ENCODER = SegmentedAdpcmEncoder(
    DTK_ADPCM_ARITHMETIC,
    Segmentation(frames_per_segment=256, samples_per_frame=SAMPLES_PER_FRAME),
    NEWEST_HISTORY_COEFFICIENTS[:FILTER_COUNT],
    OLDEST_HISTORY_COEFFICIENTS[:FILTER_COUNT],
    scale_offsets_to_try=(1, 0, -1),
)


def encode_dtk_adpcm(samples: numpy.ndarray) -> bytes:
    frames = DtkAdpcmFrames.from_encoded_frames(ENCODER.encode(samples))
    return frames.to_file_bytes(encoded_size_for_samples(len(samples)))
