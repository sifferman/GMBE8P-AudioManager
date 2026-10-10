from dataclasses import dataclass

import numpy

from smbaudio.adpcm.segmented_codec import SegmentedAdpcmEncoder
from smbaudio.adpcm.segments import previous_two_samples
from smbaudio.sound_effects.dsp_adpcm import (
    COEFFICIENT_FRACTION_BITS,
    COEFFICIENT_PAIR_COUNT,
    DSP_ADPCM_ARITHMETIC,
    SAMPLES_PER_FRAME,
    SEGMENTATION,
    DspAdpcmFrames,
    EncodedDspAdpcm,
    encoded_size_for_samples,
)

REFINEMENT_ROUNDS = 24
STABILITY_MARGIN = 0.995
SMALLEST_USABLE_DETERMINANT = 1e-6
SCALE_OFFSETS_TO_TRY = (-1, 0, 1)


def _summed_per_frame(products: numpy.ndarray) -> numpy.ndarray:
    return products.reshape(-1, SAMPLES_PER_FRAME).sum(axis=1)


def _stable_weight_pairs(newest_weight: numpy.ndarray, oldest_weight: numpy.ndarray) -> numpy.ndarray:
    stable_oldest_weight = numpy.clip(oldest_weight, -STABILITY_MARGIN, STABILITY_MARGIN)
    newest_weight_limit = (1 - stable_oldest_weight) * STABILITY_MARGIN
    stable_newest_weight = numpy.clip(newest_weight, -newest_weight_limit, newest_weight_limit)
    return numpy.stack([stable_newest_weight, stable_oldest_weight], axis=-1)


@dataclass(frozen=True)
class FrameCorrelations:
    newest_with_newest: numpy.ndarray
    newest_with_oldest: numpy.ndarray
    oldest_with_oldest: numpy.ndarray
    sample_with_newest: numpy.ndarray
    sample_with_oldest: numpy.ndarray

    @classmethod
    def of_samples(cls, samples: numpy.ndarray) -> "FrameCorrelations":
        current = samples.astype(numpy.float64)
        newest, oldest = previous_two_samples(current)
        return cls(
            _summed_per_frame(newest * newest),
            _summed_per_frame(newest * oldest),
            _summed_per_frame(oldest * oldest),
            _summed_per_frame(current * newest),
            _summed_per_frame(current * oldest),
        )

    def summed_per_group(self, group_per_frame: numpy.ndarray, group_count: int) -> "FrameCorrelations":
        return FrameCorrelations(
            *(
                numpy.bincount(group_per_frame, weights=correlation, minlength=group_count)
                for correlation in vars(self).values()
            )
        )

    def best_weight_pairs(self) -> numpy.ndarray:
        determinant = self.newest_with_newest * self.oldest_with_oldest - self.newest_with_oldest**2
        solvable = determinant > SMALLEST_USABLE_DETERMINANT
        safe_determinant = numpy.where(solvable, determinant, 1)
        newest_weight = (
            self.sample_with_newest * self.oldest_with_oldest - self.sample_with_oldest * self.newest_with_oldest
        ) / safe_determinant
        oldest_weight = (
            self.sample_with_oldest * self.newest_with_newest - self.sample_with_newest * self.newest_with_oldest
        ) / safe_determinant
        return _stable_weight_pairs(numpy.where(solvable, newest_weight, 0), numpy.where(solvable, oldest_weight, 0))

    def prediction_error_per_frame_and_pair(self, weight_pairs: numpy.ndarray) -> numpy.ndarray:
        newest_weight = weight_pairs[:, 0]
        oldest_weight = weight_pairs[:, 1]
        return (
            numpy.outer(self.newest_with_newest, newest_weight**2)
            + 2 * numpy.outer(self.newest_with_oldest, newest_weight * oldest_weight)
            + numpy.outer(self.oldest_with_oldest, oldest_weight**2)
            - 2 * numpy.outer(self.sample_with_newest, newest_weight)
            - 2 * numpy.outer(self.sample_with_oldest, oldest_weight)
        )

    def frames_grouped_by_newest_weight(self, group_count: int) -> numpy.ndarray:
        newest_weight_per_frame = self.best_weight_pairs()[:, 0]
        frame_rank = numpy.argsort(numpy.argsort(newest_weight_per_frame, kind="stable"), kind="stable")
        return frame_rank * group_count // max(len(frame_rank), 1)

    def best_weight_pair_per_group(self, group_per_frame: numpy.ndarray, group_count: int) -> numpy.ndarray:
        return self.summed_per_group(group_per_frame, group_count).best_weight_pairs()


def choose_coefficient_pairs(samples: numpy.ndarray) -> numpy.ndarray:
    correlations = FrameCorrelations.of_samples(samples)
    group_per_frame = correlations.frames_grouped_by_newest_weight(COEFFICIENT_PAIR_COUNT)
    for _ in range(REFINEMENT_ROUNDS):
        weight_pairs = correlations.best_weight_pair_per_group(group_per_frame, COEFFICIENT_PAIR_COUNT)
        group_per_frame = correlations.prediction_error_per_frame_and_pair(weight_pairs).argmin(axis=1)
    weight_pairs = correlations.best_weight_pair_per_group(group_per_frame, COEFFICIENT_PAIR_COUNT)
    return numpy.round(weight_pairs * (1 << COEFFICIENT_FRACTION_BITS)).astype(numpy.int64)


def encode_dsp_adpcm(samples: numpy.ndarray) -> EncodedDspAdpcm:
    wanted_samples = SEGMENTATION.samples_in_whole_segments(samples.astype(numpy.int64))
    coefficient_pairs = choose_coefficient_pairs(wanted_samples)
    encoder = SegmentedAdpcmEncoder(
        DSP_ADPCM_ARITHMETIC, SEGMENTATION, coefficient_pairs[:, 0], coefficient_pairs[:, 1], SCALE_OFFSETS_TO_TRY
    )
    frames = DspAdpcmFrames.from_encoded_frames(encoder.encode(wanted_samples))
    return EncodedDspAdpcm(
        frames.frame_bytes.tobytes()[: encoded_size_for_samples(len(samples))],
        tuple(int(coefficient) for coefficient in coefficient_pairs.reshape(-1)),
        len(samples),
    )
