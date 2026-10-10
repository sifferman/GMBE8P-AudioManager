from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy

from smbaudio.adpcm.segments import Segmentation, frames_by_time, previous_two_samples

MINIMUM_CODE = -8
MAXIMUM_CODE = 7
MAXIMUM_SCALE = 12


class AdpcmArithmetic(ABC):
    prediction_fraction_bits: int

    @abstractmethod
    def predict(
        self,
        newest_history: numpy.ndarray,
        oldest_history: numpy.ndarray,
        newest_coefficient: numpy.ndarray,
        oldest_coefficient: numpy.ndarray,
    ) -> numpy.ndarray: ...

    @abstractmethod
    def history_after_code(self, scaled_code: numpy.ndarray, prediction: numpy.ndarray) -> numpy.ndarray: ...

    @abstractmethod
    def history_from_samples(self, samples: numpy.ndarray) -> numpy.ndarray: ...

    @abstractmethod
    def samples_from_history(self, history: numpy.ndarray) -> numpy.ndarray: ...


class SegmentedAdpcmDecoder:
    def __init__(
        self,
        arithmetic: AdpcmArithmetic,
        segmentation: Segmentation,
        scaled_codes_per_frame: numpy.ndarray,
        newest_coefficient_per_frame: numpy.ndarray,
        oldest_coefficient_per_frame: numpy.ndarray,
        first_segment_of_each_stream: numpy.ndarray,
    ):
        self._arithmetic = arithmetic
        self._segmentation = segmentation
        self._scaled_codes = segmentation.frames_by_step(scaled_codes_per_frame)
        self._newest_coefficients = segmentation.frames_by_step(newest_coefficient_per_frame)
        self._oldest_coefficients = segmentation.frames_by_step(oldest_coefficient_per_frame)
        self._continues_previous_segment = numpy.ones(self._scaled_codes.shape[-1], dtype=bool)
        self._continues_previous_segment[first_segment_of_each_stream] = False

    def decoded_samples(self) -> numpy.ndarray:
        history_per_frame = frames_by_time(self._history_decoded_until_segments_join_up())
        history = history_per_frame.reshape(-1, *history_per_frame.shape[2:])
        return self._arithmetic.samples_from_history(history)

    def _history_decoded_until_segments_join_up(self) -> numpy.ndarray:
        starting_history = numpy.zeros((2, *self._scaled_codes.shape[2:]), dtype=numpy.int64)
        while True:
            decoded_history, ending_history = self._decode_segments(starting_history)
            carried_history = numpy.where(self._continues_previous_segment, numpy.roll(ending_history, 1, axis=-1), 0)
            if numpy.array_equal(carried_history, starting_history):
                return decoded_history
            starting_history = carried_history

    def _decode_segments(self, starting_history: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        newest_history, oldest_history = starting_history
        decoded_history = numpy.empty_like(self._scaled_codes)
        for frame_index in range(self._segmentation.frames_per_segment):
            for sample_index in range(self._segmentation.samples_per_frame):
                prediction = self._arithmetic.predict(
                    newest_history,
                    oldest_history,
                    self._newest_coefficients[frame_index],
                    self._oldest_coefficients[frame_index],
                )
                oldest_history = newest_history
                newest_history = self._arithmetic.history_after_code(
                    self._scaled_codes[frame_index, sample_index], prediction
                )
                decoded_history[frame_index, sample_index] = newest_history
        return decoded_history, numpy.stack([newest_history, oldest_history])


@dataclass(frozen=True)
class EncodedFrames:
    predictor_indices: numpy.ndarray
    scales: numpy.ndarray
    codes: numpy.ndarray


@dataclass(frozen=True)
class FrameCandidates:
    predictor_indices: numpy.ndarray
    scales: numpy.ndarray

    def chosen(self, chosen_candidates: numpy.ndarray) -> "FrameCandidates":
        return FrameCandidates(
            _chosen_per_frame(self.predictor_indices, chosen_candidates),
            _chosen_per_frame(self.scales, chosen_candidates),
        )


def _chosen_per_frame(values_per_candidate: numpy.ndarray, chosen_candidates: numpy.ndarray) -> numpy.ndarray:
    return numpy.take_along_axis(values_per_candidate, chosen_candidates[:, numpy.newaxis], axis=1)[:, 0]


def _chosen_per_stream(values_per_candidate: numpy.ndarray, chosen_candidates: numpy.ndarray) -> numpy.ndarray:
    return numpy.take_along_axis(values_per_candidate, chosen_candidates[numpy.newaxis], axis=0)[0]


def _finest_scale_that_fits(largest_difference: numpy.ndarray, largest_code: int) -> numpy.ndarray:
    doublings_needed = numpy.ceil(numpy.log2(numpy.maximum(largest_difference / largest_code, 1)))
    return doublings_needed.astype(numpy.int64)


class SegmentedAdpcmEncoder:
    def __init__(
        self,
        arithmetic: AdpcmArithmetic,
        segmentation: Segmentation,
        newest_coefficient_per_predictor: numpy.ndarray,
        oldest_coefficient_per_predictor: numpy.ndarray,
        scale_offsets_to_try: tuple[int, ...],
    ):
        self._arithmetic = arithmetic
        self._segmentation = segmentation
        self._newest_coefficients = newest_coefficient_per_predictor
        self._oldest_coefficients = oldest_coefficient_per_predictor
        self._scale_offsets_to_try = scale_offsets_to_try

    def encode(self, samples: numpy.ndarray) -> EncodedFrames:
        wanted_samples = self._segmentation.samples_in_whole_segments(samples.astype(numpy.int64))
        history = self._arithmetic.history_from_samples(wanted_samples)
        candidates = self._candidates_by_step(wanted_samples, history)
        chosen_candidates, chosen_codes = self._choose_best_candidate_per_frame(
            self._segmentation.frames_by_step(self._segmentation.samples_by_frame(wanted_samples)),
            candidates,
            self._segmentation.history_before_each_segment(history),
        )
        chosen = candidates.chosen(chosen_candidates)
        return EncodedFrames(
            frames_by_time(chosen.predictor_indices), frames_by_time(chosen.scales), frames_by_time(chosen_codes)
        )

    def _candidates_by_step(self, wanted_samples: numpy.ndarray, history: numpy.ndarray) -> FrameCandidates:
        fraction_bits = self._arithmetic.prediction_fraction_bits
        newest_history, oldest_history = previous_two_samples(history)
        candidate_predictor_indices = []
        candidate_scales = []
        for predictor_index in range(len(self._newest_coefficients)):
            prediction = self._arithmetic.predict(
                newest_history,
                oldest_history,
                self._newest_coefficients[predictor_index],
                self._oldest_coefficients[predictor_index],
            )
            differences = numpy.abs((wanted_samples << fraction_bits) - prediction)
            finest_scale = _finest_scale_that_fits(
                self._segmentation.samples_by_frame(differences).max(axis=1), MAXIMUM_CODE << fraction_bits
            )
            for scale_offset in self._scale_offsets_to_try:
                candidate_predictor_indices.append(numpy.full_like(finest_scale, predictor_index))
                candidate_scales.append(numpy.clip(finest_scale + scale_offset, 0, MAXIMUM_SCALE))
        return FrameCandidates(
            self._segmentation.candidates_by_step(numpy.stack(candidate_predictor_indices)),
            self._segmentation.candidates_by_step(numpy.stack(candidate_scales)),
        )

    def _choose_best_candidate_per_frame(
        self, wanted_samples: numpy.ndarray, candidates: FrameCandidates, starting_history: numpy.ndarray
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        frames_per_segment, samples_per_frame, *stream_shape = wanted_samples.shape
        candidate_shape = candidates.scales.shape[1:]
        newest_coefficients = self._newest_coefficients[candidates.predictor_indices]
        oldest_coefficients = self._oldest_coefficients[candidates.predictor_indices]
        code_bits = candidates.scales + self._arithmetic.prediction_fraction_bits
        chosen_candidates = numpy.empty((frames_per_segment, *stream_shape), dtype=numpy.intp)
        chosen_codes = numpy.empty(wanted_samples.shape, dtype=numpy.int8)
        candidate_codes = numpy.empty((samples_per_frame, *candidate_shape), dtype=numpy.int8)
        newest_history, oldest_history = starting_history

        for frame_index in range(frames_per_segment):
            candidate_newest = numpy.broadcast_to(newest_history, candidate_shape)
            candidate_oldest = numpy.broadcast_to(oldest_history, candidate_shape)
            squared_error = numpy.zeros(candidate_shape, dtype=numpy.int64)
            for sample_index in range(samples_per_frame):
                wanted = wanted_samples[frame_index, sample_index]
                codes, following_newest, decoded = self._quantize_sample(
                    wanted,
                    candidate_newest,
                    candidate_oldest,
                    newest_coefficients[frame_index],
                    oldest_coefficients[frame_index],
                    code_bits[frame_index],
                )
                candidate_oldest = candidate_newest
                candidate_newest = following_newest
                error = decoded.astype(numpy.int64) - wanted
                squared_error += error * error
                candidate_codes[sample_index] = codes
            best_candidates = squared_error.argmin(axis=0)
            newest_history = _chosen_per_stream(candidate_newest, best_candidates)
            oldest_history = _chosen_per_stream(candidate_oldest, best_candidates)
            chosen_candidates[frame_index] = best_candidates
            chosen_codes[frame_index] = numpy.take_along_axis(
                candidate_codes, best_candidates[numpy.newaxis, numpy.newaxis], axis=1
            )[:, 0]
        return chosen_candidates, chosen_codes

    def _quantize_sample(
        self,
        wanted_sample: numpy.ndarray,
        newest_history: numpy.ndarray,
        oldest_history: numpy.ndarray,
        newest_coefficient: numpy.ndarray,
        oldest_coefficient: numpy.ndarray,
        code_bits: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        prediction = self._arithmetic.predict(newest_history, oldest_history, newest_coefficient, oldest_coefficient)
        difference = (wanted_sample << self._arithmetic.prediction_fraction_bits) - prediction
        codes = numpy.clip((difference + (1 << (code_bits - 1))) >> code_bits, MINIMUM_CODE, MAXIMUM_CODE)
        following_history = self._arithmetic.history_after_code(codes << code_bits, prediction)
        return codes, following_history, self._arithmetic.samples_from_history(following_history)
