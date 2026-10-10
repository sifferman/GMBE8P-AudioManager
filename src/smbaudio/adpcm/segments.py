from dataclasses import dataclass

import numpy


def pad_to_multiple(values: numpy.ndarray, multiple: int) -> numpy.ndarray:
    missing_count = -len(values) % multiple
    padding = numpy.zeros((missing_count, *values.shape[1:]), dtype=values.dtype)
    return numpy.concatenate([values, padding])


def previous_two_samples(samples: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    silence = numpy.zeros((2, *samples.shape[1:]), dtype=samples.dtype)
    preceded_by_silence = numpy.concatenate([silence, samples])
    return preceded_by_silence[1:-1], preceded_by_silence[:-2]


def frames_by_time(values_per_step: numpy.ndarray) -> numpy.ndarray:
    segments_first = numpy.moveaxis(values_per_step, -1, 0)
    return segments_first.reshape(-1, *segments_first.shape[2:])


@dataclass(frozen=True)
class Segmentation:
    frames_per_segment: int
    samples_per_frame: int

    @property
    def samples_per_segment(self) -> int:
        return self.frames_per_segment * self.samples_per_frame

    def frames_in_whole_segments(self, values_per_frame: numpy.ndarray) -> numpy.ndarray:
        return pad_to_multiple(values_per_frame, self.frames_per_segment)

    def samples_in_whole_segments(self, samples: numpy.ndarray) -> numpy.ndarray:
        return pad_to_multiple(samples, self.samples_per_segment)

    def samples_by_frame(self, samples: numpy.ndarray) -> numpy.ndarray:
        return samples.reshape(-1, self.samples_per_frame, *samples.shape[1:])

    def frames_by_step(self, values_per_frame: numpy.ndarray) -> numpy.ndarray:
        segmented = values_per_frame.reshape(-1, self.frames_per_segment, *values_per_frame.shape[1:])
        return numpy.ascontiguousarray(numpy.moveaxis(segmented, 0, -1))

    def candidates_by_step(self, values_per_candidate: numpy.ndarray) -> numpy.ndarray:
        return self.frames_by_step(numpy.moveaxis(values_per_candidate, 0, 1))

    def history_before_each_segment(self, history: numpy.ndarray) -> numpy.ndarray:
        newest, oldest = previous_two_samples(history)
        return numpy.stack(
            [
                numpy.moveaxis(newest[:: self.samples_per_segment], 0, -1),
                numpy.moveaxis(oldest[:: self.samples_per_segment], 0, -1),
            ]
        )
