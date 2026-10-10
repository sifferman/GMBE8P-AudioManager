from smbaudio.sound_effects.musyx_files import (
    DirectorySample,
    SampleCoding,
    SampleDirectory,
)


def test_serialized_sample_directory_parses_back_to_the_same_samples():
    sample_directory = SampleDirectory(
        (
            DirectorySample(7, 0, 60, 44100, 1000, 0, 0, SampleCoding(0x12, 0, 0, 0, tuple(range(16)))),
            DirectorySample(9, 608, 60, 22050, 5000, 14, 4986, SampleCoding(0x34, 0x56, -3, 4, tuple(range(-8, 8)))),
        )
    )
    assert SampleDirectory.parse(sample_directory.serialized()) == sample_directory
