from smbaudio.sound_effects.sound_bank_rebuilding import AUDIO_MEMORY_BUDGET, AudioMemoryUse


def test_banks_that_fit_together_are_accepted():
    assert AudioMemoryUse({"allse": 0x200000, "comn": 0x200000, "golf": 0x200000}).problems == []


def test_minigame_bank_that_overflows_audio_memory_is_reported():
    problems = AudioMemoryUse({"allse": 0x200000, "comn": 0x200000, "golf": 0x300000, "race": 0x100000}).problems
    assert len(problems) == 1 and "golf" in problems[0]


def test_always_loaded_banks_alone_can_overflow_audio_memory():
    assert AudioMemoryUse({"allse": AUDIO_MEMORY_BUDGET, "comn": 0x20}).problems
