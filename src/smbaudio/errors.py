class AudioManagerError(Exception):
    pass


class InvalidReplacementsError(AudioManagerError):
    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems
