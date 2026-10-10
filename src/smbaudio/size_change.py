from dataclasses import dataclass

BYTES_PER_MEGABYTE = 1 << 20
BYTES_PER_KILOBYTE = 1 << 10


def _readable_size(size: int) -> str:
    if size >= BYTES_PER_MEGABYTE:
        return f"{size / BYTES_PER_MEGABYTE:.1f} MB"
    return f"{size / BYTES_PER_KILOBYTE:.1f} KB"


@dataclass(frozen=True)
class SizeChange:
    name: str
    original_size: int
    replaced_size: int
    reason: str

    def __str__(self) -> str:
        return (
            f"{self.name}  {_readable_size(self.original_size)} -> {_readable_size(self.replaced_size)} ({self.reason})"
        )
