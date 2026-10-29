"""Strict metadata and file integrity checks for a selected-CI teacher."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True)
class TeacherIdentity:
    source_sha256: str
    system_sha256: str
    root_id: int
    nalpha: int
    nbeta: int
    norbitals: int
    unit: str

    def validate(self) -> None:
        if not all(len(x) == 64 for x in (self.source_sha256, self.system_sha256)):
            raise ValueError("Expected full SHA256 source/system hashes.")
        if self.root_id < 0 or self.nalpha < 0 or self.nbeta < 0:
            raise ValueError("Invalid root or electron count.")
        if self.norbitals < max(self.nalpha, self.nbeta):
            raise ValueError("Not enough orbitals for the spin sector.")
        if self.unit not in ("bohr", "angstrom"):
            raise ValueError("Unit must be explicit: bohr or angstrom.")


def file_sha256(path: str | Path) -> str:
    """Hash a local source/artifact without modifying it."""
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
