from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

Level = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Finding:
    level: Level
    code: str
    message: str
    model: str | None = None
    file: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Result:
    findings: list[Finding] = field(default_factory=list)

    def extend(self, items: Iterable[Finding]) -> None:
        self.findings.extend(items)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    def ok(self, *, strict: bool = False) -> bool:
        if self.errors:
            return False
        return not (strict and self.warnings)

    def merge(self, other: "Result") -> "Result":
        return Result(findings=[*self.findings, *other.findings])
