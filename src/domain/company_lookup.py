from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CaenIndustryEntry:
    caen_code: str
    caen_description: Optional[str]
    industry: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CompanyIndustryResult:
    input_name: str
    source: str  # onrc | termene | not_found
    company_name: Optional[str] = None
    cui: Optional[str] = None
    cod_inmatriculare: Optional[str] = None
    entries: List[CaenIndustryEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload

