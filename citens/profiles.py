"""Domain profiles: curated terminology + venue whitelists per field.

A profile is pure data loaded from JSON (packaged defaults under
``citens/profiles/``, overridable by ``data/profiles/<name>.json``). It
injects three things into the generic pipeline:

* ``domain_terms`` — established field terminology the planner may miss;
  appended to the keyword batches (and searched field-constrained when too
  generic to free-search).
* ``venue_whitelist`` — the field's flagship journals (FT50 core for
  finance); a venue match boosts the ranking's venue factor to Q1-level.
* ``subfields`` — OpenAlex subfield names used to interpret pool stats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    name: str
    domain_terms: list[str] = field(default_factory=list)
    venue_whitelist: list[str] = field(default_factory=list)
    subfields: list[str] = field(default_factory=list)

    def venue_boost_set(self) -> set[str]:
        from citens.ranking import _norm

        return {_norm(v) for v in self.venue_whitelist if v}


def profile_paths(name: str) -> list[Path]:
    """User override first, then the packaged default."""
    return [
        Path("data/profiles") / f"{name}.json",
        Path(__file__).parent / "profiles" / f"{name}.json",
    ]


def available_profiles() -> list[str]:
    d = Path(__file__).parent / "profiles"
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def load_profile(name: str) -> Profile | None:
    """Load a profile by name; None when it doesn't exist."""
    if not name:
        return None
    for path in profile_paths(name):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        return Profile(
            name=name,
            domain_terms=[str(t) for t in data.get("domain_terms", [])],
            venue_whitelist=[str(v) for v in data.get("venue_whitelist", [])],
            subfields=[str(s) for s in data.get("subfields", [])],
        )
    return None


def merge_profile_terms(queries: list[str], profile: Profile | None) -> list[str]:
    """Append profile domain terms to a query list, deduplicated.

    Returns (merged, added_terms) so the caller can treat curated single
    concepts like seed-agent terms (field-constrained search).
    """
    if profile is None:
        return queries
    known = {q.lower() for q in queries}
    added = [t for t in profile.domain_terms if t.lower() not in known]
    return queries + added
