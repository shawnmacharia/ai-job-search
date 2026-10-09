"""A conservative registry of sources, with access decided up front.

Why this exists
---------------
The vision asks for permitted sources and for inaccessible ones to be reported
rather than hidden. Nothing in the repository enforced either. The scraper
launched a browser and whatever happened happened, and a source that was never
consulted looked exactly like a source that returned nothing.

Two rules govern everything here:

**Fail closed.** A source is active only if it is ``enabled`` *and* its access
is explicitly ``permitted``. Anything else - ``unknown``, ``restricted``, or no
decision recorded at all - is skipped. There is no implicit allow.

**Never silently omit.** Every source that is not consulted appears in the
skipped report with a human-readable reason, and that report reaches the run
ledger. Silence is the failure mode this module exists to prevent.

Access decisions are declared here, verified elsewhere
------------------------------------------------------
The registry performs no robots.txt fetch, no HTTP request, no DNS lookup and
no network access of any kind. It enforces a decision recorded somewhere else.
That remains deliberate: an access check is a point-in-time observation that
belongs at the collection boundary, not inside a configuration object that may
be consulted in contexts where network access is unavailable or unwanted.

What changed is that ``permitted`` is no longer a bare assertion. It is backed
by :mod:`app.sources.access`, which fetches robots.txt and the terms page and
records an :class:`~app.sources.access.AccessDecision` with evidence.

A recorded decision is **necessary** for a source to be active, and
:func:`SourceRegistry.enforce_recorded_decisions` can require it to be
*present*, so a source cannot become permitted on nothing but a config edit.
See :mod:`app.sources.access` for why a permissive robots.txt is necessary but
not sufficient - a site may allow a path and still challenge the client that
asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from app.jobs.runner import (
    SKIP_ACCESS_NOT_PERMITTED,
    SKIP_ACCESS_RESTRICTED,
    SKIP_ACCESS_UNKNOWN,
    SKIP_DISABLED,
    SKIP_NO_FETCHER,
    SkippedSource,
    SourceSpec,
)


#: Access levels a source may declare.
PERMITTED = "permitted"
UNKNOWN = "unknown"
RESTRICTED = "restricted"

ACCESS_LEVELS = frozenset({PERMITTED, UNKNOWN, RESTRICTED})


class SourceConfigError(ValueError):
    """A source configuration is not usable as written."""


@dataclass(frozen=True)
class SourceConfig:
    """How one source is configured.

    ``access`` defaults to :data:`UNKNOWN`, so a source that is merely added is
    never active until someone states that access is permitted. That is the
    fail-closed default working as intended.
    """

    name: str
    enabled: bool = True
    access: str = UNKNOWN
    note: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise SourceConfigError("source name must not be empty")
        if self.access not in ACCESS_LEVELS:
            raise SourceConfigError(
                f"source {self.name!r} has invalid access {self.access!r}; "
                f"expected one of {sorted(ACCESS_LEVELS)}"
            )

    @property
    def is_permitted(self) -> bool:
        return self.access == PERMITTED

    @property
    def is_active(self) -> bool:
        """Only an enabled, explicitly permitted source may run."""
        return self.enabled and self.is_permitted

    def skip_reason(self) -> Optional[SkippedSource]:
        """Return why this source must not run, or ``None`` if it may."""
        if self.is_active:
            return None
        if not self.enabled:
            reason = self.note or "source is disabled in configuration"
            return SkippedSource(self.name, SKIP_DISABLED, reason)
        if self.access == RESTRICTED:
            reason = self.note or (
                "access is recorded as restricted; the site's terms or "
                "robots policy do not permit automated collection"
            )
            return SkippedSource(self.name, SKIP_ACCESS_RESTRICTED, reason)
        if self.access == UNKNOWN:
            reason = self.note or (
                "access has not been decided; fail closed until it is "
                "explicitly recorded as permitted"
            )
            return SkippedSource(self.name, SKIP_ACCESS_UNKNOWN, reason)
        return SkippedSource(
            self.name,
            SKIP_ACCESS_NOT_PERMITTED,
            self.note or f"access {self.access!r} does not permit collection",
        )


@dataclass(frozen=True)
class SourcePlan:
    """What a run should do: the specs to execute, and what it will skip.

    Both halves are always populated. ``specs`` alone would be a silent
    omission; ``skipped`` alone would be a report with nothing to run.
    """

    specs: List[SourceSpec] = field(default_factory=list)
    skipped: List[SkippedSource] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.specs

    def __bool__(self) -> bool:
        # An empty plan is falsy: `if plan:` reads as "there is work to do".
        return bool(self.specs)


class SourceRegistry:
    """Registered sources, in registration order.

    Order is insertion order and never re-sorted, so a run's source sequence is
    reproducible and a diff of the ledger is readable.
    """

    def __init__(self, configs: Iterable[SourceConfig] = ()) -> None:
        self._configs: Dict[str, SourceConfig] = {}
        for config in configs:
            self.register(config)

    def register(self, config: SourceConfig) -> SourceConfig:
        """Add a source. Duplicate names are rejected rather than overwritten."""
        if not isinstance(config, SourceConfig):
            raise SourceConfigError(f"expected a SourceConfig, got {type(config).__name__}")
        if config.name in self._configs:
            raise SourceConfigError(f"source {config.name!r} is already registered")
        self._configs[config.name] = config
        return config

    def __contains__(self, name: object) -> bool:
        return name in self._configs

    def __len__(self) -> int:
        return len(self._configs)

    def get(self, name: str) -> SourceConfig:
        try:
            return self._configs[name]
        except KeyError:
            raise SourceConfigError(f"unknown source: {name!r}") from None

    def configs(self) -> List[SourceConfig]:
        """Every registered source, in registration order."""
        return list(self._configs.values())

    def active(self) -> List[SourceConfig]:
        """Enabled and explicitly permitted sources, in registration order."""
        return [config for config in self._configs.values() if config.is_active]

    def skipped(self) -> List[SkippedSource]:
        """Every source that will not run, each with a reason."""
        return [
            skip
            for skip in (config.skip_reason() for config in self._configs.values())
            if skip is not None
        ]

    def plan(
        self, fetchers: Mapping[str, Callable[[], Iterable[Mapping[str, Any]]]]
    ) -> SourcePlan:
        """Build a run plan from a name -> fetch mapping.

        An active source with no fetcher is *skipped with a reason* rather than
        raising: a missing fetcher is a configuration gap in one source, not a
        reason to abandon every other source in the run.
        """
        specs: List[SourceSpec] = []
        skipped: List[SkippedSource] = []

        for config in self._configs.values():
            reason = config.skip_reason()
            if reason is not None:
                skipped.append(reason)
                continue
            fetch = fetchers.get(config.name)
            if fetch is None:
                skipped.append(
                    SkippedSource(
                        config.name,
                        SKIP_NO_FETCHER,
                        "no fetch callable was supplied for this source",
                    )
                )
                continue
            specs.append(SourceSpec(name=config.name, fetch=fetch))

        return SourcePlan(specs=specs, skipped=skipped)

    def enforce_recorded_decisions(
        self, decisions: Mapping[str, Any]
    ) -> "SourceRegistry":
        """Downgrade any source whose recorded verification does not permit it.

        Configuration can claim a source is permitted; this consults the
        evidence instead. A source with no recorded decision is downgraded to
        ``unknown``, because an unverified claim is not a permission.

        Returns a new registry - the original is untouched - so a caller cannot
        accidentally leave the unverified version in place.
        """
        downgraded: List[SourceConfig] = []
        for config in self._configs.values():
            decision = decisions.get(config.name)
            level = getattr(decision, "level", None)
            level_value = getattr(level, "value", level)
            if level_value == PERMITTED:
                downgraded.append(config)
                continue
            detail = getattr(decision, "reason", "") or "no verification decision recorded"
            downgraded.append(
                replace(
                    config,
                    access=RESTRICTED if level_value == RESTRICTED else UNKNOWN,
                    note=f"recorded access verification: {detail}",
                )
            )
        return SourceRegistry(downgraded)


def default_registry() -> SourceRegistry:
    """The registry this repository ships with.

    One source is configured, and it is configured honestly: enabled, with its
    access recorded as ``unknown`` rather than assumed. That means a default
    run consults nothing and says so, which is the correct behaviour for a
    repository whose only shipped source is a browser-driven scraper whose terms
    have not been assessed. Live verification is a later increment; until then
    this fails closed rather than quietly hitting a site.
    """
    return SourceRegistry(
        [
            SourceConfig(
                name="hiring.cafe",
                enabled=True,
                access=UNKNOWN,
                note=(
                    "access not yet assessed; enable only after a deliberate "
                    "decision is recorded"
                ),
            ),
        ]
    )
