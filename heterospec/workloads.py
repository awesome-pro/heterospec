"""Workload families and prompt classes.

PROJECT.md Phase 5: build controlled populations, not "100 identical prompts".

Two deliberate design choices
----------------------------
1. **Classes are named neutrally, not by expected acceptance.** `repetitive` and
   `open_ended` describe *what the prompt asks for*, not how well a draft model
   will do on it. Code, reasoning and chat are not assumed to be high- or
   low-acceptance: the analysis labels classes from observed traces
   (PROJECT.md: "Do not label code/chat as inherently high/low acceptance without
   measuring it"). The `high`/`low` *family* names below are hypotheses under
   test, and the README must report them as measured, not assumed.

2. **Composition is interleaved across arrival order, not blocked.** Requests
   are dispatched concurrently, so a batch is a sliding window of the arrival
   sequence. If classes were grouped -- all `repetitive` first, then all
   `open_ended` -- no batch would ever contain both, and a "mixed" workload would
   silently measure two homogeneous phases instead. :func:`build_plan`
   therefore interleaves classes deterministically by weight.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

__all__ = [
    "PROMPT_CLASSES",
    "PromptClass",
    "RequestSpec",
    "WorkloadSpec",
    "WORKLOADS",
    "build_plan",
    "get_workload",
    "workload_names",
]


@dataclass(frozen=True)
class PromptClass:
    """A labelled population of prompts."""

    name: str
    description: str
    prompts: tuple[str, ...]
    hypothesis: str = ""

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError(f"prompt class {self.name!r} has no prompts")


# ---------------------------------------------------------------------------
# Prompt classes
# ---------------------------------------------------------------------------

# Repetitive / low-entropy continuations. The draft model has an easy job, so
# these are expected to accept deep.
_REPETITIVE = PromptClass(
    name="repetitive",
    description="Low-entropy, highly predictable continuations",
    hypothesis="expected high acceptance",
    prompts=(
        "Output exactly 200 new lines. Every line must be the single digit 1. "
        "No numbering, no punctuation, no commentary.",
        "Output exactly 200 new lines. Every line must be the word READY in "
        "uppercase. No numbering, no punctuation, no commentary.",
        "Write the integers from 1 to 200, one per line. Nothing else.",
        "Repeat the following line exactly 150 times, one per line: "
        "the quick brown fox jumps over the lazy dog",
        "Output exactly 150 lines, each identical to: abcabcabc",
        "Print the string 'SGLANG' 200 times, each on its own line, with no "
        "other text.",
        "Output 100 lines. Each line must be exactly: 0 1 2 3 4 5 6 7 8 9",
        "Emit exactly 120 lines, each consisting only of the character x.",
    ),
)

# High-entropy, creative continuations. The draft model must commit early to
# content the target will diverge from.
_OPEN_ENDED = PromptClass(
    name="open_ended",
    description="High-entropy creative generation",
    hypothesis="expected low acceptance",
    prompts=(
        "Compose a poem in the style of Emily Dickinson about quantum "
        "entanglement. Make it emotionally resonant.",
        "Write 100 two-sentence biographies of eccentric inventors with unique "
        "names, hometowns, and inventions.",
        "Write a long travel diary from a botanist visiting a chain of floating "
        "islands. Every paragraph should introduce new flora, customs, weather, "
        "and political tensions.",
        "Write 80 newspaper headlines and subheads from 80 different "
        "alternate-history worlds. Each headline must introduce a different "
        "place, conflict, and technology.",
        "Imagine 60 distinct dream sequences, each two sentences, no two sharing "
        "an image or a setting.",
        "Invent 50 new colours. For each, give a name, a hexadecimal code, and a "
        "one-sentence description of where it occurs in nature.",
        "Write a dialogue between two rival cartographers, each trying to "
        "discredit the other's map. Include invented place names.",
        "Describe 40 impossible machines, each in exactly three sentences, with "
        "no repeated mechanical principle.",
    ),
)

_CODE = PromptClass(
    name="code",
    description="Source-code generation",
    hypothesis="unmeasured -- do not assume high or low",
    prompts=(
        "Write a Python function that merges two sorted lists in linear time. "
        "Include docstring and type hints.",
        "Implement a thread-safe LRU cache in Python with a configurable capacity.",
        "Write a Rust function that parses a simple arithmetic expression into "
        "an AST. Include error handling.",
        "Write a SQL query that finds the top 5 customers by revenue in each "
        "region, handling ties deterministically.",
        "Implement quicksort in C with a median-of-three pivot. Explain the "
        "partition step.",
        "Write a Python decorator that retries a function with exponential "
        "backoff and jitter.",
        "Implement a trie in TypeScript supporting prefix search and wildcard "
        "matching.",
        "Write a bash script that finds and reports duplicate files by content hash.",
    ),
)

_REASONING = PromptClass(
    name="reasoning",
    description="Multi-step analytical reasoning",
    hypothesis="unmeasured -- do not assume high or low",
    prompts=(
        "A bat and a ball cost $1.10 together. The bat costs $1.00 more than "
        "the ball. How much does the ball cost? Show each step.",
        "Three people check into a hotel for $30. Explain the missing-dollar "
        "puzzle step by step and identify the error in reasoning.",
        "How many times do the hands of a clock overlap in 24 hours? Derive it.",
        "Prove that the square root of 2 is irrational, explaining each step "
        "carefully.",
        "A farmer has 17 sheep and all but 9 die. Then he buys back half of "
        "what died. How many does he have? Show the reasoning.",
        "You have 12 coins, one counterfeit with unknown weight. Find it in 3 "
        "weighings and explain the decision tree.",
        "Explain why the Monty Hall problem gives 2/3, deriving it from "
        "conditional probability.",
        "Derive the closed form for the sum of the first n cubes, and prove it "
        "by induction.",
    ),
)

_CHAT = PromptClass(
    name="chat",
    description="Conversational turn-taking",
    hypothesis="unmeasured -- do not assume high or low",
    prompts=(
        "Hey, how's it going? What have you been up to?",
        "I'm thinking about learning to cook. Where should I start?",
        "Can you explain what you are and what you can help with?",
        "What's a good way to stay motivated when working on long projects?",
        "I had a rough day. Any suggestions for winding down?",
        "What are some interesting facts about the ocean?",
        "Do you have any tips for sleeping better?",
        "What should I consider before getting a pet?",
    ),
)

PROMPT_CLASSES: dict[str, PromptClass] = {
    c.name: c for c in (_REPETITIVE, _OPEN_ENDED, _CODE, _REASONING, _CHAT)
}


# ---------------------------------------------------------------------------
# Workload families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase:
    """One composition regime, for ``n_requests`` requests."""

    composition: Mapping[str, float]
    n_requests: int
    label: str = ""


@dataclass(frozen=True)
class WorkloadSpec:
    """A workload: one or more phases of class composition."""

    name: str
    description: str
    phases: tuple[Phase, ...]
    max_new_tokens: int = 256
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError(f"workload {self.name!r} has no phases")
        for phase in self.phases:
            if phase.n_requests <= 0:
                raise ValueError(f"{self.name}: n_requests must be > 0")
            bad = set(phase.composition) - set(PROMPT_CLASSES)
            if bad:
                raise ValueError(
                    f"{self.name}: unknown prompt classes {sorted(bad)}; "
                    f"known: {sorted(PROMPT_CLASSES)}"
                )
            total = sum(phase.composition.values())
            if total <= 0:
                raise ValueError(f"{self.name}: composition weights sum to <= 0")
            if any(w < 0 for w in phase.composition.values()):
                raise ValueError(f"{self.name}: negative composition weight")

    @property
    def total_requests(self) -> int:
        return sum(p.n_requests for p in self.phases)

    @property
    def is_phase_shifting(self) -> bool:
        return len(self.phases) > 1

    def scaled(self, num_requests: int) -> WorkloadSpec:
        """Scale the workload to ``num_requests`` total, preserving proportions.

        Raises if there are fewer requests than phases, since every phase must
        contain at least one request -- silently dropping a phase would remove a
        composition regime from the experiment without saying so.
        """
        if num_requests <= 0:
            raise ValueError("num_requests must be > 0")
        if num_requests < len(self.phases):
            raise ValueError(
                f"workload {self.name!r} has {len(self.phases)} phases; "
                f"num_requests={num_requests} is too few to fill each phase with "
                f"at least one request"
            )
        base = self.total_requests
        scaled: list[Phase] = []
        remaining = num_requests
        for i, phase in enumerate(self.phases):
            if i == len(self.phases) - 1:
                n = remaining
            else:
                n = max(1, round(phase.n_requests * num_requests / base))
                n = min(n, remaining - (len(self.phases) - i - 1))
            remaining -= n
            scaled.append(Phase(phase.composition, n, phase.label))
        return WorkloadSpec(
            name=self.name,
            description=self.description,
            phases=tuple(scaled),
            max_new_tokens=self.max_new_tokens,
            notes=self.notes,
        )


def _mix(*pairs: tuple[str, float], **kw) -> Phase:
    return Phase(composition=dict(pairs), **kw)


WORKLOADS: dict[str, WorkloadSpec] = {
    "high": WorkloadSpec(
        name="high",
        description="Predominantly repetitive, low-entropy prompts",
        phases=(_mix(("repetitive", 1.0), n_requests=200, label="high"),),
        notes="Hypothesis: deep speculation helps. Must be confirmed by traces.",
    ),
    "low": WorkloadSpec(
        name="low",
        description="Predominantly open-ended, high-entropy prompts",
        phases=(_mix(("open_ended", 1.0), n_requests=200, label="low"),),
        notes="Hypothesis: aggressive speculation wastes work. Confirm by traces.",
    ),
    "mixed_75_25": WorkloadSpec(
        name="mixed_75_25",
        description="75% repetitive / 25% open-ended",
        phases=(
            _mix(
                ("repetitive", 0.75),
                ("open_ended", 0.25),
                n_requests=200,
                label="mixed_75_25",
            ),
        ),
    ),
    "mixed_50_50": WorkloadSpec(
        name="mixed_50_50",
        description="50% repetitive / 50% open-ended",
        phases=(
            _mix(
                ("repetitive", 0.5),
                ("open_ended", 0.5),
                n_requests=200,
                label="mixed_50_50",
            ),
        ),
    ),
    "mixed_25_75": WorkloadSpec(
        name="mixed_25_75",
        description="25% repetitive / 75% open-ended",
        phases=(
            _mix(
                ("repetitive", 0.25),
                ("open_ended", 0.75),
                n_requests=200,
                label="mixed_25_75",
            ),
        ),
    ),
    "phase_shift": WorkloadSpec(
        name="phase_shift",
        description="Composition changes during the run: low -> high -> low -> high",
        phases=(
            _mix(("open_ended", 1.0), n_requests=60, label="low_1"),
            _mix(("repetitive", 1.0), n_requests=60, label="high_1"),
            _mix(("open_ended", 1.0), n_requests=60, label="low_2"),
            _mix(("repetitive", 1.0), n_requests=60, label="high_2"),
        ),
        notes="Stresses controller switching. Homogeneous within each phase.",
    ),
    "real_mixed": WorkloadSpec(
        name="real_mixed",
        description="Code + reasoning + chat, no synthetic high/low prompts",
        phases=(
            _mix(
                ("code", 0.34),
                ("reasoning", 0.33),
                ("chat", 0.33),
                n_requests=200,
                label="real_mixed",
            ),
        ),
        notes="Answers whether heterogeneity matters outside a constructed "
        "benchmark. Classes are characterised from traces, not assumed.",
    ),
}


def get_workload(name: str) -> WorkloadSpec:
    if name not in WORKLOADS:
        raise KeyError(f"unknown workload {name!r}; available: {workload_names()}")
    return WORKLOADS[name]


def workload_names() -> list[str]:
    return sorted(WORKLOADS)


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestSpec:
    """One request to send."""

    index: int
    rid: str
    prompt_class: str
    prompt: str
    phase_label: str
    workload: str
    max_new_tokens: int
    seed: int


def _largest_remainder(weights: Mapping[str, float], n: int) -> dict[str, int]:
    """Apportion ``n`` slots across classes proportionally, exactly and stably."""
    total = sum(weights.values())
    exact = {k: v * n / total for k, v in weights.items()}
    counts = {k: int(v) for k, v in exact.items()}
    remainder = n - sum(counts.values())
    # Largest fractional part first; ties broken by name for determinism.
    order = sorted(exact, key=lambda k: (-(exact[k] - counts[k]), k))
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def _interleave(per_class: Mapping[str, int]) -> list[str]:
    """Evenly spread each class across the arrival sequence.

    Places class ``k``'s ``j``-th item at fractional position
    ``(j + 0.5) * total / count_k`` and walks positions in order. This is a
    Bresenham-style spread: it respects the exact counts from
    :func:`_largest_remainder` while keeping every class present throughout the
    sequence, so any batch window of adequate size sees a genuine mixture.
    Ties break on class name for determinism.

    Blocking by class instead -- all ``repetitive`` then all ``open_ended`` --
    would mean no batch ever mixes, silently turning a mixed workload into two
    homogeneous phases and destroying the experiment.
    """
    remaining = {k: c for k, c in per_class.items() if c > 0}
    total = sum(remaining.values())
    if total == 0:
        return []

    slots: list[tuple[float, str]] = []
    for cls, count in remaining.items():
        for j in range(count):
            slots.append(((j + 0.5) * total / count, cls))
    slots.sort(key=lambda t: (t[0], t[1]))
    return [cls for _, cls in slots]


def build_plan(
    workload: str | WorkloadSpec,
    num_requests: int | None = None,
    *,
    seed: int = 0,
    max_new_tokens: int | None = None,
    rid_prefix: str = "r",
) -> list[RequestSpec]:
    """Build the concrete request plan for a workload.

    Prompts are sampled per class with replacement using a seeded RNG, so a plan
    is reproducible from ``(workload, num_requests, seed)`` alone.
    """
    spec = get_workload(workload) if isinstance(workload, str) else workload
    if num_requests is not None:
        spec = spec.scaled(num_requests)

    rng = random.Random(seed)
    mnt = max_new_tokens if max_new_tokens is not None else spec.max_new_tokens

    plan: list[RequestSpec] = []
    idx = 0
    for phase in spec.phases:
        counts = _largest_remainder(phase.composition, phase.n_requests)
        for cls in _interleave(counts):
            prompt = rng.choice(PROMPT_CLASSES[cls].prompts)
            plan.append(
                RequestSpec(
                    index=idx,
                    rid=f"{rid_prefix}{idx:05d}",
                    prompt_class=cls,
                    prompt=prompt,
                    phase_label=phase.label or spec.name,
                    workload=spec.name,
                    max_new_tokens=mnt,
                    # Per-request seed: fixes sampling without coupling requests.
                    seed=rng.randrange(2**31),
                )
            )
            idx += 1
    return plan


@dataclass
class PlanSummary:
    """Counts for reporting what a plan actually contains."""

    total: int
    by_class: dict[str, int] = field(default_factory=dict)
    by_phase: dict[str, int] = field(default_factory=dict)
    max_run_of_single_class: int = 0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_class": self.by_class,
            "by_phase": self.by_phase,
            "max_run_of_single_class": self.max_run_of_single_class,
        }


def summarise_plan(plan: Iterable[RequestSpec]) -> PlanSummary:
    plan = list(plan)
    by_class: dict[str, int] = {}
    by_phase: dict[str, int] = {}
    run = best = 0
    prev: str | None = None
    for r in plan:
        by_class[r.prompt_class] = by_class.get(r.prompt_class, 0) + 1
        by_phase[r.phase_label] = by_phase.get(r.phase_label, 0) + 1
        run = run + 1 if r.prompt_class == prev else 1
        prev = r.prompt_class
        best = max(best, run)
    return PlanSummary(
        total=len(plan),
        by_class=by_class,
        by_phase=by_phase,
        max_run_of_single_class=best,
    )
