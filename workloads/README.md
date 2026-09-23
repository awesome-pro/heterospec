# workloads/

Prompt data and workload definitions.

Prompt *classes* are declared here; their acceptance behaviour is **measured**,
never assumed. A class is only labelled high- or low-acceptance after traces
show it (PROJECT.md Phase 5).

Workload families (PROJECT.md Phase 5):

| Family | Composition |
| --- | --- |
| `high` | predominantly high-acceptance prompts |
| `low` | predominantly low-acceptance prompts |
| `mixed_75_25` | 75% high / 25% low |
| `mixed_50_50` | 50% high / 50% low |
| `mixed_25_75` | 25% high / 75% low |
| `phase_shift` | composition changes during the run |
| `real_mixed` | code + reasoning + chat dataset |

`high`/`low` prompt seeds are adapted from SGLang's own
`benchmark/bench_adaptive_speculative.py`, which already encodes the
repetitive-output (high) vs open-ended-diversity (low) distinction.
