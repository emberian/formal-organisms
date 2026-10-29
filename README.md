# formal-organisms

Small, checkable experiments on how language models learn, misread, repair and stop when the ground truth is a formal system. The experiments are designed so that no judge model is needed: every score comes from a checker.

The occasion was a LessWrong replication of "subtly wrong advice misaligns models more than obviously wrong advice" (2026-10-28), and ember's comment on it:

> I have some light intuitions into the way the realizability toposes ingress into the more localic models. "On-policy" can be fragile, arbitrarily fragile, for higher coherences. I have some studies (that I need to tidy up and publish...) about in-context learning of various logics and protocol families and the first error in a trajectory when corrected by human can lead to very long, complicated reasonings. Automated reasoning was a young field but double categories, parametricity, trace spaces, and metric enrichment all studied somewhat and (in my opinion) we're seeing the "shadows" of formalisms. I believe that purely computational (esp proof theoretic in interesting logics) model organisms would shed light into these questions if studied.

## The organisms

| | question | status |
|---|---|---|
| `o1_repair` | A connective is misread early in an in-context derivation, and the learner is then corrected (generic, location, semantics, rollback). How long does stable correct reuse take, and what shape does the repair take? | pilot |
| `o2_witness` | Truth versus witnesses: are classical answers right while constructive witness production fails, and does seeing witnesses move truth answers? | designed |
| `o3_coherence` | Generating relations practised on-policy; never-exercised coherences (pentagon, critical pairs) tested. | designed |
| `o4_orbit` | A known fixed point iterated by the model on its own output: true fixed point, model-specific attractor, or cycle? Does a planted first error shift the basin? Does self-referential commentary change it? | pilot |
| `o5_search` | The Librarian's falsifier: valence × search termination, with entrance refusal, persistence, looping and verified witnesses, base model against post-trained model. | pilot |
| `negmonoid` | Intuitionistic negation as the monoid {id, ¬, ¬¬} with ¬³ = ¬ against its classical quotient Z2: which circuit do tiny transformers learn? | designed |

The designs, and the reasoning behind them, are in `docs/` (the study note `MODEL-ORGANISMS.md`, written with Astra). Each organism's results are in `results/<organism>/REPORT.md`.

## Running

Python via `uv`. Model calls go through `common/llm.py` (OpenRouter, pinned model ids, every call appended to `results/ledger.jsonl` with its cost, global spend cap `FO_BUDGET_USD`). Tiny transformers run locally on MPS (`uv run --with torch`), and larger ones on Kaggle.

```bash
uv run python -m organisms.o1_repair.run --help
```

## Conventions

A checker returns `VALID`, `INVALID(counterexample)`, `UNRESOLVED` or `BUDGET_EXHAUSTED`, and failing to find a witness is never treated as proof that none exists. Every model is pinned by id and every condition is run on matched prompts. Reports state their denominators, list censored episodes, and include null results.
