# Results update: real bugs found, a real hit, and what's still unproven

This supersedes the refinement claim in the original deck and
`TECHNICAL_APPENDIX.md` (candidate #3, "ipTM 0.43 → 0.55"). An external code
review of the source found that result rested on a mislabeled, non-target-aware
tool. This document covers what was found, what was fixed, and what we
actually know now.

## What the review found (verified against source, not taken on faith)

1. **`optimize_sequence` doesn't do affinity optimization.** `target_pdb` is
   used for exactly one thing — `Path(target_pdb).exists()`. The
   `optimization_target` parameter (`"stability"`/`"affinity"`/`"both"`) is
   validated and then never used again; every code path is identical
   regardless of what you pass. `affinity_delta` in its output is literally
   `ESMFold(optimized_seq).ptm − ESMFold(baseline_seq).ptm` — a **monomer**
   refold, no target, no complex.
2. **Its scoring isn't masked marginal scoring**, despite the docstring —
   one unmasked forward pass, logits reused for every position, no rejection
   if a round's overall score decreases.
3. The original "ipTM 0.43 → 0.55" refinement used this tool. The refined
   sequence was also compositionally extreme (70% → 76% glutamate), a further
   reason to doubt the gain was real rather than a model artifact on
   low-complexity input.
4. Separately, `design_for_interface` — the method that actually should do
   target-aware refinement (keeps the target chain fixed via ProteinMPNN's
   real multi-chain workflow) — existed but was never exposed as an MCP tool,
   had a dead `interface_residues` parameter, and called a ProteinMPNN CLI
   flag (`--chain_id_design`) that doesn't exist on this ProteinMPNN version.
   It would have failed immediately if anyone had actually called it.

## What was fixed (this local copy only)

- `design_for_interface` rewritten to use ProteinMPNN's own documented
  multi-chain workflow (`parse_multiple_chains.py` → `assign_fixed_chains.py`
  → `make_fixed_positions_dict.py`), the real flags (`--jsonl_path`,
  `--chain_id_jsonl`, `--fixed_positions_jsonl`), and added real
  `fixed_positions` support so it can do exactly what we need: keep the
  target chain fixed *and* lock specific binder residues.
- A second bug found while using the fix: the first refinement attempt fed
  ProteinMPNN the raw RFdiffusion backbone (placeholder glycines, no real
  sequence) as the structure to lock positions against — so "locking" 78
  positions locked them to glycine, producing a garbage poly-glycine
  sequence. Fixed by threading the real designed sequence onto a proper
  structure (using AlphaFold2-Multimer's own predicted complex, which has
  the correct sequence and correct atoms for both chains) before locking
  anything.

## The real result: a genuine hit

First candidate generated with the corrected pipeline (fresh RFdiffusion
backbone → properly interface-aware ProteinMPNN design → independent
AlphaFold2-Multimer verification), no refinement involved:

| Metric | Value |
|---|---|
| pLDDT (complex) | 95.3 |
| pTM (complex) | 0.88 |
| **ipTM** | **0.81** |
| Interface pAE | low across most of the binder (mean 5.14 over confident residues) |

Sequence: `ELAKLKAEAEKLELEALKVRREALKKEAKAKELKEEAKATGNKELEEKAKKAEKEAEKLKKKAEELKKKAEELKKKAEEL`

This clears the field-standard screening bar (ipTM > 0.8) on the **first
try** with the corrected tool. Reproduced exactly on a second independent
`predict_complex` call (identical plddt/ptm/ipTM), confirming AlphaFold2-Multimer
is deterministic here and the result isn't a fluke of one run. 78 of 80 binder
residues show confident (PAE < 10) contact with real PD-L1 hotspots (Tyr56,
Met115, Asp122, Arg121, and more).

Running the fixed refinement mechanism on this candidate correctly changed
**zero positions** — with 78/80 residues already confident, there was nothing
useful to redesign, and the mechanism correctly recognized that rather than
degrading a good candidate.

## PAE-guided refinement, tested cleanly (the actual open question)

With both bugs fixed, three more independent tests:

| Case | Confident residues | Before ipTM | After ipTM | Δ | Composition check |
|---|---|---|---|---|---|
| Winning candidate (already strong) | 78/80 (97.5%) | 0.81 | 0.81 | 0 | n/a — no positions changed |
| Weak-partial candidate | 60/80 (75%, high mean PAE) | 0.23 | 0.26 | +0.03 | clean (38.8%→41.2% Ala, minor) |
| Middle candidate | 32/80 (40%) | 0.45 | **0.77** | **+0.32** | clean (35.0%→31.2% Ala, diverse residues throughout) |

The middle-candidate result is the strongest evidence so far that locking a
confident region and redesigning the rest can produce a real, substantial,
independently-verified improvement — nearly closing the gap to the 0.8 bar
from a genuinely mediocre starting point, with no composition red flags.

### Why this still isn't proof PAE-guidance works

1. **No random-position control.** We haven't compared "lock the PAE-confident
   positions" against "lock the same *number* of positions, chosen randomly."
   ProteinMPNN properly redesigning ~40–50% of a mediocre sequence, correctly
   conditioned on the real target, might help regardless of *which* positions
   are held fixed. The PAE step's real contribution — picking the right
   positions to keep — is exactly the part this hasn't isolated.
2. **Effect size is wildly inconsistent across the only three trials**: 0,
   +0.03, +0.32. That's not yet a characterized effect, just three data
   points with high variance. Could be a real effect whose size depends on
   the candidate; could be partly luck, the same way the 0.81 winning
   backbone was itself a lucky draw among six independent attempts this
   session (0.20, 0.23, 0.39, 0.43, 0.45→0.77, 0.81 — one clear hit, wide
   spread otherwise).
3. **Unquantified selection step.** Each refinement call generates ~8
   ProteinMPNN samples; only the single best-scored (by ProteinMPNN's own
   metric, not ipTM) was independently verified. Whether other samples from
   the same batch would have scored better or worse than the one we picked
   is unknown.

**The experiment that would actually answer this**: same weak starting
candidate, two matched conditions (PAE-selected positions locked vs. an equal
number of randomly-selected positions locked), each run multiple times,
independently verified. Not yet run.

## All independent candidates this session, for reference

| # | Source | Confident % | ipTM | Interface pAE | Notes |
|---|---|---|---|---|---|
| 1 | Original batch (pre-fix) | 61% | 0.39 | 11.6 | first real independent-verification result |
| 2 | Original batch (pre-fix) | — | 0.23 | 16.5 | |
| 3 | New backbone, wider hotspots (pre-fix) | — | 0.43 | 14.7 | later "refined" to 0.55 via the compromised optimizer — see caveats above |
| 4 | Fresh backbone, corrected pipeline | 0% (self-score already weak) | 0.20 | 19.9 | weakest candidate; correctly not worth refining |
| 5 | Fresh backbone, corrected pipeline | 75% | 0.23 → 0.26 | — | first clean refinement test |
| 6 | Fresh backbone, corrected pipeline | **97.5%** | **0.81** | low | **the real hit** |
| 7 | Fresh backbone, corrected pipeline | 40% | 0.45 → 0.77 | — | largest clean refinement gain |

## Bottom line

- The tooling bugs are real, verified, and fixed in this local copy.
- We have one genuine, reproduced, independently-verified passing candidate
  (ipTM 0.81) from the corrected pipeline running once, straight through, no
  refinement needed.
- PAE-guided refinement, tested cleanly for the first time, shows a
  promising but not yet proven effect — real in the sense of "not a bug or
  artifact," unproven in the sense of "isolated from the alternative
  explanation that redesigning a chunk of a mediocre sequence helps
  regardless of which positions are chosen."
- No experimental (wet-lab) validation of any candidate. Everything above is
  model prediction.
