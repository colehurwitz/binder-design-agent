# Binder Design Agent

**A protein binder design pipeline, exposed as agent tools, running on serverless GPUs.**

---

## What we built

An AI agent (Claude, via Claude Code) that can design candidate protein binders against a
real target protein, and independently evaluate whether those designs are any good —
end to end, without a human running any of the underlying tools by hand.

The agent doesn't run any biology software itself. It calls out to a dedicated
**protein design MCP server** ([`protein-design-mcp`](https://github.com/jasonkim8652/protein-design-mcp)),
deployed on [Modal](https://modal.com) serverless GPUs, which wraps several established
computational biology tools behind a simple set of callable functions ("tools" in agent
terminology).

```
 ┌────────────┐        MCP tool calls        ┌──────────────────────────┐
 │   Agent    │ ───────────────────────────▶ │  protein-design-mcp      │
 │  (Claude)  │ ◀─────────────────────────── │  server on Modal (GPU)   │
 └────────────┘        results (JSON)        └──────────────────────────┘
                                                  │  RFdiffusion
                                                  │  ProteinMPNN
                                                  │  ESMFold
                                                  │  AlphaFold2-Multimer (ColabFold)
                                                  ▼
                                              real GPU compute, on demand
```

## How Modal and Claude Code fit together

Two separate pieces of infrastructure, each doing one job:

- **Modal** is the compute layer. It hosts the MCP server as a serverless GPU
  deployment: no GPU sits idle waiting for work — a container spins up on demand when a
  tool is called, runs RFdiffusion/ProteinMPNN/ESMFold/AlphaFold2-Multimer, and shuts
  down automatically after a few minutes of inactivity. Model weights (tens of GB across
  all the tools) are cached on a persistent Modal Volume, so they're downloaded once and
  reused by every future container instead of being re-fetched or baked into the image.
  Redeploying (`modal deploy`) updates the running service in place, typically in
  seconds once the underlying image layers are already built.
- **Claude Code** is the agent layer. It's where the actual design work happens: an
  agent (Claude) holds the MCP connection to the deployed server and calls its tools —
  `design_binder`, `validate_design`, `predict_complex`, and so on — directly as part of
  a normal conversation, the same way it would call any other tool. There's no separate
  orchestration script standing between the agent and the biology tools; the agent
  drives the workflow itself, decides what to try next, and can inspect results
  mid-run to decide whether to continue, retry with different hotspots, or stop.

The two also meet at the infrastructure level in a more literal way: this session
itself runs inside a Modal Sandbox, and Claude Code's own shell access made it possible
to debug the deployment directly — `modal app logs` to watch a running GPU container's
output live, and `modal container exec` to exec into an in-flight container and inspect
process state, GPU utilization, and intermediate output files while a slow call was
still running, rather than only being able to see the final result or a bare error.

## The pipeline

**Input:** a target protein structure (PDB file) + the specific residues on it we want
the binder to attach to ("hotspots")

1. **Backbone generation (RFdiffusion)** — a generative diffusion model proposes
   candidate 3D shapes that could sit against the target at the chosen hotspots. No
   sequence yet, just geometry.
2. **Sequence design (ProteinMPNN)** — for each candidate backbone, an inverse-folding
   model designs an amino acid sequence predicted to actually fold into that shape.
3. **Structure validation (ESMFold)** — each designed sequence is independently
   re-folded from scratch to check whether it really adopts the intended structure.
4. **Scoring & filtering** — each candidate is scored on pLDDT (per-residue confidence),
   pTM (overall fold confidence), and MPNN score (sequence design confidence); weak
   candidates are dropped.
5. **Independent interface check (AlphaFold2-Multimer / ColabFold)** — the designed
   binder and the real target are folded *together* as a complex, using a completely
   different structure-prediction model than the one used for design. This is the step
   that actually tests whether the binder docks against the target the way it was
   intended to — steps 1–4 only prove the binder folds correctly on its own.

Steps 1–4 use the same underlying model family for both generation and self-scoring.
Step 5 exists specifically to avoid taking the pipeline's word for its own results —
it's a second opinion from an unrelated model.

## Real test case: PD-L1

We ran this against a real, clinically relevant target rather than a toy structure:
**PD-L1** (Programmed Death-Ligand 1), the immune checkpoint protein targeted by
checkpoint-blockade cancer therapies.

- **Target structure:** PDB [4ZQK](https://www.rcsb.org/structure/4ZQK) (Zak et al.,
  *Structure* 2015) — the real crystal structure of the human PD-1/PD-L1 complex.
- **Hotspot residues:** computed directly from the 3D coordinates — every PD-L1 residue
  within 4.5 Å of PD-1 in the real co-crystal — rather than taken from a paper's residue
  numbering, since numbering conventions can differ between a publication and a
  structure file. Core hotspots used: **Tyr56, Met115, Tyr123, Arg125**.

## Results

Best of 8 candidate designs that passed internal filters:

| Metric | Value | Meaning |
|---|---|---|
| pLDDT | **93.5** | per-residue structure confidence (0–100) |
| pTM | **0.85** | overall fold confidence (0–1) |
| MPNN score | **0.95** | sequence design confidence |
| Fold | single α-helix | a "helical minibinder" — a well-precedented binder architecture |

Independently re-folding the designed sequence (ESMFold, ignoring how it was designed)
reproduced the same pLDDT/pTM — internally consistent.

**Independent interface verification (AlphaFold2-Multimer):** binder + PD-L1 folded
together as a complex, to test real docking against the target rather than just the
binder's own standalone fold — a genuinely different model from the one used for design
and self-scoring above.

| Metric | Value | Field threshold | Meaning |
|---|---|---|---|
| ipTM | **0.39** | > 0.8 | inter-chain fold confidence — fails |
| Interface pAE | **11.6** | < 10 | positional error at the interface — fails |
| Complex pLDDT | 88.0 | — | overall complex structure confidence |
| Complex pTM | 0.70 | — | overall complex fold confidence |

**This is the actual finding.** The monomer self-score (93.5 pLDDT / 0.85 pTM) said this
was a strong binder. The independent complex prediction says the opposite: by the
thresholds the field uses to predict experimental hits, this candidate does not bind
PD-L1. The two steps are measuring different things — a well-folded helix and a
well-folded helix that binds its target score identically on step 1–4's metrics, and
only diverge once something actually checks the interface. That gap is the reason step 5
exists, and this run demonstrates it concretely rather than hypothetically.

## What an agent does with this result

A failed independent check isn't an endpoint, it's a decision point. Concretely, an
agent sitting on this ipTM 0.39 result has several cheaper moves before generating
anything new:

1. **Check the other candidates first** — only the single highest self-scoring design
   (of 8 that passed internal filters) was interface-checked. Since self-score is now
   demonstrated not to predict binding, checking the remaining 7 is the cheapest next
   step — one may score better despite a lower self-score.
2. **Re-optimize for affinity, not just foldability** — `optimize_sequence` can re-run
   ProteinMPNN on the same RFdiffusion backbone targeting binding affinity specifically,
   rather than the stability-only objective the initial design used. The backbone
   geometry may be fine; the sequence chosen for it may not be.
3. **Get better hotspots, not just more candidates** — `suggest_hotspots` (SASA,
   conservation, literature evidence) is available but unused here; hotspots were
   derived manually instead. If the chosen hotspots don't define a real binding groove,
   resampling sequences against them won't fix it.
4. **Widen the funnel** — the durable answer. Published hit rates for RFdiffusion+MPNN
   are well under 1%, so one failed candidate is the expected case, not a pipeline
   failure. The agent's job is generating enough candidates, cheaply self-score-filtered,
   that the rare real hit surfaces — not treating candidate #1 as the verdict.
5. **Keep failures as data** — every ipTM/pAE pair, rejected or not, is exactly the
   score-distribution data the negative-control idea below asks for, produced for free
   as a byproduct of normal operation.

What an agent should *not* do: retry the identical call hoping for a different number.
ipTM/pAE from a deterministic complex prediction don't change without changing the input.

**Tried step 1 live:** ran the independent check on candidate #2 from the same batch
(pLDDT 93.1 / pTM 0.85 self-score — nearly identical to candidate #1). Result: **ipTM
0.23, interface pAE 16.5** — worse than candidate #1, not better.

| Candidate | Self-score (pLDDT / pTM) | ipTM | Interface pAE | Passes? |
|---|---|---|---|---|
| #1 | 93.5 / 0.85 | 0.39 | 11.6 | No |
| #2 | 93.1 / 0.85 | 0.23 | 16.5 | No |
| #3 (new backbone, wider hotspot set) | 91.2 / 0.61 | 0.43 | 14.7 | No |

**Tried lever #1 (backbone diversity) + hotspot widening together:** a fresh
`design_binder` call, this time with 7 hotspot residues instead of 4. The result
(sequence composition, self-score) looked genuinely different from the first backbone —
evidence the fresh call did sample new backbone geometry, not just new sequences on the
old one. Interface check: **ipTM 0.43, interface pAE 14.7** — still fails, but the best
ipTM of the three so far. Note this changed two variables at once (hotspots and
backbone), so the small improvement can't be cleanly attributed to either one — a proper
test would isolate them.

Three candidates, three independent-verification failures, despite strong-looking
self-scores on two of them. This is the expected shape of the problem, not a sign
anything is broken — RFdiffusion+MPNN hit rates are reported well under 1%. It's direct
evidence for the funnel-scaling item below: self-score alone doesn't separate these
candidates in any way that predicts the real outcome, so the only way to find an actual
hit is running the independent check across many more candidates, not picking one
"best-looking" design and stopping.

### Using the model's own exhaust to guide refinement

The scalar scores above (ipTM, interface pAE) are summaries. `predict_complex` also
returns a full residue-by-residue PAE matrix (`pae_matrix_path`) that's much richer —
and looking at it on candidate #3 changed the read on that result entirely.

Pulling the matrix and checking, for each binder residue, its best (minimum) PAE against
any target residue: **49 of 80 binder residues (60%) have *some* confident (PAE < 10)
contact with the target** — the flat 14.7 average was hiding this. The confident
contacts aren't scattered noise, either. Mapping matrix positions back to real PD-L1
residue numbers:

- Binder residues 29–39 ↔ **Met115** (PAE as low as 4.73)
- Binder residues 13–27 ↔ **Tyr56** (PAE 6–10)
- Binder residues 49–63 ↔ **Ile54** (PAE 8.6–9.9, borderline)

Met115, Tyr56, and Ile54 are three of the real, structure-verified PD-1 contact
residues. This candidate isn't a random dud — it's a **partial binder**: it found the
right neighborhood and formed one genuinely confident contact point (residues 29–39
anchored on Met115), while the rest of the 80-residue helix (roughly 1–12 and 64–80)
contributes nothing and drags the average down.

That's a specific, actionable target for refinement rather than generating a fresh
backbone from scratch: `optimize_sequence` with `fixed_positions` locking residues
29–39 in place, `optimization_target="affinity"`, redesigning everything else.

| | ipTM | Interface pAE |
|---|---|---|
| Before (candidate #3) | 0.43 | 14.7 |
| After targeted `optimize_sequence` | **0.55** | **11.6** |

A real, independently-verified improvement (+0.12 ipTM, −3.1 pAE) from 5 targeted
mutations — still short of the 0.8 / 10 threshold, but a genuine step closer, and larger
than the optimizer's own self-predicted improvement (+0.010 pTM) suggested. That's the
same lesson from a different angle: self-score isn't just overconfident (candidates #1
and #2) — here it *underestimated* a real gain. The fix in both directions is the same:
don't trust the self-score, verify independently.


candidates in any way that predicts the real outcome, so the only way to find an actual
hit is running the independent check across many more candidates, not picking one
"best-looking" design and stopping.

### Improving the hit rate, not just the throughput

Scaling the funnel increases the number of attempts. A separate question is whether each
attempt's odds can be improved — and the two candidates tested above already point at the
biggest lever.

Looking at the actual sequences of all 8 designs from the original batch (`#1`:
`SVAAGQ...LLAVDPSAAP...AAAAAG`, `#2`: `SVEEIK...TILSADPSAG...EVEEKAAG`, `#3`:
`STAEGQ...LLAVDPALAP...QAAAAAG`, `#4`: `SVAETR...LLSADPSAG...KAVAAAAAG`) — same length,
same recurring motif around residue 52 (`...DP...`), same start/end pattern. That's not 8
independent structural ideas; that's ProteinMPNN sampling different sequences on what
looks like the same RFdiffusion backbone. Both interface checks above very likely tested
one structural hypothesis twice, not two.

In rough order of leverage:

1. **Force genuine backbone diversity, not just sequence resampling** — request multiple
   independent RFdiffusion runs (different seeds) instead of treating several MPNN
   samples off one backbone as diverse candidates. The single most likely fix given the
   evidence above, and cheap for an agent to do automatically.
2. **Use `suggest_hotspots` instead of raw contact distance** — hotspots here were chosen
   by "closest to PD-1 in the crystal structure," which isn't the same as "residues that
   actually drive binding energy." `suggest_hotspots` folds in conservation and SASA,
   which correlates better with true hotspots than geometric proximity alone.
3. **Treat hotspot choice as a variable to search, not a fixed input** — try a few
   different subsets of the real contact residues instead of committing to one set picked
   once.
4. **Sweep binder length / scaffold topology** — only the 80-residue default was tried,
   which produced only helical minibinders. Different epitopes suit different topologies
   (helical bundle vs. beta-sheet-rich); real campaigns sweep this.
5. **Refine near-misses instead of discarding them** — `optimize_sequence` with
   `target="affinity"` can redesign a promising backbone's sequence specifically for
   binding rather than just stability, cheaper than generating a new backbone from
   scratch.

Honest caveat: none of this guarantees a hit — even done well, published rates are still
well under 1%. These improve the odds per attempt; scaling the funnel is what turns
better odds into an actual result.

## What's next

The current result (pLDDT/pTM from a monomer refold) shows the binder folds correctly
on its own — it does not yet show that it binds PD-L1. Closing that gap is the priority,
ahead of adding new capabilities:

1. **Self-consistency RMSD** — the right metrics (ipTM, interface pAE) are now reported
   above; still missing is refolding the designed sequence and checking it lands within
   ~2 Å of the RFdiffusion backbone it was designed for — checks that the geometry
   reproduces, not just that confidence does.
2. **Add a negative control** — run the identical pipeline against scrambled hotspots or
   an unrelated target and report the score distribution alongside the real one, so a
   given score has a reference point.
3. **Generalize target intake** — turn hotspot derivation (PDB + partner chain → contact
   residues within 4.5 Å) into a reusable tool, so a new target is one sentence instead
   of a manual analysis.
4. **Scale the funnel, budget-aware** — published RFdiffusion+MPNN hit rates are well
   under 1%, so the pipeline's value is running many candidates cheaply and discarding
   almost all of them. The agent should apply cheap self-score filters first and only
   spend AF2-Multimer compute on survivors, rather than running the expensive check on
   every candidate (as today's single-candidate demo does).
5. **AF2 initial-guess rescoring** (Bennett et al. 2023) — seed AF2 with the designed
   complex's own coordinates instead of folding from scratch; sharper discrimination
   than naive multimer folding, but requires custom inference code, not just a config
   flag on the existing ColabFold call.
6. **Lab-ready output** — codon-optimized DNA, expression-liability checks (free
   cysteines, hydrophobic patches, aggregation-prone stretches), and a ranked shortlist
   — the difference between an interesting result and one a wet lab can order tomorrow.
7. **PyRosetta-based physics scoring** (binding energy, buried surface area) as a further
   independent signal, once the funnel feeding it is validated.
