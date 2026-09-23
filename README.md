# Binder Design Agent

Hackathon project: an AI agent (Claude, via Claude Code) that drives a real protein
binder design pipeline — RFdiffusion → ProteinMPNN → ESMFold → AlphaFold2-Multimer —
running on serverless GPUs, and uses the models' own output to decide what to try next.

## The pipeline

Five MCP tool calls, each a real model running on a Modal GPU — plus the agent decision
point where step 5 fails and something has to happen next:

```
 INPUT: target PDB + hotspot residues
   │
   ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 1  RFDIFFUSION          design_binder                    │
 │    candidate 3D backbones, shape only,                   │
 │    conditioned on target + hotspots                      │
 └──────────────────────────────────────────────────────────┘
   │
   ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 2  PROTEINMPNN          inverse folding                  │◀────┐
 │    sequence predicted to fold into                       │     │
 │    that backbone                                         │     │
 └──────────────────────────────────────────────────────────┘     │
   │                                                              │
   ▼                                                              │
 ┌──────────────────────────────────────────────────────────┐     │
 │ 3  ESMFOLD              refold ALONE                     │     │
 │    target removed -> pLDDT, pTM                          │     │
 │    (self-score / monomer confidence)                     │     │
 └──────────────────────────────────────────────────────────┘     │
   │                                                              │
   ▼                                                              │
 ┌──────────────────────────────────────────────────────────┐     │
 │ 4  FILTER               agent or human                   │     │
 │    drop weak candidates by self-score                    │     │
 └──────────────────────────────────────────────────────────┘     │
   │                                                              │
   ▼                                                              │
 ┌──────────────────────────────────────────────────────────┐     │
 │ 5  ALPHAFOLD2-MULTIMER  predict_complex                  │     │
 │    binder+target folded TOGETHER, a                      │     │
 │    different model -> ipTM, interface                    │     │
 │    pAE (independent verification)                        │     │
 └──────────────────────────────────────────────────────────┘     │
   │                                                              │
   ▼                                                              │
 PASSES THRESHOLD? (ipTM > 0.8, interface pAE < 10)               │
   │                                                              │
   ├── yes --> OUTPUT: sequence + predicted complex +             │
   │           independent confidence scores                      │
   │                                                              │
   └── no  --> AGENT reads the full per-residue PAE               │
               matrix, not just the summary score --              │
               finds which binder residues already                │
               contact the target and which don't.                │
               Decides: refine (lock what works,                  │
               redesign the rest) or discard and try              │
               a fresh backbone / another candidate.              │
                       │                                          │
                       └─────────────────────────────────────────┘
                           (or step 1), re-runs step 5
```

Steps 1–4 share one model family (ESMFold) for both generation and self-scoring — a
binder can score well there just by being a stable, well-folded shape, whether or not it
actually binds anything. Step 5 uses a completely different model to check the thing
steps 1–4 can't: does it dock against the real target. That gap — and the loop-back it
forces — is what the whole project is about.

**What actually happened in this repo's run:** candidate #3 failed step 5 (ipTM 0.43).
The agent read the PAE matrix, found a real partial contact on Met115, looped back to
step 2 with that residue range locked, and re-ran step 5 on the result (ipTM 0.55). See
`TECHNICAL_APPENDIX.md` sections 3–5 for the exact calls.

**Important caveat:** in this run, every step of that loop — reading the matrix,
deciding to refine rather than discard, choosing which residues to lock — was done
interactively, by a human directing the agent through each decision, not by the agent
running the loop on its own. The deck is explicit about this. Whether an agent can make
these calls autonomously, and whether doing so beats a fixed parameter sweep at equal
compute budget, is the open question this project sets up but doesn't answer.

**Start here: [`Binder Design Agent.pptx`](./Binder%20Design%20Agent.pptx)** — the
presentation, and the canonical summary of what was built and found, including
a mid-project retraction (an early refinement claim turned out to rest on a
broken, non-target-aware tool) and the corrected results that followed.
**[`RESULTS_UPDATE.md`](./RESULTS_UPDATE.md)** has the same corrected story in
more detail, with exact reproducible values.

## Headline result

Ran against a real target (PD-L1, PDB 4ZQK) with hotspots derived directly from the
real PD-1/PD-L1 crystal contacts.

An early result (best candidate: ipTM 0.39, "refined" to 0.55) was retracted
after a code review found the refinement tool never actually used the
target — see the deck or `RESULTS_UPDATE.md` for the full story. With that
fixed, the corrected pipeline produced a real, reproduced, independently-verified
**passing candidate: ipTM 0.81** (screening bar is 0.8), and clean tests of
PAE-guided refinement on weaker candidates (+0.03 and +0.32 ipTM) — real
effects, though not yet isolated from a simpler explanation (see the deck's
final slides).

No experimental binding data. This is a model-prediction pipeline and its results
should be read as that, not as validated binder discovery.

## Repo layout

- **`Binder Design Agent.pptx`** — the presentation, including the retraction
  and corrected results.
- **`RESULTS_UPDATE.md`** — corrected results after an external code review;
  read this alongside or instead of the deck's refinement claim.
- **`TECHNICAL_APPENDIX.md`** — full audit trail for the *original* (now
  superseded) PAE-guided refinement result: exact tool-call inputs/outputs,
  scripts, backend settings, and an explicit accounting of which raw
  artifacts (predicted structures, full PAE matrices) were and weren't
  preserved. Kept for transparency, not as the current headline.
- **`protein-design-mcp/`** — vendored, patched clone of
  [`jasonkim8652/protein-design-mcp`](https://github.com/jasonkim8652/protein-design-mcp),
  the MCP server that wraps RFdiffusion/ProteinMPNN/ESMFold/AlphaFold2-Multimer and
  deploys them to [Modal](https://modal.com) serverless GPUs. See "Fixes made here"
  below — this is not upstream-clean.
- **`targets/`** — the real target structures used: `4zqk_full.pdb` (full PD-1/PD-L1
  co-crystal) and `pdl1_target.pdb` (PD-L1 chain A only, the actual design target).
- **`debug/`** — scripts kept for transparency rather than deleted: the
  `_diag_*` scripts that root-caused a real production hang (JAX/XLA GPU
  memory allocator issue in `predict_complex`), and the `_run_phase_*`
  scripts used to find/fix the `design_for_interface` bugs and produce the
  results in `RESULTS_UPDATE.md`.
- **`drafts/`** — earlier presentation drafts, superseded by the `.pptx`. Kept for
  history, not canonical.

## Fixes made here (not upstream)

`protein-design-mcp` as cloned was not directly deployable. Real bugs found and fixed
in this local copy, documented with file:line references in
`TECHNICAL_APPENDIX.md` section 7:

1. `mcp` SDK dependency was unpinned and resolved an incompatible 2.x release that broke
   tool registration entirely — pinned to `>=1.0.0,<2.0.0`.
2. Missing `pyrsistent` dependency — RFdiffusion's symmetry module failed to import.
3. RFdiffusion model-weight symlink bug — weights downloaded to persistent storage were
   never linked into the path the inference script actually reads, breaking every fresh
   container.
4. AlphaFold2/ColabFold was listed as available but its backend had been stripped from
   the deployment — installed from scratch in its own isolated environment.
5. `predict_complex` hung indefinitely under real pipeline memory pressure (root-caused
   via `py-spy` stack traces on the live container — see `debug/`) — fixed with
   `XLA_PYTHON_CLIENT_PREALLOCATE=false` / `XLA_PYTHON_CLIENT_ALLOCATOR=platform`.
6. Modal web endpoints redirect (HTTP 303) requests running past 150s to a polling URL;
   the local MCP proxy didn't follow that redirect, so slow-but-successful calls looked
   like failures — fixed to poll correctly.

## Running it

```bash
cd protein-design-mcp
pip install -e .
modal setup                                  # authenticate to your own Modal account
modal deploy deploy/modal_app.py             # deploys the GPU backend
```

Then register it as an MCP server (e.g. `claude mcp add -s user protein-design --env
MODAL_URL=<printed-endpoint> -- python3 -m protein_design_mcp.modal_proxy`) and its
tools (`design_binder`, `predict_complex`, `optimize_sequence`, etc.) become available
to drive directly from an agent session.

## Open questions / next steps

See the deck's final slides. In short: can an agent reliably choose the next experiment
(which candidate to refine, when to widen the search, when to stop) and beat a fixed
parameter sweep at equal compute budget? Not yet tested — this run was human-directed
at every decision point.
