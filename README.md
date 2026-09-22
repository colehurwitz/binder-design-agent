# Binder Design Agent

Hackathon project: an AI agent (Claude, via Claude Code) that drives a real protein
binder design pipeline — RFdiffusion → ProteinMPNN → ESMFold → AlphaFold2-Multimer —
running on serverless GPUs, and uses the models' own output to decide what to try next.

**Start here: [`Binder Design Agent.pptx`](./Binder%20Design%20Agent.pptx)** — the
presentation, and the canonical summary of what was built and found.

## The pipeline

Five MCP tool calls, each a real model running on a Modal GPU:

```
INPUT   target PDB + hotspot residues
           │
           ▼
1. design_binder      RFdiffusion    →  candidate 3D backbones (shape only, no sequence)
                       │ conditioned on target + hotspots
                       ▼
2.                     ProteinMPNN    →  amino acid sequence for that backbone
                       │ (inverse folding)
                       ▼
3.                     ESMFold        →  refold the sequence ALONE (no target)
                       │                 → pLDDT, pTM  (self-score / monomer confidence)
                       ▼
4. (agent/human)       filter         →  drop weak candidates by self-score
                       │
                       ▼
5. predict_complex     AlphaFold2-    →  fold binder + target TOGETHER, different model
                       Multimer          → ipTM, interface pAE  (independent verification)
                       ▼
OUTPUT  ranked candidate(s): sequence + predicted complex + independent confidence scores
```

Steps 1–4 share one model family (ESMFold) for both generation and self-scoring — a
binder can score well there just by being a stable, well-folded shape, whether or not it
actually binds anything. Step 5 uses a completely different model to check the thing
steps 1–4 can't: does it dock against the real target. That's the gap the whole project
is about — see the deck and `TECHNICAL_APPENDIX.md` for a run where step 5 overturned a
confident-looking step 1–4 result.

## Headline result

Ran against a real target (PD-L1, PDB 4ZQK) with hotspots derived directly from the
real PD-1/PD-L1 crystal contacts. Best candidate scored well on self-reported monomer
metrics (pLDDT 93.5, pTM 0.85) but failed independent complex verification (ipTM 0.39
vs. a 0.8 screening bar). A PAE-guided, targeted refinement improved that to ipTM 0.55 —
still below threshold, and flagged with an open caveat (the refined sequence is
unusually glutamate-rich, so the gain may be partly a model artifact rather than a real
binding improvement — see the appendix).

No experimental binding data. This is a model-prediction pipeline and its results
should be read as that, not as validated binder discovery.

## Repo layout

- **`Binder Design Agent.pptx`** — the presentation. Primary artifact.
- **`TECHNICAL_APPENDIX.md`** — full audit trail for the PAE-guided refinement result:
  exact tool-call inputs/outputs, scripts, backend settings, and an explicit accounting
  of which raw artifacts (predicted structures, full PAE matrices) were and weren't
  preserved.
- **`protein-design-mcp/`** — vendored, patched clone of
  [`jasonkim8652/protein-design-mcp`](https://github.com/jasonkim8652/protein-design-mcp),
  the MCP server that wraps RFdiffusion/ProteinMPNN/ESMFold/AlphaFold2-Multimer and
  deploys them to [Modal](https://modal.com) serverless GPUs. See "Fixes made here"
  below — this is not upstream-clean.
- **`targets/`** — the real target structures used: `4zqk_full.pdb` (full PD-1/PD-L1
  co-crystal) and `pdl1_target.pdb` (PD-L1 chain A only, the actual design target).
- **`debug/`** — the diagnostic scripts written to root-cause a real production hang
  (JAX/XLA GPU memory allocator issue in `predict_complex`), kept for transparency
  rather than deleted once the bug was found.
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
