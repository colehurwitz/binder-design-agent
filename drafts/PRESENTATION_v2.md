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
INPUT   4ZQK.pdb — PD-L1 chain A (the target)
        hotspots: Tyr56, Met115, Tyr123, Arg125
        binder length range
           │
           ▼
                                             protein-design-mcp on Modal (GPU)
  ┌─────────────┐                            ┌────────────────────────────────────────┐
  │             │        design_backbone ───▶│ RFdiffusion                            │
  │             │                            │  in   target PDB + hotspot residues    │
  │    Agent    │◀──────────── backbone PDBs │  out  N backbones — Cα only, no seq    │
  │   (Claude)  │                            └────────────────────────────────────────┘
  │             │                            ┌────────────────────────────────────────┐
  │             │        design_sequence ───▶│ ProteinMPNN                            │
  │  holds one  │                            │  in   one backbone PDB                 │
  │ MCP session │◀───────── sequence + score │  out  sequence + MPNN score            │
  │  to Modal;  │                            └────────────────────────────────────────┘
  │  feeds each │                            ┌────────────────────────────────────────┐
  │ result into │        validate_design ───▶│ ESMFold                                │
  │   the next  │                            │  in   sequence alone — no target       │
  │     call    │◀─────────────── pLDDT, pTM │  out  monomer PDB + pLDDT, pTM         │
  │             │                            └────────────────────────────────────────┘
  │             │                            
  │             │                              agent-side filter on self-scores
  │             │                              (this run: 8 in, 1 forward)
  │             │                            
  │             │                            ┌────────────────────────────────────────┐
  │             │        predict_complex ───▶│ AlphaFold2-Multimer (ColabFold)        │
  │             │                            │  in   binder seq + PD-L1 together      │
  │             │◀──────────────── ipTM, pAE │  out  complex PDB + ipTM, interface pAE│
  │             │                            └────────────────────────────────────────┘
  └─────────────┘
           │
           ▼
OUTPUT  [ASPIRATIONAL — not built] ranked batch of designs, each with a
        sequence, a predicted complex structure, and interface scores from
        a model that had no part in generating it. Today this run produces
        one hand-picked candidate, and the interface scores are still pending.
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
  orchestration script standing between the agent and the biology tools. In this run the
  agent's substantive contribution was at setup — deriving the hotspot set from the 4ZQK
  coordinates rather than from published residue numbering — and the pipeline steps
  themselves ran in fixed order. Making the agent decide what to try next mid-campaign is
  design intent, not something demonstrated here; the next section sets out where that
  judgment actually pays.

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
binder's own standalone fold. *[Results pending — fill in interface pTM / ipTM and
contact residues here once complete.]*

## Where the agent adds value

**None of what follows has been built.** This run was a fixed, human-configured sequence
with the agent making one real decision (the hotspot derivation). This section is the
case for where agent judgment *would* pay, and is the argument for the work in the next
section — not a description of current behaviour.

Steps 1–5 are a fixed sequence. Run in order against a fixed config, they are a build
script, and an agent that executes them in order is a slower, less reproducible build
script. The agent earns its place only where the next action depends on the content of
the previous result and the branching space is too large to enumerate in advance.

That is true at the two ends of the pipeline, not in the middle.

**Intake: target to design spec.** Going from "design a binder against PD-L1" to a
runnable config requires selecting the structure and the relevant chain, distinguishing
the biological interface from crystal-packing contacts, choosing hotspots, and setting
binder length and fold class. This is unstructured judgment over literature and
coordinates, and it is the one part of this run the agent genuinely performed.

**Triage: results to next spec.** When a batch fails, the corrective action depends on
how it failed. A flat, polar interface, a bad hotspot set, a mis-specified length range,
and mode collapse onto a single motif all produce poor scores and each implies a
different retry. Encoding this deterministically requires enumerating failure modes that
have not been enumerated.

**Budget allocation.** Which survivors justify expensive independent verification, when
to stop sampling a hotspot set that is not producing candidates, and when to widen the
search. Cheap to express as judgment, brittle as fixed thresholds.

**Ad hoc analysis.** Questions that arise once and do not justify a script: why the
candidates converged on a single fold, how the designed interface compares to the native
PD-1 contacts, which residues carry the predicted contacts.

The agent does not belong in the middle. Orchestrating a fixed DAG, applying numeric
score cutoffs, and any step that should be bit-identical on rerun are better served by
the deterministic pipeline.

This reframes the deliverable. The pipeline would become a single tool invoked with a
structured config rather than five tools called in sequence, with the agent running a
campaign loop around it: propose a spec, execute, diagnose, propose the next one. That
loop does not exist yet, and whether it beats a fixed parameter sweep at equal compute
budget is an open question — but it is a measurable one, and worth treating as the
evaluation criterion rather than assuming the agent framing pays for itself.

## What's next

The current result (pLDDT/pTM from a monomer refold) shows the binder folds correctly
on its own — it does not yet show that it binds PD-L1. Closing that gap is the priority,
ahead of adding new capabilities:

1. **Report the right metrics** — ipTM and interface pAE from the AF2-Multimer complex
   prediction (not monomer pLDDT/pTM); field-standard thresholds are ipTM > 0.8 and
   interface pAE < 10. Also add self-consistency RMSD: refold the designed sequence and
   check it lands within ~2 Å of the RFdiffusion backbone it was designed for — checks
   that the geometry reproduces, not just that confidence does.
2. **Add a negative control** — run the identical pipeline against scrambled hotspots or
   an unrelated target and report the score distribution alongside the real one, so a
   given score has a reference point.
3. **Generalize target intake** — turn hotspot derivation (PDB + partner chain → contact
   residues within 4.5 Å) into a reusable tool, so a new target is one sentence instead
   of a manual analysis.
4. **Scale the funnel, budget-aware** — published RFdiffusion+MPNN hit rates are well
   under 1%, so the pipeline's value is running many candidates cheaply and discarding
   almost all of them. The agent should apply cheap self-score filters first and only
   spend AF2-Multimer compute on survivors. Today's run generated 8 candidates and
   verified only the single best one by hand; the selection step is what needs to become
   a real, budget-aware filter over a much larger batch.
5. **AF2 initial-guess rescoring** (Bennett et al. 2023) — seed AF2 with the designed
   complex's own coordinates instead of folding from scratch; sharper discrimination
   than naive multimer folding, but requires custom inference code, not just a config
   flag on the existing ColabFold call.
6. **Lab-ready output** — codon-optimized DNA, expression-liability checks (free
   cysteines, hydrophobic patches, aggregation-prone stretches), and a ranked shortlist
   — the difference between an interesting result and one a wet lab can order tomorrow.
7. **PyRosetta-based physics scoring** (binding energy, buried surface area) as a further
   independent signal, once the funnel feeding it is validated.
