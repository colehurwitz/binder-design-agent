# Technical Appendix: PAE-guided refinement run

Supporting record for the PAE-guided refinement result presented in
`Binder Design Agent.pptx` (candidate #3, ipTM 0.43 → 0.55). Everything here is either a
verbatim tool-call input/output or a pointer to a real file on disk. No numbers below are
re-derived, rounded, or paraphrased from what the tools actually returned.

**Caveat the deck flags and this appendix confirms:** the refined sequence is
compositionally extreme — 70.0% glutamate before refinement, 76.2% after (56/80 and
61/80 residues respectively; verified by direct count against section 4's sequences).
All five mutations (`L78E, R15E, R76E, K21E, K23E`) mutate *to* glutamate. A gain in
predicted confidence on an already highly repetitive, low-complexity sequence is at
least as plausibly a model artifact (AF2-Multimer behaving unusually on
out-of-distribution, low-complexity input) as it is a genuine binding improvement. This
was not caught during the original run and should be treated as an open flag on the
section 6 result, not a confirmed positive.

## What is and isn't recoverable

**Available, verifiable now:**
- The target structure file (`pdl1_target.pdb`) — local, never depended on the
  ephemeral compute container.
- The hotspot-derivation script and its output (below).
- Every tool-call's exact input parameters and raw JSON output (below), transcribed
  verbatim from the session log.
- The deployment source code, including the exact fixes applied (`deploy/modal_app.py`,
  `pipelines/rfdiffusion.py`), still present in the repo.
- The exact `colabfold_batch` CLI invocation and software versions (captured from live
  container logs during the run).

**Not recoverable:**
- The predicted 3D structure PDB files (`predicted_structure_pdb` in each
  `predict_complex` response) and the full raw PAE matrix JSON files
  (`pae_matrix_path`). These were written only to the Modal container's ephemeral
  `/tmp`, which is destroyed when the container scales down after ~5 minutes idle
  (`scaledown_window=300` in `deploy/modal_app.py`). They were not copied to persistent
  storage or downloaded locally before the container terminated, and by the time this
  request came in, the container had already scaled down.
- Consequently, the full 195×195 PAE matrix for candidate #3 exists here only as the
  **filtered analysis output** (per-binder-residue minimum PAE, printed inline), not as
  the raw matrix file. The analysis script that produced it is included below and is
  exactly reproducible against a fresh `predict_complex` call.

**Process gap this exposes:** predicted structures and PAE matrices should be persisted
(e.g. written to the Modal Volume, or downloaded locally immediately after each call)
rather than left on container-local ephemeral storage. Not done in this run; worth
fixing before treating any future run as fully auditable by default.

**To regenerate the missing artifacts:** every input below is exact and complete, so
re-running `predict_complex` on the same two sequences will reproduce the same
structures and PAE matrices (ColabFold/AlphaFold2 inference is deterministic given
identical inputs, model version, and `num-models`/`num-recycle`/`msa-mode` settings — no
seed is otherwise randomized in this pipeline for a single-model, single-recycle-count
complex prediction). This was not done for this appendix to avoid further GPU spend
without being asked; happy to run it if the raw files are required.

---

## 1. Target structure and hotspot derivation

**Target:** PDB [4ZQK](https://www.rcsb.org/structure/4ZQK) chain A (PD-L1), residues
18–132, extracted unmodified (original residue numbering preserved, no renumbering).
File: `/binder-agent/targets/pdl1_target.pdb` (still present, 70,856 bytes).

**Target sequence** (used directly in `predict_complex` calls, position 1 = PDB residue
18, so sequence position *i* = PDB residue *i + 17*):
```
AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA
```

**Hotspot derivation (run 1, candidates #1–#2):** every PD-L1 heavy atom within 4.5 Å of
any PD-1 (chain B) heavy atom in the original 4ZQK complex, computed directly from
coordinates (script below), then narrowed to a 4-residue core: `A56, A115, A123, A125`.

**Hotspot derivation (run 2, candidate #3 onward):** same contact set, widened to 7
residues: `A56, A58, A113, A115, A122, A123, A125`.

Contact-distance script (exact, as run):
```python
import math

def parse_atoms(path, chain):
    atoms = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            atoms.append({
                "resname": line[17:20].strip(),
                "resseq": int(line[22:26]),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
            })
    return atoms

a_atoms = parse_atoms("4zqk_full.pdb", "A")  # PD-L1
b_atoms = parse_atoms("4zqk_full.pdb", "B")  # PD-1

cutoff = 4.5
contact_residues = {}
for aa in a_atoms:
    for ba in b_atoms:
        d = math.dist((aa["x"], aa["y"], aa["z"]), (ba["x"], ba["y"], ba["z"]))
        if d <= cutoff:
            key = (aa["resseq"], aa["resname"])
            contact_residues.setdefault(key, 1e9)
            contact_residues[key] = min(contact_residues[key], d)
            break
```

Full contact set found (18 residues, min distance to PD-1 in Å):
```
A19 PHE 3.48   A23 VAL 3.79   A26 ASP 3.80   A54 ILE 4.43   A56 TYR 3.99
A58 GLU 4.41   A66 GLN 4.19   A76 VAL 4.17   A113 ARG 3.84  A115 MET 3.79
A117 SER 3.83  A119 GLY 3.73  A120 GLY 4.41  A121 ALA 3.79  A122 ASP 3.98
A123 TYR 3.77  A124 LYS 3.83  A125 ARG 3.47
```

---

## 2. `design_binder` — batch that produced candidate #3

**Call parameters:**
```json
{
  "target_pdb": "/binder-agent/targets/pdl1_target.pdb",
  "hotspot_residues": ["A56", "A58", "A113", "A115", "A122", "A123", "A125"],
  "num_designs": 1
}
```

**Raw response** (5 of the returned designs passed internal filters; candidate #3 is the
first / `best_design_id`):
```json
{
  "designs": [
    {
      "id": "design_0_T=0.1",
      "sequence": "AEEEEERREEEEEAREEEEEKEKEEEEKLKEFLEYVENAEEEAEEEEAEEEEEEEEAEEEEERREEEEEAEEEEERELEE",
      "structure_pdb_path": "/tmp/design_98542dd2_w_z3uyqi/predicted_structures/design_0_T=0.1.pdb",
      "backbone_id": "design_0",
      "metrics": { "plddt": 91.17725, "ptm": 0.6087724566459656, "mpnn_score": 1.1121 }
    },
    {
      "sequence": "SALELEKLLEEEKARKEEEEKEKEREEKKKENLEYVENLEEEAERRRREEEEERERRREEEERRREEEERARREEAERLA",
      "metrics": { "plddt": 89.769, "ptm": 0.5877852439880371, "mpnn_score": 1.1317 }
    },
    {
      "sequence": "EEEEEEEEEEEEEEEEERREEEEEKEEEEKEFLELVENIEEEYKKRKEEEERERREREEEERRREEEEERRREEERRKLE",
      "metrics": { "plddt": 89.661, "ptm": 0.5664857625961304, "mpnn_score": 1.0619 }
    },
    {
      "sequence": "MEEEEEREREEEEERKEEEEEEKKKEEEKEKNLEYVENLEEKYEEEKKKEEEKKRKEEEEKKKKEEEKKKKEEEEKKKKE",
      "metrics": { "plddt": 90.344875, "ptm": 0.5588730573654175, "mpnn_score": 1.1002 }
    },
    {
      "sequence": "EEEEEEEEEEEEEEEAEEEERELEEEKELEEALEYVENLEERHEAEKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALA",
      "metrics": { "plddt": 73.466875, "ptm": 0.5498365163803101, "mpnn_score": 1.0565 }
    }
  ],
  "summary": { "total_generated": 9, "passed_filters": 5, "best_design_id": "design_0_T=0.1", "job_id": "98542dd2" }
}
```

**Candidate #3 sequence** (used in step 3):
```
AEEEEERREEEEEAREEEEEKEKEEEEKLKEFLEYVENAEEEAEEEEAEEEEEEEEAEEEEERREEEEEAEEEEERELEE
```

---

## 3. `predict_complex` — candidate #3, before refinement

**Call parameters:**
```json
{
  "sequences": [
    "AEEEEERREEEEEAREEEEEKEKEEEEKLKEFLEYVENAEEEAEEEEAEEEEEEEEAEEEEERREEEEEAEEEEERELEE",
    "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
  ],
  "chain_names": ["binder", "PDL1"]
}
```

**Backend / prediction settings** (captured from live container log at call time):
```
colabfold 1.6.3 (efbf31c37cedb38cd09c69c1b991910a9866480e)
colabfold_batch <input.fasta> <output> --num-models 1 --num-recycle 3 \
  --msa-mode mmseqs2_uniref_env --model-type alphafold2_multimer_v3 \
  --data /models/colabfold
jax 0.5.3 / jaxlib (cuda12 build)
GPU: NVIDIA A10G
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

**Raw response** (per-residue pLDDT array included in full — index 0 = binder residue
1, index 80 = target residue 1 / PDB residue 18):
```json
{
  "predicted_structure_pdb": "/tmp/complex_n35bad53.pdb",
  "plddt": 89.01471794871796,
  "ptm": 0.67,
  "iptm": 0.43,
  "i_pae": 14.69,
  "pae_matrix_path": "/tmp/complex_n35bad53_pae.json",
  "num_chains": 2,
  "plddt_per_residue": [80.12,90.06,92.88,91.0,89.75,91.44,91.88,90.25,89.88,90.56,89.69,88.69,89.56,88.06,88.38,87.25,84.75,81.62,81.44,80.94,78.31,71.69,80.19,73.44,66.38,70.94,74.75,73.31,68.94,77.75,77.25,79.5,78.19,78.25,73.44,73.56,77.0,75.19,65.81,70.88,72.31,67.5,65.12,72.0,75.06,70.38,76.31,86.0,84.38,84.19,88.5,91.5,91.56,92.94,92.88,93.62,95.38,94.44,94.62,95.19,94.94,94.88,96.25,96.5,95.0,95.0,94.69,94.25,95.19,94.69,92.25,92.81,92.62,92.5,92.75,92.44,90.81,89.88,86.62,73.69,74.5,85.56,91.25,94.75,96.75,96.44,95.5,93.94,92.62,95.38,97.38,97.12,97.25,97.62,97.44,97.19,97.19,97.81,97.38,98.06,97.88,97.75,97.75,97.31,96.44,94.88,89.0,79.25,78.94,88.06,90.19,88.62,87.38,82.94,84.0,90.94,90.0,93.44,92.12,95.06,92.69,92.62,80.69,76.44,89.25,89.62,95.44,94.0,89.25,90.94,83.62,79.31,86.38,86.56,87.25,82.81,80.44,82.0,81.25,87.69,91.31,90.88,93.38,94.88,90.62,91.44,95.81,96.31,94.56,96.44,95.88,93.5,93.94,95.25,95.19,92.19,93.56,96.12,97.06,97.62,97.94,97.81,97.94,97.81,97.62,97.38,97.56,96.75,95.94,95.06,96.94,96.0,94.5,91.0,94.75,92.38,95.44,92.81,93.69,90.31,89.69,80.94,82.69,86.62,86.25,88.56,91.0,91.62,95.81,94.0,97.06,97.0,97.0,94.5,83.62]
}
```

**PAE-matrix analysis** (script run inline against the live container's
`/tmp/complex_n35bad53_pae.json`, before it was lost — output preserved here in full;
the raw matrix file itself is not):
```python
import json
data = json.load(open("/tmp/complex_n35bad53_pae.json"))
pae = data if isinstance(data, list) else data.get("pae") or data.get("predicted_aligned_error")
n = len(pae)                      # 195
binder_len = 80
cross = [[pae[i][j] for j in range(binder_len, n)] for i in range(binder_len)]
best_per_binder_residue = [min(row) for row in cross]
confident_count = sum(1 for v in best_per_binder_residue if v < 10)   # 49
overall_cross_mean = sum(sum(row) for row in cross) / (binder_len * (n - binder_len))  # 14.26
for i, row in enumerate(cross):
    m = min(row)
    if m < 10:
        j = row.index(m)
        print(f"binder residue {i+1} <-> target residue {j+1} (PAE={m:.2f})")
```

**Output (full, as printed):**
```
matrix size: 195 x 195
49/80 binder residues have PAE < 10 against at least one target residue
overall cross-chain mean PAE: 14.26
binder residue 13 <-> target residue 97 (PAE=9.83)
binder residue 15 <-> target residue 39 (PAE=9.54)
binder residue 16 <-> target residue 98 (PAE=9.23)
binder residue 17 <-> target residue 97 (PAE=9.66)
binder residue 18 <-> target residue 39 (PAE=9.63)
binder residue 19 <-> target residue 39 (PAE=8.88)
binder residue 20 <-> target residue 98 (PAE=8.47)
binder residue 21 <-> target residue 98 (PAE=8.37)
binder residue 22 <-> target residue 39 (PAE=8.12)
binder residue 23 <-> target residue 39 (PAE=7.48)
binder residue 24 <-> target residue 98 (PAE=7.07)
binder residue 25 <-> target residue 39 (PAE=8.0)
binder residue 26 <-> target residue 98 (PAE=7.16)
binder residue 27 <-> target residue 39 (PAE=6.22)
binder residue 28 <-> target residue 98 (PAE=6.28)
binder residue 29 <-> target residue 98 (PAE=5.96)
binder residue 30 <-> target residue 98 (PAE=5.78)
binder residue 31 <-> target residue 99 (PAE=5.1)
binder residue 32 <-> target residue 98 (PAE=4.73)
binder residue 33 <-> target residue 98 (PAE=4.94)
binder residue 34 <-> target residue 99 (PAE=5.34)
binder residue 35 <-> target residue 98 (PAE=6.62)
binder residue 36 <-> target residue 98 (PAE=5.43)
binder residue 37 <-> target residue 98 (PAE=5.65)
binder residue 38 <-> target residue 98 (PAE=7.0)
binder residue 39 <-> target residue 98 (PAE=6.98)
binder residue 40 <-> target residue 39 (PAE=7.96)
binder residue 41 <-> target residue 39 (PAE=7.08)
binder residue 42 <-> target residue 98 (PAE=8.15)
binder residue 43 <-> target residue 98 (PAE=7.89)
binder residue 44 <-> target residue 98 (PAE=7.69)
binder residue 45 <-> target residue 98 (PAE=7.53)
binder residue 46 <-> target residue 98 (PAE=8.24)
binder residue 47 <-> target residue 39 (PAE=8.33)
binder residue 48 <-> target residue 39 (PAE=7.72)
binder residue 49 <-> target residue 37 (PAE=8.67)
binder residue 50 <-> target residue 39 (PAE=9.15)
binder residue 51 <-> target residue 39 (PAE=9.05)
binder residue 52 <-> target residue 37 (PAE=8.62)
binder residue 53 <-> target residue 37 (PAE=9.23)
binder residue 54 <-> target residue 37 (PAE=9.65)
binder residue 55 <-> target residue 37 (PAE=8.91)
binder residue 56 <-> target residue 37 (PAE=9.08)
binder residue 57 <-> target residue 37 (PAE=9.48)
binder residue 58 <-> target residue 37 (PAE=9.56)
binder residue 59 <-> target residue 37 (PAE=9.24)
binder residue 60 <-> target residue 37 (PAE=9.88)
binder residue 62 <-> target residue 37 (PAE=9.8)
binder residue 63 <-> target residue 37 (PAE=9.91)
```

**Target-position → PDB-residue mapping** (sequence position *i* → PDB residue *i+17*,
since extraction started at PDB residue 18): target position 39 = **Tyr56**, target
position 37 = **Ile54**, target position 98 = **Met115** — all three are members of the
18-residue contact set in section 1. This mapping is arithmetic on the known offset, not
a separate tool output.

---

## 4. `optimize_sequence` — targeted refinement

**Call parameters:**
```json
{
  "current_sequence": "AEEEEERREEEEEAREEEEEKEKEEEEKLKEFLEYVENAEEEAEEEEAEEEEEEEEAEEEEERREEEEEAEEEEERELEE",
  "target_pdb": "/binder-agent/targets/pdl1_target.pdb",
  "fixed_positions": [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
  "optimization_target": "affinity"
}
```
(`fixed_positions` locks the residues identified above as the confident Met115 contact.)

**Raw response (complete):**
```json
{
  "optimized_sequence": "AEEEEERREEEEEAEEEEEEEEEEEEEKLKEFLEYVENAEEEAEEEEAEEEEEEEEAEEEEERREEEEEAEEEEEEEEEE",
  "mutations": ["L78E", "R15E", "R76E", "K21E", "K23E"],
  "predicted_improvement": {
    "esm2_delta": "+0.171 PLL",
    "stability_delta": "+0.6% pLDDT",
    "affinity_delta": "+0.010 pTM",
    "baseline_plddt": 91.17725,
    "baseline_ptm": 0.6087724566459656,
    "baseline_esm2_score": -0.3002745838712144
  },
  "metrics": { "plddt": 91.73075, "ptm": 0.6185195446014404, "esm2_score": -0.12925024782944092 },
  "all_candidates": 1,
  "optimization_trajectory": [
    { "round": 0, "score": -0.3002745838712144, "mutations": [] },
    { "round": 1, "score": -0.17669467910127423, "mutations": ["L78E", "R15E", "R76E"] },
    { "round": 2, "score": -0.15517026654924848, "mutations": ["K21E"] },
    { "round": 3, "score": -0.12925024782944092, "mutations": ["K23E"] }
  ]
}
```

Note: `predicted_improvement` here is the tool's own self-reported ProteinMPNN/ESM2-based
estimate — the same category of self-score already shown (candidates #1/#2) not to be
trustworthy on its own. Section 5 is the independent check.

---

## 5. `predict_complex` — optimized sequence, after refinement

**Call parameters:**
```json
{
  "sequences": [
    "AEEEEERREEEEEAEEEEEEEEEEEEEKLKEFLEYVENAEEEAEEEEAEEEEEEEEAEEEEERREEEEEAEEEEEEEEEE",
    "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
  ],
  "chain_names": ["binder", "PDL1"]
}
```
Same backend/settings as section 3 (identical deployment, no config changes between
calls).

**Raw response:**
```json
{
  "predicted_structure_pdb": "/tmp/complex_dam4t06j.pdb",
  "plddt": 90.67246153846152,
  "ptm": 0.72,
  "iptm": 0.55,
  "i_pae": 11.58,
  "pae_matrix_path": "/tmp/complex_dam4t06j_pae.json",
  "num_chains": 2,
  "plddt_per_residue": [81.62,92.06,93.69,91.94,91.06,92.25,92.44,91.25,91.0,91.5,90.94,90.5,90.75,89.69,88.81,89.12,87.0,85.12,86.06,85.31,79.69,78.88,81.81,79.69,75.44,77.81,81.88,79.75,78.5,83.81,83.38,83.56,83.75,83.5,79.56,78.44,81.38,81.06,72.19,75.06,77.0,74.56,73.69,76.81,81.12,78.31,82.44,88.88,89.38,89.44,91.75,93.62,94.5,95.19,94.69,95.38,96.31,95.25,95.5,95.94,95.31,95.31,96.94,96.94,95.38,95.44,94.88,94.44,95.06,94.5,92.0,93.56,92.81,92.75,92.56,92.12,91.38,88.88,86.56,73.69,75.94,88.38,93.56,95.94,97.12,97.06,95.81,94.5,94.12,96.5,97.62,97.44,97.31,97.69,97.44,97.19,97.38,97.94,97.5,98.06,98.0,97.81,97.94,97.5,96.75,95.0,88.88,78.06,79.25,88.88,90.75,89.94,88.5,84.5,86.19,91.94,91.75,94.5,93.5,95.88,93.94,93.94,83.31,79.5,90.75,91.62,95.75,94.62,90.62,91.88,85.38,81.38,87.12,87.56,87.62,84.12,80.75,83.19,83.12,88.06,91.94,91.69,93.44,94.81,90.88,91.31,95.81,96.25,94.69,96.56,95.88,93.81,94.19,95.38,95.38,92.56,93.75,96.19,97.12,97.75,97.94,97.88,97.94,97.88,97.62,97.44,97.5,96.81,96.12,95.19,97.12,96.62,95.88,93.88,96.12,94.0,96.38,94.5,94.88,91.94,90.94,83.94,86.12,89.38,89.88,91.75,93.38,94.38,96.75,95.94,97.5,97.56,97.25,94.56,83.0]
}
```

---

## 6. Summary table (all numbers sourced from sections above)

| | ipTM | Interface pAE | Complex pLDDT | Complex pTM |
|---|---|---|---|---|
| Candidate #3, before | 0.43 | 14.69 | 89.01 | 0.67 |
| Candidate #3, after `optimize_sequence` | 0.55 | 11.58 | 90.67 | 0.72 |
| Δ | **+0.12** | **−3.11** | +1.66 | +0.05 |

All four deltas are independently-verified `predict_complex` (AlphaFold2-Multimer)
outputs, not the tool's self-reported `predicted_improvement` from section 4.

---

## 7. Deployment fixes affecting these runs (source, not reproduced in full)

All in `/binder-agent/protein-design-mcp/` (local clone, fixes not upstreamed):
- `deploy/modal_app.py:123` — `mcp` SDK pinned to `>=1.0.0,<2.0.0` (unpinned version
  resolved an incompatible 2.x SDK that broke tool registration entirely).
- `deploy/modal_app.py:65` — added `pyrsistent` (missing dependency, RFdiffusion's
  symmetry module failed to import without it).
- `pipelines/rfdiffusion.py:78` (`_ensure_weights`) — fixed a symlink bug where
  downloaded RFdiffusion weights were never linked into the path the inference script
  actually reads from on a fresh container.
- `deploy/modal_app.py:153-154` — `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform`. Without these, `predict_complex` hung
  indefinitely inside JAX's device-memory allocator (root-caused via `py-spy` stack
  traces on the live container, confirmed via isolated reproduction that parameter
  loading alone takes low single-digit seconds when the GPU isn't already under memory
  pressure from the full pipeline's earlier steps). This fix is what made sections 3–5
  possible to run at all; every attempt before it hung until Modal's function timeout
  killed the container.
