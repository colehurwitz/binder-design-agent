import json

with open("/tmp/complex_n35bad53_pae.json") as f:
    data = json.load(f)

pae = data if isinstance(data, list) else data.get("pae") or data.get("predicted_aligned_error")
n = len(pae)
print("matrix size:", n, "x", len(pae[0]))

binder_len = 80  # first chain
target_len = n - binder_len
print(f"binder residues: 1-{binder_len}, target residues: {binder_len+1}-{n}")

# cross-chain block: binder rows vs target columns
cross = [[pae[i][j] for j in range(binder_len, n)] for i in range(binder_len)]

# per binder-residue min PAE against ANY target residue (best-case confidence for that residue)
best_per_binder_residue = [min(row) for row in cross]

print("\nper-binder-residue best (min) PAE against target:")
for i, v in enumerate(best_per_binder_residue):
    marker = " <-- confident" if v < 10 else ""
    print(f"  binder residue {i+1}: min PAE = {v:.2f}{marker}")

confident_count = sum(1 for v in best_per_binder_residue if v < 10)
print(f"\n{confident_count}/{binder_len} binder residues have PAE < 10 against at least one target residue")

overall_cross_mean = sum(sum(row) for row in cross) / (binder_len * target_len)
print(f"overall cross-chain mean PAE: {overall_cross_mean:.2f}")

# which target residue is each confident binder residue closest to?
if confident_count > 0:
    print("\nconfident binder residues and their best-matching target residue:")
    for i, row in enumerate(cross):
        m = min(row)
        if m < 10:
            j = row.index(m)
            print(f"  binder residue {i+1} <-> target residue {j+1} (PAE={m:.2f})")
