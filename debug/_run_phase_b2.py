import asyncio
import json

WINNING_SEQ = "ELAKLKAEAEKLELEALKVRREALKKEAKAKELKEEAKATGNKELEEKAKKAEKEAEKLKKKAEELKKKAEELKKKAEEL"
TARGET_SEQ = (
    "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNA"
    "ALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
)


def pae_contacts(pae, binder_len):
    n = len(pae)
    cross = [[pae[i][j] for j in range(binder_len, n)] for i in range(binder_len)]
    confident_count = sum(1 for row in cross if min(row) < 10)
    overall_cross_mean = sum(sum(row) for row in cross) / (binder_len * (n - binder_len))
    contacts = []
    for i, row in enumerate(cross):
        m = min(row)
        if m < 10:
            j = row.index(m)
            contacts.append((i + 1, j + 1, j + 18, m))
    return confident_count, overall_cross_mean, contacts


async def main():
    from protein_design_mcp.pipelines.alphafold2 import AlphaFold2Runner
    from protein_design_mcp.pipelines.proteinmpnn import ProteinMPNNRunner
    from protein_design_mcp.pipelines.esmfold import ESMFoldRunner

    af2 = AlphaFold2Runner()
    mpnn = ProteinMPNNRunner()
    esmfold = ESMFoldRunner()

    print("=== re-predicting the winning complex (to save it properly + get a real structure) ===", flush=True)
    result = await af2.predict_complex(
        sequences=[WINNING_SEQ, TARGET_SEQ], chain_names=["binder", "PDL1"]
    )
    print(f"REPRODUCED: plddt={result.plddt} ptm={result.ptm} iptm={result.iptm}", flush=True)

    # Save the real structure + full PAE matrix locally in-container immediately
    complex_pdb_path = "/tmp/winning_complex.pdb"
    with open(complex_pdb_path, "w") as f:
        f.write(result.pdb_string)

    chains_present = sorted(set(
        line[21] for line in result.pdb_string.splitlines()
        if line.startswith("ATOM")
    ))
    print("chains present in predicted complex PDB:", chains_present, flush=True)

    # Sanity check which chain is the binder: count residues per chain, binder is len(WINNING_SEQ)
    resnums_by_chain = {}
    for line in result.pdb_string.splitlines():
        if line.startswith("ATOM"):
            c = line[21]
            resnums_by_chain.setdefault(c, set()).add(int(line[22:26]))
    for c, resnums in resnums_by_chain.items():
        print(f"  chain {c}: {len(resnums)} residues", flush=True)
    binder_chain = [c for c, r in resnums_by_chain.items() if len(r) == len(WINNING_SEQ)]
    assert len(binder_chain) == 1, f"couldn't uniquely identify binder chain: {resnums_by_chain}"
    binder_chain = binder_chain[0]
    print(f"binder_chain identified as: {binder_chain}", flush=True)

    pae = result.pae_matrix
    pae = pae.tolist() if hasattr(pae, "tolist") else pae
    confident_count, overall_mean, contacts = pae_contacts(pae, len(WINNING_SEQ))
    print(f"{confident_count}/{len(WINNING_SEQ)} confident, mean={overall_mean:.2f}", flush=True)

    with open("/tmp/winning_complex_analysis.json", "w") as f:
        json.dump({
            "sequence": WINNING_SEQ, "plddt": result.plddt, "ptm": result.ptm,
            "iptm": result.iptm, "contacts": contacts, "pae_matrix": pae,
            "pdb_string": result.pdb_string,
        }, f)
    print("saved winning complex + full analysis to /tmp/winning_complex_analysis.json", flush=True)

    if not contacts:
        print("no confident contacts -- stopping", flush=True)
        return

    positions = sorted(set(c[0] for c in contacts))
    print(f"=== refining: locking positions {positions} on the REAL sequence-bearing complex ===", flush=True)

    designs = await mpnn.design_for_interface(
        complex_pdb=complex_pdb_path,
        design_chain=binder_chain,
        interface_residues=[],
        output_dir="/tmp/phase_b2_mpnn",
        fixed_positions=positions,
    )
    print(f"{len(designs)} refined designs returned", flush=True)
    for d in designs:
        print(d.get("id"), d.get("sequence"), "score=", d.get("score"), flush=True)

    best = min(designs, key=lambda d: d.get("score", 1e9))
    seq2 = best["sequence"]
    if "/" in seq2:
        seq2 = seq2.split("/")[-1]
    print("REFINED SEQUENCE:", seq2, flush=True)
    print("positions changed vs winning seq:",
          [i + 1 for i, (a, b) in enumerate(zip(WINNING_SEQ, seq2)) if a != b], flush=True)

    pred2 = await esmfold.predict_structure(seq2)
    print(f"refined self-score: plddt={pred2.plddt} ptm={pred2.ptm}", flush=True)

    print("=== independently re-verifying refined sequence ===", flush=True)
    result2 = await af2.predict_complex(sequences=[seq2, TARGET_SEQ], chain_names=["binder", "PDL1"])
    print(f"REFINED RESULT: plddt={result2.plddt} ptm={result2.ptm} iptm={result2.iptm}", flush=True)

    with open("/tmp/refined_complex_analysis.json", "w") as f:
        json.dump({
            "sequence": seq2, "plddt": result2.plddt, "ptm": result2.ptm,
            "iptm": result2.iptm, "pdb_string": result2.pdb_string,
        }, f)

    print("=== DONE ===", flush=True)


asyncio.run(main())
