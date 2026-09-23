import asyncio
import json
import urllib.request


HOTSPOTS = ["A56", "A58", "A113", "A115", "A122", "A123", "A125"]


def pae_contacts(pae, binder_len):
    n = len(pae)
    cross = [[pae[i][j] for j in range(binder_len, n)] for i in range(binder_len)]
    best_per_binder_residue = [min(row) for row in cross]
    confident_count = sum(1 for v in best_per_binder_residue if v < 10)
    overall_cross_mean = sum(sum(row) for row in cross) / (binder_len * (n - binder_len))
    contacts = []
    for i, row in enumerate(cross):
        m = min(row)
        if m < 10:
            j = row.index(m)
            contacts.append((i + 1, j + 1, j + 18, m))
    return confident_count, overall_cross_mean, contacts


async def main():
    urllib.request.urlretrieve("https://files.rcsb.org/download/4ZQK.pdb", "/tmp/4zqk_full.pdb")
    with open("/tmp/4zqk_full.pdb") as f, open("/tmp/pdl1_target.pdb", "w") as out:
        out.write("HEADER    PD-L1\n")
        for line in f:
            if line.startswith("ATOM") and line[21] == "A":
                out.write(line)
        out.write("TER\nEND\n")

    target_seq = []
    aa3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
              "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
              "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    seen = set()
    with open("/tmp/pdl1_target.pdb") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resseq = int(line[22:26])
                if resseq not in seen:
                    seen.add(resseq)
                    target_seq.append(aa3to1.get(line[17:20].strip(), "X"))
    target_seq = "".join(target_seq)
    print("target_seq len:", len(target_seq), flush=True)

    from protein_design_mcp.pipelines.rfdiffusion import RFdiffusionRunner
    from protein_design_mcp.pipelines.proteinmpnn import ProteinMPNNRunner
    from protein_design_mcp.pipelines.esmfold import ESMFoldRunner
    from protein_design_mcp.pipelines.alphafold2 import AlphaFold2Runner

    rfdiffusion = RFdiffusionRunner()
    mpnn = ProteinMPNNRunner()
    esmfold = ESMFoldRunner()
    af2 = AlphaFold2Runner()

    print("=== generating backbone ===", flush=True)
    backbones = await rfdiffusion.generate_backbones(
        target_pdb="/tmp/pdl1_target.pdb",
        hotspot_residues=HOTSPOTS,
        output_dir="/tmp/phase_b_backbone",
        num_designs=1,
        binder_length=80,
    )
    backbone_pdb = backbones[0]["pdb_path"]
    print("backbone:", backbone_pdb, flush=True)

    print("=== initial interface-aware design ===", flush=True)
    designs = await mpnn.design_for_interface(
        complex_pdb=backbone_pdb, design_chain="B",
        interface_residues=HOTSPOTS, output_dir="/tmp/phase_b_mpnn1",
    )
    best = min(designs, key=lambda d: d.get("score", 1e9))
    seq = best["sequence"]
    if "/" in seq:
        seq = seq.split("/")[-1]
    print("INITIAL SEQUENCE:", seq, flush=True)

    pred = await esmfold.predict_structure(seq)
    print(f"initial self-score: plddt={pred.plddt} ptm={pred.ptm}", flush=True)

    print("=== independent verification ===", flush=True)
    result = await af2.predict_complex(sequences=[seq, target_seq], chain_names=["binder", "PDL1"])
    print(f"INITIAL RESULT: plddt={result.plddt} ptm={result.ptm} iptm={result.iptm}", flush=True)

    pae = result.pae_matrix
    if pae is None:
        print("NO PAE MATRIX -- stopping", flush=True)
        return
    pae = pae.tolist() if hasattr(pae, "tolist") else pae
    confident_count, overall_mean, contacts = pae_contacts(pae, len(seq))
    print(f"{confident_count}/{len(seq)} binder residues confident (PAE<10), mean={overall_mean:.2f}", flush=True)
    for b, tpos, pdbres, m in contacts:
        print(f"  binder {b} <-> target PDB residue {pdbres} PAE={m:.2f}", flush=True)

    with open("/tmp/phase_b_initial_result.json", "w") as f:
        json.dump({"sequence": seq, "plddt": result.plddt, "ptm": result.ptm,
                    "iptm": result.iptm, "contacts": contacts, "pae_matrix": pae}, f)

    if not contacts:
        print("=== NO CONFIDENT CONTACTS -- nothing to lock, stopping here ===", flush=True)
        return

    positions = sorted(set(c[0] for c in contacts))
    print(f"=== locking positions {positions}, redesigning rest ===", flush=True)
    designs2 = await mpnn.design_for_interface(
        complex_pdb=backbone_pdb, design_chain="B",
        interface_residues=HOTSPOTS, output_dir="/tmp/phase_b_mpnn2",
        fixed_positions=positions,
    )
    best2 = min(designs2, key=lambda d: d.get("score", 1e9))
    seq2 = best2["sequence"]
    if "/" in seq2:
        seq2 = seq2.split("/")[-1]
    print("REFINED SEQUENCE:", seq2, flush=True)

    pred2 = await esmfold.predict_structure(seq2)
    print(f"refined self-score: plddt={pred2.plddt} ptm={pred2.ptm}", flush=True)

    print("=== independently re-verifying refined sequence ===", flush=True)
    result2 = await af2.predict_complex(sequences=[seq2, target_seq], chain_names=["binder", "PDL1"])
    print(f"REFINED RESULT: plddt={result2.plddt} ptm={result2.ptm} iptm={result2.iptm}", flush=True)

    print("=== DONE ===", flush=True)


asyncio.run(main())
