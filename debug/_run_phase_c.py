import asyncio
import json
import urllib.request

HOTSPOTS = ["A56", "A58", "A113", "A115", "A122", "A123", "A125"]


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
    urllib.request.urlretrieve("https://files.rcsb.org/download/4ZQK.pdb", "/tmp/4zqk_full.pdb")
    with open("/tmp/4zqk_full.pdb") as f, open("/tmp/pdl1_target.pdb", "w") as out:
        out.write("HEADER    PD-L1\n")
        for line in f:
            if line.startswith("ATOM") and line[21] == "A":
                out.write(line)
        out.write("TER\nEND\n")

    aa3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
              "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
              "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
    target_seq, seen = [], set()
    with open("/tmp/pdl1_target.pdb") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resseq = int(line[22:26])
                if resseq not in seen:
                    seen.add(resseq)
                    target_seq.append(aa3to1.get(line[17:20].strip(), "X"))
    target_seq = "".join(target_seq)

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
        target_pdb="/tmp/pdl1_target.pdb", hotspot_residues=HOTSPOTS,
        output_dir="/tmp/phase_c_backbone", num_designs=1, binder_length=80,
    )
    backbone_pdb = backbones[0]["pdb_path"]

    print("=== initial interface-aware design ===", flush=True)
    designs = await mpnn.design_for_interface(
        complex_pdb=backbone_pdb, design_chain="B",
        interface_residues=HOTSPOTS, output_dir="/tmp/phase_c_mpnn1",
    )
    best = min(designs, key=lambda d: d.get("score", 1e9))
    seq = best["sequence"]
    if "/" in seq:
        seq = seq.split("/")[-1]
    print("INITIAL SEQUENCE:", seq, flush=True)

    pred = await esmfold.predict_structure(seq)
    print(f"initial self-score: plddt={pred.plddt} ptm={pred.ptm}", flush=True)

    print("=== independent verification (also gives us a real threaded complex structure) ===", flush=True)
    result = await af2.predict_complex(sequences=[seq, target_seq], chain_names=["binder", "PDL1"])
    print(f"INITIAL RESULT: plddt={result.plddt} ptm={result.ptm} iptm={result.iptm}", flush=True)

    complex_pdb_path = "/tmp/phase_c_complex.pdb"
    with open(complex_pdb_path, "w") as f:
        f.write(result.pdb_string)
    resnums_by_chain = {}
    for line in result.pdb_string.splitlines():
        if line.startswith("ATOM"):
            c = line[21]
            resnums_by_chain.setdefault(c, set()).add(int(line[22:26]))
    binder_chain = [c for c, r in resnums_by_chain.items() if len(r) == len(seq)][0]
    print(f"binder_chain: {binder_chain}, chains: {[(c, len(r)) for c, r in resnums_by_chain.items()]}", flush=True)

    pae = result.pae_matrix
    pae = pae.tolist() if hasattr(pae, "tolist") else pae
    confident_count, overall_mean, contacts = pae_contacts(pae, len(seq))
    print(f"{confident_count}/{len(seq)} confident, mean={overall_mean:.2f}", flush=True)

    with open("/tmp/phase_c_initial.json", "w") as f:
        json.dump({"sequence": seq, "plddt": result.plddt, "ptm": result.ptm, "iptm": result.iptm,
                    "confident_count": confident_count, "contacts": contacts,
                    "pae_matrix": pae, "pdb_string": result.pdb_string}, f)
    print("saved initial result to /tmp/phase_c_initial.json", flush=True)

    frac = confident_count / len(seq)
    print(f"confident fraction: {frac:.2f}", flush=True)
    if frac == 0:
        print("=== NO confident signal at all -- nothing to guide, stopping ===", flush=True)
        return
    if frac > 0.9:
        print("=== ALREADY near-complete (>90% confident) -- little room to refine, stopping ===", flush=True)
        return

    positions = sorted(set(c[0] for c in contacts))
    print(f"=== PARTIAL signal ({frac:.0%}) -- this is the interesting case. Locking positions {positions}, redesigning rest ===", flush=True)

    designs2 = await mpnn.design_for_interface(
        complex_pdb=complex_pdb_path, design_chain=binder_chain,
        interface_residues=[], output_dir="/tmp/phase_c_mpnn2",
        fixed_positions=positions,
    )
    best2 = min(designs2, key=lambda d: d.get("score", 1e9))
    seq2 = best2["sequence"]
    if "/" in seq2:
        seq2 = seq2.split("/")[-1]
    print("REFINED SEQUENCE:", seq2, flush=True)
    print("positions changed:", [i + 1 for i, (a, b) in enumerate(zip(seq, seq2)) if a != b], flush=True)

    pred2 = await esmfold.predict_structure(seq2)
    print(f"refined self-score: plddt={pred2.plddt} ptm={pred2.ptm}", flush=True)

    print("=== independently re-verifying refined sequence ===", flush=True)
    result2 = await af2.predict_complex(sequences=[seq2, target_seq], chain_names=["binder", "PDL1"])
    print(f"REFINED RESULT: plddt={result2.plddt} ptm={result2.ptm} iptm={result2.iptm}", flush=True)

    with open("/tmp/phase_c_refined.json", "w") as f:
        json.dump({"sequence": seq2, "plddt": result2.plddt, "ptm": result2.ptm,
                    "iptm": result2.iptm, "pdb_string": result2.pdb_string}, f)

    print(f"=== SUMMARY: initial ipTM={result.iptm} -> refined ipTM={result2.iptm} ===", flush=True)
    print("=== DONE ===", flush=True)


asyncio.run(main())
