import asyncio
import urllib.request


async def main():
    # Recreate the PD-L1 target locally in the container (chain A, residues 18-132
    # from 4ZQK) -- same construction as targets/pdl1_target.pdb, rebuilt here since
    # local files aren't mounted into this container.
    urllib.request.urlretrieve("https://files.rcsb.org/download/4ZQK.pdb", "/tmp/4zqk_full.pdb")
    with open("/tmp/4zqk_full.pdb") as f, open("/tmp/pdl1_target.pdb", "w") as out:
        out.write("HEADER    PD-L1 (chain A from PDB 4ZQK, PD-1 complex)\n")
        for line in f:
            if line.startswith("ATOM") and line[21] == "A":
                out.write(line)
        out.write("TER\nEND\n")
    print("target rebuilt: /tmp/pdl1_target.pdb", flush=True)

    from protein_design_mcp.pipelines.rfdiffusion import RFdiffusionRunner
    from protein_design_mcp.pipelines.proteinmpnn import ProteinMPNNRunner
    from protein_design_mcp.pipelines.esmfold import ESMFoldRunner

    rfdiffusion = RFdiffusionRunner()
    mpnn = ProteinMPNNRunner()
    esmfold = ESMFoldRunner()

    hotspots = ["A56", "A58", "A113", "A115", "A122", "A123", "A125"]

    print("=== generating backbone ===", flush=True)
    backbones = await rfdiffusion.generate_backbones(
        target_pdb="/tmp/pdl1_target.pdb",
        hotspot_residues=hotspots,
        output_dir="/tmp/phase_a_backbones",
        num_designs=1,
        binder_length=80,
    )
    print("backbones:", backbones, flush=True)
    backbone_pdb = backbones[0]["pdb_path"]

    print("=== BACKBONE PDB CONTENT START ===", flush=True)
    with open(backbone_pdb) as f:
        print(f.read(), flush=True)
    print("=== BACKBONE PDB CONTENT END ===", flush=True)

    print("=== running design_for_interface (target chain A fixed, design chain B) ===", flush=True)
    designs = await mpnn.design_for_interface(
        complex_pdb=backbone_pdb,
        design_chain="B",
        interface_residues=hotspots,
        output_dir="/tmp/phase_a_mpnn",
    )
    print(f"{len(designs)} designs returned", flush=True)
    for d in designs:
        print(d.get("id"), d.get("sequence"), "score=", d.get("score"), "recovery=", d.get("recovery"), flush=True)

    # Pick the best (lowest score = best per ProteinMPNN convention) design
    best = min(designs, key=lambda d: d.get("score", 1e9))
    seq = best["sequence"]
    if "/" in seq:
        # multi-chain output "TARGET/BINDER" -- keep the binder (design_chain) portion
        seq = seq.split("/")[-1]
    print("BEST BINDER SEQUENCE:", seq, flush=True)

    print("=== ESMFold self-score ===", flush=True)
    pred = await esmfold.predict_structure(seq)
    print(f"plddt={pred.plddt} ptm={pred.ptm}", flush=True)

    print("=== DONE ===", flush=True)


asyncio.run(main())
