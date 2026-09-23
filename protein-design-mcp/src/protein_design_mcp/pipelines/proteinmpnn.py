"""
ProteinMPNN wrapper for sequence design.

ProteinMPNN is a message passing neural network for designing
amino acid sequences for protein backbones.
"""

import asyncio
import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protein_design_mcp.exceptions import ProteinMPNNError

logger = logging.getLogger(__name__)

# ProteinMPNN weight URL (downloaded lazily on first use)
_PROTEINMPNN_WEIGHT_URL = (
    "https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt"
)


@dataclass
class ProteinMPNNConfig:
    """Configuration for ProteinMPNN runs."""

    proteinmpnn_path: Path = Path(os.environ.get("PROTEINMPNN_PATH", "/opt/ProteinMPNN"))
    python_path: str = os.environ.get("BIO_PYTHON_PATH", "python")
    num_sequences: int = 8
    sampling_temp: float = 0.1
    model_name: str = "v_48_020"


class ProteinMPNNRunner:
    """Wrapper for running ProteinMPNN."""

    def __init__(self, config: ProteinMPNNConfig | None = None):
        """Initialize ProteinMPNN runner."""
        self.config = config or ProteinMPNNConfig()

    def _ensure_weights(self) -> None:
        """Download ProteinMPNN weights on first use if not present."""
        weight_name = f"{self.config.model_name}.pt"
        weight_path = self.config.proteinmpnn_path / "vanilla_model_weights" / weight_name

        if weight_path.exists():
            return

        # Also check MODELS_DIR
        models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
        alt_path = models_dir / "ProteinMPNN" / "vanilla_model_weights" / weight_name

        if alt_path.exists():
            # Symlink into ProteinMPNN directory
            weight_path.parent.mkdir(parents=True, exist_ok=True)
            weight_path.symlink_to(alt_path)
            return

        logger.info(f"Downloading ProteinMPNN weights: {weight_name} (~150MB)...")
        weight_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(_PROTEINMPNN_WEIGHT_URL, str(weight_path))
            logger.info(f"Downloaded {weight_name} to {weight_path}")
        except Exception as e:
            raise ProteinMPNNError(f"Failed to download {weight_name}: {e}") from e

    def _parse_fasta_header(self, header: str) -> dict[str, Any]:
        """
        Parse a FASTA header line from ProteinMPNN output.

        Args:
            header: FASTA header string (e.g., ">seq_0, score=-1.5, recovery=0.80")

        Returns:
            Dictionary with parsed values
        """
        result = {"id": "", "score": None, "recovery": None}

        # Remove leading >
        header = header.lstrip(">").strip()

        # Try to parse structured format: "seq_0, score=-1.5, recovery=0.80"
        parts = header.split(",")
        result["id"] = parts[0].strip()

        for part in parts[1:]:
            part = part.strip()
            if "score=" in part:
                try:
                    result["score"] = float(part.split("=")[1])
                except (ValueError, IndexError):
                    pass
            elif "recovery=" in part:
                try:
                    result["recovery"] = float(part.split("=")[1])
                except (ValueError, IndexError):
                    pass

        return result

    def _build_command(
        self,
        backbone_pdb: str,
        output_dir: str,
        num_sequences: int,
        fixed_positions: list[int] | None = None,
    ) -> list[str]:
        """
        Build ProteinMPNN command line.

        Args:
            backbone_pdb: Path to backbone PDB structure
            output_dir: Output directory
            num_sequences: Number of sequences to design
            fixed_positions: Optional positions to keep fixed

        Returns:
            Command line as list of strings
        """
        script_path = self.config.proteinmpnn_path / "protein_mpnn_run.py"

        cmd = [
            self.config.python_path,
            str(script_path),
            "--pdb_path", backbone_pdb,
            "--out_folder", output_dir,
            "--num_seq_per_target", str(num_sequences),
            "--sampling_temp", str(self.config.sampling_temp),
            "--model_name", self.config.model_name,
            "--seed", "42",
        ]

        # Add fixed positions if specified
        if fixed_positions:
            # Create JSON file for fixed positions
            fixed_json = Path(output_dir) / "fixed_positions.json"
            fixed_dict = {
                "fixed_positions": {
                    "A": fixed_positions  # Assuming chain A
                }
            }
            fixed_json.parent.mkdir(parents=True, exist_ok=True)
            fixed_json.write_text(json.dumps(fixed_dict))
            cmd.extend(["--jsonl_path", str(fixed_json)])

        return cmd

    def _parse_outputs(self, output_dir: str) -> list[dict[str, Any]]:
        """
        Parse ProteinMPNN output files.

        Args:
            output_dir: Directory containing output files

        Returns:
            List of dictionaries with sequence info
        """
        output_path = Path(output_dir)
        results = []

        # ProteinMPNN outputs FASTA files in seqs/ subdirectory
        seqs_dir = output_path / "seqs"
        if not seqs_dir.exists():
            seqs_dir = output_path  # Fallback to main directory

        # Find all FASTA files
        fasta_files = list(seqs_dir.glob("*.fa")) + list(seqs_dir.glob("*.fasta"))

        for fasta_file in fasta_files:
            content = fasta_file.read_text()
            lines = content.strip().split("\n")

            current_header = None
            current_seq = []

            for line in lines:
                if line.startswith(">"):
                    # Save previous sequence if exists
                    if current_header and current_seq:
                        parsed = self._parse_fasta_header(current_header)
                        parsed["sequence"] = "".join(current_seq)
                        results.append(parsed)

                    current_header = line
                    current_seq = []
                else:
                    current_seq.append(line.strip())

            # Don't forget last sequence
            if current_header and current_seq:
                parsed = self._parse_fasta_header(current_header)
                parsed["sequence"] = "".join(current_seq)
                results.append(parsed)

        return results

    async def _run_proteinmpnn(self, cmd: list[str], output_dir: str) -> None:
        """
        Execute ProteinMPNN command.

        Args:
            cmd: Command line arguments
            output_dir: Output directory

        Raises:
            ProteinMPNNError: If execution fails
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # 10 min timeout (ProteinMPNN is typically fast)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=600
                )
            except asyncio.TimeoutError:
                process.kill()
                raise ProteinMPNNError(
                    "ProteinMPNN timed out after 10 minutes"
                )

            if process.returncode != 0:
                raise ProteinMPNNError(
                    f"ProteinMPNN failed with code {process.returncode}: "
                    f"{stderr.decode()}"
                )

        except FileNotFoundError as e:
            raise ProteinMPNNError(
                f"ProteinMPNN not found. Ensure PROTEINMPNN_PATH is set correctly. "
                f"Current path: {self.config.proteinmpnn_path}"
            ) from e

    async def design_sequences(
        self,
        backbone_pdb: str,
        output_dir: str,
        fixed_positions: list[int] | None = None,
        num_sequences: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Design sequences for a protein backbone.

        Args:
            backbone_pdb: Path to backbone PDB structure
            output_dir: Directory to save designed sequences
            fixed_positions: Residue positions to keep fixed
            num_sequences: Number of sequences to design

        Returns:
            List of dictionaries containing sequence info:
            - id: Sequence identifier
            - sequence: Designed amino acid sequence
            - score: ProteinMPNN score (lower is better)
            - recovery: Sequence recovery rate

        Raises:
            FileNotFoundError: If backbone PDB doesn't exist
            ProteinMPNNError: If design fails
        """
        # Validate inputs
        backbone_path = Path(backbone_pdb)
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone PDB not found: {backbone_pdb}")

        # Ensure weights are downloaded
        self._ensure_weights()

        # Use config defaults if not specified
        num_sequences = num_sequences or self.config.num_sequences

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build and run command
        cmd = self._build_command(
            backbone_pdb=str(backbone_path.absolute()),
            output_dir=str(output_path.absolute()),
            num_sequences=num_sequences,
            fixed_positions=fixed_positions,
        )

        await self._run_proteinmpnn(cmd, str(output_path))

        # Parse and return results
        return self._parse_outputs(str(output_path))

    async def _run_helper_script(self, cmd: list[str]) -> None:
        """Run one of ProteinMPNN's own helper_scripts/ preprocessing steps."""
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        if process.returncode != 0:
            raise ProteinMPNNError(
                f"ProteinMPNN helper script failed with code {process.returncode}: "
                f"{stderr.decode()}"
            )

    async def design_for_interface(
        self,
        complex_pdb: str,
        design_chain: str,
        interface_residues: list[str],
        output_dir: str,
        fixed_positions: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Design sequences for one chain of a complex, keeping the other
        chain(s) fixed -- e.g. redesign a binder chain while leaving the
        target chain exactly as given.

        Uses ProteinMPNN's own documented multi-chain workflow
        (helper_scripts/parse_multiple_chains.py -> assign_fixed_chains.py
        -> [make_fixed_positions_dict.py]), NOT the single-PDB `--pdb_path`
        mode -- that mode has no way to mark other chains as fixed, so a
        naive `--pdb_path` + `--chain_id_design` call (this method's
        previous implementation) fails outright: `--chain_id_design` is not
        a real flag on this ProteinMPNN CLI (confirmed against its own
        --help; the real flags are --pdb_path_chains for single-PDB mode,
        or --chain_id_jsonl/--fixed_positions_jsonl for the parsed-jsonl
        mode used here).

        Args:
            complex_pdb: Path to protein complex PDB (target + binder chains)
            design_chain: Chain ID to redesign; every other chain in
                complex_pdb is left fixed
            interface_residues: Interface residues to prioritize. NOTE:
                accepted but not currently used -- kept for API
                compatibility with callers. Use `fixed_positions` for
                actual position-level control.
            output_dir: Output directory
            fixed_positions: Positions within `design_chain` to also hold
                fixed (1-indexed into that chain's sequence), letting
                ProteinMPNN redesign only the remaining positions in that
                chain.

        Returns:
            List of designed sequences with scores
        """
        complex_path = Path(complex_pdb)
        if not complex_path.exists():
            raise FileNotFoundError(f"Complex PDB not found: {complex_pdb}")

        self._ensure_weights()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        helper_dir = self.config.proteinmpnn_path / "helper_scripts"

        # parse_multiple_chains.py wants a DIRECTORY of PDBs, not a file path
        pdb_dir = output_path / "pdb_in"
        pdb_dir.mkdir(exist_ok=True)
        linked_pdb = pdb_dir / complex_path.name
        if not linked_pdb.exists():
            linked_pdb.symlink_to(complex_path.absolute())

        parsed_jsonl = output_path / "parsed.jsonl"
        await self._run_helper_script([
            self.config.python_path, str(helper_dir / "parse_multiple_chains.py"),
            "--input_path", str(pdb_dir),
            "--output_path", str(parsed_jsonl),
        ])

        # Mark design_chain as the (only) chain to design; every other
        # chain present (the target) is implicitly fixed
        chains_jsonl = output_path / "chains_to_design.jsonl"
        await self._run_helper_script([
            self.config.python_path, str(helper_dir / "assign_fixed_chains.py"),
            "--input_path", str(parsed_jsonl),
            "--output_path", str(chains_jsonl),
            "--chain_list", design_chain,
        ])

        extra_args: list[str] = []
        if fixed_positions:
            fixed_jsonl = output_path / "fixed_positions.jsonl"
            await self._run_helper_script([
                self.config.python_path, str(helper_dir / "make_fixed_positions_dict.py"),
                "--input_path", str(parsed_jsonl),
                "--output_path", str(fixed_jsonl),
                "--chain_list", design_chain,
                "--position_list", " ".join(str(p) for p in fixed_positions),
            ])
            extra_args = ["--fixed_positions_jsonl", str(fixed_jsonl)]

        script_path = self.config.proteinmpnn_path / "protein_mpnn_run.py"
        cmd = [
            self.config.python_path,
            str(script_path),
            "--jsonl_path", str(parsed_jsonl),
            "--chain_id_jsonl", str(chains_jsonl),
            "--out_folder", str(output_path.absolute()),
            "--num_seq_per_target", str(self.config.num_sequences),
            "--sampling_temp", str(self.config.sampling_temp),
            "--model_name", self.config.model_name,
            "--seed", "42",
        ] + extra_args

        await self._run_proteinmpnn(cmd, str(output_path))

        return self._parse_outputs(str(output_path))
