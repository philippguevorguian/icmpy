from icmpy.ConfGenerator import ConfGenerator, ConformerResult
from icmpy.confgen_utils import (
    calculate_rmsd_matrix,
    print_rmsd_matrix,
    print_mol_conf_energies,
)
import argparse
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=False, default="config.yaml")
parser.add_argument("--icm_path", type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    config_file = args.config_file
    icm_path = args.icm_path

    mol_identifiers = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CCCCC",
    ]  # This can also include paths to sdf files
    generator = ConfGenerator(config_file=config_file, icm_path=icm_path)
    results: List[ConformerResult] = generator.process_mol_list(
        mol_identifiers, n_jobs=1, show_progress=True
    )
    print(results)
    rmsd_matrix = calculate_rmsd_matrix(results[0].mol)
    print_rmsd_matrix(rmsd_matrix)
    print_mol_conf_energies(results[0].mol)
