from rdkit.Chem import rdMolAlign, Mol
import numpy as np
from tabulate import tabulate


def calculate_rmsd_matrix(mol):
    num_conformers = mol.GetNumConformers()

    # Initialize an RMSD matrix with zeros
    rmsd_matrix = np.zeros((num_conformers, num_conformers))

    # Calculate RMSD between each pair of conformers
    for i in range(num_conformers):
        for j in range(i + 1, num_conformers):  # Only need to compute upper triangle
            rmsd = rdMolAlign.GetBestRMS(mol, mol, prbId=i, refId=j)
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = None  # Symmetric matrix

    return rmsd_matrix


def print_rmsd_matrix(matrix: np.ndarray) -> None:
    matrix_list = matrix.tolist()
    headers = [f"Conf {i}" for i in range(len(matrix_list))]
    print(
        tabulate(
            matrix_list, headers, tablefmt="grid", floatfmt=".3f", showindex=headers
        )
    )


def print_mol_conf_energies(mol: Mol):
    energies = []
    headers = ["Conformer", "Energy"]
    for conf in mol.GetConformers():
        energy = conf.GetDoubleProp("ENERGY")
        energies.append([f"Conf {conf.GetId()}", energy])

    print(tabulate(energies, headers=headers, tablefmt="grid"))
