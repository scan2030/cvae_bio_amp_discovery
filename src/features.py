"""Peptide feature helpers extracted from the notebook."""

from __future__ import annotations

import peptides


HYDROPHOBICITY_SCALE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}


AMINO_ACID_CLASSES = {
    "Tiny": ["A", "C", "G", "S", "T"],
    "Small": ["A", "B", "C", "D", "G", "N", "P", "S", "T", "V"],
    "Aliphatic": ["A", "I", "L", "V"],
    "Aromatic": ["F", "H", "W", "Y"],
    "Non-polar": ["A", "C", "F", "G", "I", "L", "M", "P", "V", "W", "Y"],
    "Polar": ["D", "E", "H", "K", "N", "Q", "R", "S", "T", "Z"],
    "Charged": ["B", "D", "E", "H", "K", "R", "Z"],
    "Basic": ["H", "K", "R"],
    "Acidic": ["B", "D", "E", "Z"],
}


def calculate_hydrophobicity(sequence: str) -> float:
    total_hydrophobicity = sum(HYDROPHOBICITY_SCALE.get(residue, 0) for residue in sequence)
    return total_hydrophobicity / len(sequence) if sequence else 0.0


def calculate_hydrophobic_moment(sequence: str, angle: int = 100, window: int = 11) -> float:
    if not sequence:
        return 0.0
    return float(peptides.Peptide(sequence).hydrophobic_moment(angle=angle, window=window))


def calculate_charge(sequence: str, pk_scale: str = "Lehninger") -> float:
    if not sequence:
        return 0.0
    return float(peptides.Peptide(sequence).charge(pKscale=pk_scale))


def calculate_stability(sequence: str) -> float:
    if len(sequence) <= 1:
        return 0.0
    return round(float(peptides.Peptide(sequence).instability_index()), 2)


def calculate_boman(sequence: str) -> float:
    if not sequence:
        return 0.0
    return round(float(peptides.Peptide(sequence).boman()), 2)


def calculate_aliphatic_proportion(sequence: str) -> float:
    if not sequence:
        return 0.0
    aliphatic_amino_acids = {"A", "I", "L", "V"}
    return sum(1 for aa in sequence if aa in aliphatic_amino_acids) / len(sequence)


def calculate_amino_acid_class_properties(sequence: str) -> dict[str, float]:
    peptide = peptides.Peptide(sequence)
    counts = peptide.counts()
    total_length = sum(counts.values())
    properties: dict[str, float] = {"Sequence": sequence}

    for class_name, amino_acids in AMINO_ACID_CLASSES.items():
        count = sum(counts.get(aa, 0) for aa in amino_acids)
        percentage = (count / total_length * 100) if total_length > 0 else 0.0
        properties[f"{class_name}_Count"] = count
        properties[f"{class_name}_Percentage"] = percentage

    return properties


def calculate_physicochemical_properties(sequence: str) -> dict[str, float]:
    peptide = peptides.Peptide(sequence)
    return {
        "Molecular_Weight": peptide.molecular_weight(),
        "Aliphatic_Index": peptide.aliphatic_index(),
        "Boman": peptide.boman(),
        "Instability": peptide.instability_index(),
        "Auto_correlation": peptide.auto_correlation(
            table=peptides.tables.HYDROPHOBICITY["KyteDoolittle"]
        ),
        "Auto_covariance": peptide.auto_covariance(
            table=peptides.tables.HYDROPHOBICITY["KyteDoolittle"]
        ),
        "Hydrophobic_moment_ang_100": peptide.hydrophobic_moment(angle=100),
        "Hydrophobic_moment_ang_160": peptide.hydrophobic_moment(angle=160),
        "Isoelectric_point": peptide.isoelectric_point(pKscale="EMBOSS"),
    }
