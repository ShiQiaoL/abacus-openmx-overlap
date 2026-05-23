#!/usr/bin/env python3
"""Compute DeepH overlap.h5 directly from POSCAR and numerical atomic orbitals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


LOGGER = logging.getLogger("direct-overlap")
SCHEMA_VERSION = "direct-overlap-basis/2"
ANGULAR_MOMENTA = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}
HARTREE_TO_EV = 27.211386245988
MATRIX_DATASETS = ("atom_pairs", "chunk_boundaries", "chunk_shapes", "entries")
PREPARED_BAND_FILES = (
    "POSCAR",
    "K_PATH",
    "hamiltonian.h5",
    "overlap.h5",
    "info.json",
    "band_prepare_manifest.json",
)
STALE_BAND_OUTPUTS = (
    "band.h5",
    "band.png",
    "fermi_energy.json",
    "eigval.h5",
    "dos.h5",
    "dos.png",
)


def normalize_path(path: Path) -> Path:
    """Return an absolute path without requiring the target to exist."""
    return path.expanduser().resolve(strict=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def software_versions() -> dict[str, str | None]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "h5py": package_version("h5py"),
        "hpro": package_version("hpro"),
        "deepx-dock": package_version("deepx-dock"),
    }


def parse_energy_unit(unit_text: str | None) -> str:
    if unit_text is None:
        return "hartree"
    normalized = unit_text.strip().lower()
    if normalized in {"ev", "electronvolt", "electronvolts"}:
        return "ev"
    if normalized in {"hartree", "ha", "hartrees", "a.u.", "au"}:
        return "hartree"
    raise ValueError(f"Unsupported energy unit: {unit_text}")


def convert_energy_to_ev(value: float, unit_text: str | None) -> float:
    unit = parse_energy_unit(unit_text)
    if unit == "ev":
        return value
    if unit == "hartree":
        return value * HARTREE_TO_EV
    raise AssertionError(f"Unhandled energy unit: {unit}")


def parse_fermi_log(path: Path) -> dict[str, Any]:
    """Parse a Fermi level / chemical potential from a text output file."""
    patterns = [
        re.compile(
            r"Chemical\s+potential\s*\((?P<unit>Hartree|Ha|eV)\)\s*[:=]?\s*"
            r"(?P<value>[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:Fermi\s+(?:level|energy)|E[_-]?F)\s*\((?P<unit>Hartree|Ha|eV)\)\s*[:=]?\s*"
            r"(?P<value>[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<label>ChemP)\s*(?:\((?P<unit>Hartree|Ha|eV)\))?\s*[:=]?\s*"
            r"(?P<value>[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)",
            re.IGNORECASE,
        ),
    ]

    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        for pattern in patterns:
            match = pattern.search(line)
            if match is None:
                continue
            unit = match.groupdict().get("unit")
            value = float(match.group("value"))
            matches.append(
                {
                    "fermi_energy_eV": convert_energy_to_ev(value, unit),
                    "raw_value": value,
                    "raw_unit": parse_energy_unit(unit),
                    "line_number": line_number,
                    "line": line.strip(),
                    "path": str(path),
                }
            )
            break

    if not matches:
        raise ValueError(
            f"Could not find a Fermi level / chemical potential in {path}. "
            "For OpenMX, use a converged .out/.log containing 'Chemical potential (Hartree)', "
            "or pass --fermi-energy-ev explicitly."
        )
    return matches[-1]


def resolve_fermi_energy(args: argparse.Namespace) -> dict[str, Any]:
    if args.fermi_energy_ev is not None and args.fermi_log is not None:
        raise ValueError("Use only one of --fermi-energy-ev or --fermi-log, not both.")

    if args.fermi_energy_ev is not None:
        return {
            "provided": True,
            "source": "cli",
            "fermi_energy_eV": float(args.fermi_energy_ev),
            "note": "User-provided SCF Fermi energy.",
        }

    if args.fermi_log is not None:
        log_path = normalize_path(args.fermi_log)
        if not log_path.is_file():
            raise FileNotFoundError(f"Missing Fermi log file: {log_path}")
        parsed = parse_fermi_log(log_path)
        parsed["provided"] = True
        parsed["source"] = "log"
        parsed["note"] = "Parsed from a user-provided SCF output/log file."
        return parsed

    if args.require_fermi_energy:
        raise ValueError(
            "No SCF Fermi energy was provided. Direct overlap generation cannot determine the Fermi level. "
            "Pass --fermi-energy-ev VALUE or --fermi-log PATH, or remove --require-fermi-energy."
        )

    return {
        "provided": False,
        "source": "default_zero",
        "fermi_energy_eV": 0.0,
        "note": (
            "No SCF Fermi energy was provided. info.json will contain 0.0 eV as a placeholder; "
            "do not treat it as a converged Fermi level."
        ),
    }


def read_poscar_summary(poscar_path: Path) -> dict[str, Any]:
    lines = poscar_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"{poscar_path} is too short to be a VASP 5 POSCAR")

    species = lines[5].split()
    count_tokens = lines[6].split()
    if not species or not count_tokens:
        raise ValueError(f"{poscar_path} does not contain a VASP 5 species/count block")
    if not all(re.fullmatch(r"[A-Z][a-z]?", item) for item in species):
        raise ValueError(
            f"{poscar_path} must be a VASP 5 POSCAR with an explicit element line"
        )
    try:
        counts = [int(item) for item in count_tokens]
    except ValueError as exc:
        raise ValueError(f"{poscar_path} has invalid species counts: {count_tokens}") from exc
    if len(species) != len(counts):
        raise ValueError(
            f"{poscar_path} has {len(species)} species labels but {len(counts)} counts"
        )

    composition = OrderedDict((element, count) for element, count in zip(species, counts, strict=True))
    formula = "".join(f"{element}{count}" for element, count in composition.items())
    return {
        "path": str(poscar_path),
        "sha256": sha256_file(poscar_path),
        "species_order": species,
        "counts": counts,
        "composition": dict(composition),
        "formula": formula,
        "natoms": int(sum(counts)),
    }


def parse_formula(formula: str) -> dict[str, int]:
    matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    reconstructed = "".join(element + count for element, count in matches)
    if not matches or reconstructed != formula:
        raise ValueError(f"Invalid formula string: {formula}")
    parsed: dict[str, int] = {}
    for element, count_text in matches:
        parsed[element] = parsed.get(element, 0) + int(count_text or "1")
    return parsed


def parse_element_path_map(items: list[str], option_name: str) -> dict[str, Path]:
    basis_map: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {option_name} item {item!r}; expected ELEMENT=/path/to/file")
        element, path_text = item.split("=", 1)
        if not re.fullmatch(r"[A-Z][a-z]?", element):
            raise ValueError(f"Invalid element in {option_name}: {element!r}")
        basis_map[element] = normalize_path(Path(path_text))
    return basis_map


def parse_openmx_basis_items(items: list[str]) -> dict[str, str]:
    basis_specs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --openmx-basis item {item!r}; expected ELEMENT=BASIS_SPEC, "
                "for example Au=Au7.0-s2p2d2f1"
            )
        element, basis_spec = item.split("=", 1)
        if not re.fullmatch(r"[A-Z][a-z]?", element):
            raise ValueError(f"Invalid element in --openmx-basis: {element!r}")
        parse_openmx_basis_spec(basis_spec)
        basis_specs[element] = basis_spec
    return basis_specs


def parse_openmx_basis_spec(basis_spec: str) -> tuple[str, dict[int, int]]:
    if "-" not in basis_spec:
        raise ValueError(
            f"Invalid OpenMX basis spec {basis_spec!r}; expected a label like Au7.0-s2p2d2f1"
        )
    file_stem, orbital_spec = basis_spec.split("-", 1)
    if not file_stem:
        raise ValueError(f"Invalid OpenMX basis spec {basis_spec!r}: missing PAO file stem")

    l_counts: dict[int, int] = {}
    position = 0
    pattern = re.compile(r"([spdfgh])(\d+)(?:>(\d+))?")
    while position < len(orbital_spec):
        match = pattern.match(orbital_spec, position)
        if match is None:
            raise ValueError(
                f"Unsupported OpenMX orbital spec {orbital_spec!r} in {basis_spec!r}. "
                "Use compact labels such as s2p2d1 or simple contracted labels such as s2>1p2>1."
            )
        label, primitive_count, contracted_count = match.groups()
        count = int(contracted_count or primitive_count)
        if count <= 0:
            raise ValueError(f"OpenMX basis count must be positive in {basis_spec!r}")
        angular_momentum = ANGULAR_MOMENTA[label]
        l_counts[angular_momentum] = l_counts.get(angular_momentum, 0) + count
        position = match.end()
    return file_stem, l_counts


def openmx_cutoff_from_stem(file_stem: str) -> float | None:
    match = re.match(r"[A-Z][a-z]?(\d+(?:\.\d+)?)$", file_stem)
    return float(match.group(1)) if match else None


def load_runtime() -> SimpleNamespace:
    try:
        from HPRO.io.aodata import AOData
        from HPRO.io.deephio import save_mat_deeph
        from HPRO.io.struio import from_poscar
        from HPRO.utils.misc import atom_number2name
        from HPRO.utils.orbutils import GridFunc, LinearRGD, RadialGrid
        from deepx_dock.compute.overlap.overlap import calc_overlap
    except Exception as exc:
        raise RuntimeError(
            "Failed to import HPRO/deepx-dock runtime. Use the command wrapper or source "
            "the ABACUS Intel MPI environment before running this script, then install "
            "hpro and deepx-dock in the selected Python environment."
        ) from exc

    return SimpleNamespace(
        AOData=AOData,
        GridFunc=GridFunc,
        LinearRGD=LinearRGD,
        RadialGrid=RadialGrid,
        atom_number2name=atom_number2name,
        calc_overlap=calc_overlap,
        from_poscar=from_poscar,
        save_mat_deeph=save_mat_deeph,
    )


def parse_abacus_orb(path: Path, runtime: SimpleNamespace) -> list[Any]:
    """Parse ABACUS .orb radial numerical orbitals into HPRO GridFunc objects."""
    lines = path.read_text(encoding="utf-8").splitlines()
    mesh: int | None = None
    dr: float | None = None
    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "Mesh":
            mesh = int(fields[1])
        elif len(fields) >= 2 and fields[0] == "dr":
            dr = float(fields[1])
    if mesh is None or dr is None:
        raise ValueError(f"Cannot find Mesh/dr in {path}")

    orbitals: list[Any] = []
    i = 0
    while i < len(lines):
        fields = lines[i].split()
        if len(fields) == 3 and all(re.fullmatch(r"-?\d+", item) for item in fields):
            angular_momentum = int(fields[1])
            values: list[float] = []
            i += 1
            while i < len(lines) and len(values) < mesh:
                values.extend(float(item) for item in lines[i].split())
                i += 1
            if len(values) != mesh:
                raise ValueError(f"Incomplete orbital block in {path}: got {len(values)} values")
            radial_grid = runtime.LinearRGD(0.0, dr * (mesh - 1), mesh)
            phi = np.asarray(values, dtype=np.float64)
            orbitals.append(runtime.GridFunc(radial_grid, phi, l=angular_momentum, rcut=radial_grid.rend))
            continue
        i += 1
    if not orbitals:
        raise ValueError(f"No orbital blocks found in {path}")
    return orbitals


def parse_openmx_pao(path: Path, l_counts: dict[int, int], cutoff: float | None, runtime: SimpleNamespace) -> list[Any]:
    """Parse selected radial orbitals from an OpenMX .pao file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    orbitals: list[Any] = []

    for angular_momentum in sorted(l_counts):
        count = l_counts[angular_momentum]
        start_match = re.search(rf"<pseudo\.atomic\.orbitals\.L={angular_momentum}\s*\n", text)
        end_match = re.search(rf"pseudo\.atomic\.orbitals\.L={angular_momentum}>", text)
        if start_match is None or end_match is None or end_match.start() <= start_match.end():
            raise ValueError(f"Cannot find OpenMX PAO block L={angular_momentum} in {path}")

        block = text[start_match.end(): end_match.start()]
        rows: list[list[float]] = []
        for line in block.splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                rows.append([float(item) for item in fields])
            except ValueError:
                continue
        if not rows:
            raise ValueError(f"OpenMX PAO block L={angular_momentum} in {path} contains no numeric rows")

        table = np.asarray(rows, dtype=np.float64)
        available = table.shape[1] - 2
        if count > available:
            raise ValueError(
                f"{path} block L={angular_momentum} has {available} radial functions, "
                f"but the requested OpenMX basis needs {count}"
            )

        radial_grid_values = table[:, 1]
        if np.any(np.diff(radial_grid_values) <= 0):
            raise ValueError(f"OpenMX radial grid is not strictly increasing in {path}, L={angular_momentum}")

        effective_cutoff = cutoff if cutoff is not None else float(radial_grid_values[-1])
        for radial_index in range(count):
            phi = table[:, 2 + radial_index].copy()
            phi[radial_grid_values > effective_cutoff] = 0.0
            if radial_grid_values[0] > 0.0:
                origin_value = phi[0] if angular_momentum == 0 else 0.0
                r_values = np.concatenate(([0.0], radial_grid_values))
                phi_values = np.concatenate(([origin_value], phi))
            else:
                r_values = radial_grid_values
                phi_values = phi

            radial_grid = runtime.RadialGrid(r_values)
            orbitals.append(
                runtime.GridFunc(radial_grid, phi_values, l=angular_momentum, rcut=effective_cutoff)
            )

    if not orbitals:
        raise ValueError(f"No OpenMX orbital selected from {path}")
    return orbitals


def find_abacus_basis_file(element: str, basis_dir: Path, basis_map: dict[str, Path]) -> Path:
    if element in basis_map:
        path = basis_map[element]
        if not path.is_file():
            raise FileNotFoundError(f"--basis-map for {element} does not exist: {path}")
        return path

    matches = sorted(basis_dir.glob(f"{element}_*.orb"))
    if len(matches) != 1:
        match_text = ", ".join(str(path) for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected exactly one {element}_*.orb in {basis_dir}; found {match_text}. "
            "Use --basis-map ELEMENT=/path/to/file.orb if the directory contains alternatives."
        )
    return matches[0]


def find_openmx_basis_file(
    element: str,
    basis_dir: Path,
    basis_map: dict[str, Path],
    openmx_basis_specs: dict[str, str],
) -> tuple[Path, str, dict[int, int], float | None]:
    if element not in openmx_basis_specs:
        raise ValueError(
            f"Missing OpenMX basis spec for {element}. Add --openmx-basis {element}={element}7.0-s2p2d1 "
            "with the PAO stem and orbital counts you intend to use."
        )
    file_stem, l_counts = parse_openmx_basis_spec(openmx_basis_specs[element])
    if element in basis_map:
        path = basis_map[element]
        if not path.is_file():
            raise FileNotFoundError(f"--basis-map for {element} does not exist: {path}")
    else:
        path = basis_dir / f"{file_stem}.pao"
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected OpenMX PAO file for {element}: {path}. "
                "Use --basis-map ELEMENT=/path/to/file.pao if it is stored elsewhere."
            )
    return path, openmx_basis_specs[element], l_counts, openmx_cutoff_from_stem(file_stem)


def build_aodata(
    structure: Any,
    basis_dir: Path,
    basis_code: str,
    basis_map: dict[str, Path],
    openmx_basis_specs: dict[str, str],
    runtime: SimpleNamespace,
) -> SimpleNamespace:
    """Build the minimal AOData-like object required by calc_overlap/save_mat_deeph."""
    species_numbers = structure.atomic_species
    species_names = runtime.atom_number2name(species_numbers)
    phirgrids_spc: dict[int, list[Any]] = {}
    ls_spc: dict[int, list[int]] = {}
    nradial_spc: dict[int, int] = {}
    cutoffs_orb: dict[str, list[float]] = {}
    cutoffs: dict[str, float] = {}
    basis_files: dict[str, Path] = {}
    basis_specs: dict[str, str] = {}

    for number, name in zip(species_numbers, species_names, strict=True):
        if basis_code == "abacus":
            basis_file = find_abacus_basis_file(name, basis_dir, basis_map)
            grids = parse_abacus_orb(basis_file, runtime)
            basis_specs[name] = basis_file.name
        elif basis_code == "openmx":
            basis_file, basis_spec, l_counts, cutoff = find_openmx_basis_file(
                name, basis_dir, basis_map, openmx_basis_specs
            )
            grids = parse_openmx_pao(basis_file, l_counts, cutoff, runtime)
            basis_specs[name] = basis_spec
        else:
            raise ValueError(f"Unsupported basis code: {basis_code}")

        grids = [grid for _, grid in sorted(enumerate(grids), key=lambda item: (item[1].l, item[0]))]
        phirgrids_spc[int(number)] = grids
        ls_spc[int(number)] = [int(grid.l) for grid in grids]
        nradial_spc[int(number)] = len(grids)
        cutoffs_orb[name] = [float(grid.rcut) for grid in grids]
        cutoffs[name] = float(max(cutoffs_orb[name]))
        basis_files[name] = basis_file

    orbslices_spc: dict[int, list[int]] = {}
    norbfull_spc: dict[int, int] = {}
    for number, orbital_types in ls_spc.items():
        slices = [0]
        for angular_momentum in orbital_types:
            slices.append(slices[-1] + 2 * angular_momentum + 1)
        orbslices_spc[number] = slices
        norbfull_spc[number] = slices[-1]

    return SimpleNamespace(
        structure=structure,
        aocode="abacus-orb-direct",
        spinful=False,
        magnetic=False,
        ls_spc=ls_spc,
        phirgrids_spc=phirgrids_spc,
        nradial_spc=nradial_spc,
        orbslices_spc=orbslices_spc,
        norbfull_spc=norbfull_spc,
        cutoffs=cutoffs,
        cutoffs_orb=cutoffs_orb,
        basis_files=basis_files,
        basis_specs=basis_specs,
        phiQlist_spc=None,
        phiQEcut=None,
        calc_phiQ=None,
    )


def attach_calc_phiq(aodata: SimpleNamespace, runtime: SimpleNamespace) -> None:
    """Reuse HPRO AOData.calc_phiQ implementation on the lightweight object."""
    aodata.calc_phiQ = runtime.AOData.calc_phiQ.__get__(aodata, SimpleNamespace)


def basis_report(aodata: SimpleNamespace, runtime: SimpleNamespace) -> dict[str, Any]:
    species_names = runtime.atom_number2name(aodata.structure.atomic_species)
    report: dict[str, Any] = {}
    for number, name in zip(aodata.structure.atomic_species, species_names, strict=True):
        basis_file = aodata.basis_files[name]
        norms = [
            float(grid.rgd.integrate(grid.func * grid.func))
            for grid in aodata.phirgrids_spc[int(number)]
        ]
        report[name] = {
            "basis_file": str(basis_file),
            "basis_spec": aodata.basis_specs[name],
            "basis_sha256": sha256_file(basis_file),
            "angular_momenta": aodata.ls_spc[int(number)],
            "nradial": aodata.nradial_spc[int(number)],
            "norbfull": aodata.norbfull_spc[int(number)],
            "cutoffs_bohr": aodata.cutoffs_orb[name],
            "norms": norms,
            "max_norm_deviation": max(abs(norm - 1.0) for norm in norms),
        }
    return report


def validate_basis_norms(report: dict[str, Any], tolerance: float, allow_unnormalized: bool) -> None:
    bad_items = []
    for element, item in report.items():
        deviation = float(item["max_norm_deviation"])
        if deviation > tolerance:
            bad_items.append((element, deviation))
    if not bad_items:
        return

    message = ", ".join(f"{element}: {deviation:.3e}" for element, deviation in bad_items)
    if allow_unnormalized:
        LOGGER.warning("Orbital normalization deviation exceeds tolerance %g: %s", tolerance, message)
        return
    raise ValueError(
        f"Orbital normalization deviation exceeds tolerance {tolerance:g}: {message}. "
        "Use --allow-unnormalized-orbitals only if this is intentional."
    )


def hdf5_summary(path: Path) -> dict[str, Any]:
    import h5py

    datasets: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets[name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "size": int(obj.size),
                }

        handle.visititems(visitor)

    total_size = sum(item["size"] for item in datasets.values())
    if not datasets or total_size <= 0:
        raise ValueError(f"{path} is readable but does not contain non-empty datasets")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "datasets": datasets,
        "total_dataset_size": int(total_size),
    }


@dataclass(frozen=True)
class MatrixSignature:
    path: Path
    atom_pairs: np.ndarray
    chunk_shapes: np.ndarray
    entries_dtype: str
    entries_len: int
    size_bytes: int

    @property
    def unique_shapes(self) -> list[list[int]]:
        return np.unique(self.chunk_shapes, axis=0).astype(int).tolist()

    @property
    def max_shape(self) -> list[int]:
        return self.chunk_shapes.max(axis=0).astype(int).tolist()


def read_matrix_signature(path: Path) -> MatrixSignature:
    import h5py

    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        missing = [dataset for dataset in MATRIX_DATASETS if dataset not in handle]
        if missing:
            raise ValueError(f"{path} is missing HDF5 datasets: {missing}")
        atom_pairs = np.asarray(handle["atom_pairs"][:], dtype=np.int64)
        chunk_shapes = np.asarray(handle["chunk_shapes"][:], dtype=np.int64)
        entries = handle["entries"]
        entries_dtype = str(entries.dtype)
        entries_len = int(entries.shape[0])
    if atom_pairs.ndim != 2 or atom_pairs.shape[1] != 5:
        raise ValueError(f"{path}: atom_pairs must have shape (npairs, 5), got {atom_pairs.shape}")
    if chunk_shapes.ndim != 2 or chunk_shapes.shape[1] != 2:
        raise ValueError(f"{path}: chunk_shapes must have shape (npairs, 2), got {chunk_shapes.shape}")
    if len(atom_pairs) != len(chunk_shapes):
        raise ValueError(f"{path}: atom_pairs/chunk_shapes length mismatch")
    return MatrixSignature(
        path=path,
        atom_pairs=atom_pairs,
        chunk_shapes=chunk_shapes,
        entries_dtype=entries_dtype,
        entries_len=entries_len,
        size_bytes=path.stat().st_size,
    )


def read_poscar_elements(poscar_path: Path) -> tuple[list[str], list[int], list[str]]:
    lines = poscar_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"{poscar_path} is too short to be a VASP 5 POSCAR")
    species = lines[5].split()
    counts = [int(item) for item in lines[6].split()]
    if len(species) != len(counts):
        raise ValueError(f"{poscar_path}: element/count length mismatch")
    elements_by_atom: list[str] = []
    for element, count in zip(species, counts, strict=True):
        elements_by_atom.extend([element] * count)
    return species, counts, elements_by_atom


def orbital_count_from_l_list(angular_momenta: list[int]) -> int:
    return int(sum(2 * int(angular_momentum) + 1 for angular_momentum in angular_momenta))


def orbital_counts_from_info(info: dict[str, Any], poscar_path: Path) -> tuple[list[int], dict[str, int], int]:
    species, counts, elements_by_atom = read_poscar_elements(poscar_path)
    orbital_map = info.get("elements_orbital_map")
    if not isinstance(orbital_map, dict) or not orbital_map:
        raise ValueError("info.json does not contain a valid elements_orbital_map")
    per_element = {
        element: orbital_count_from_l_list(orbital_map[element])
        for element in species
    }
    per_atom = [per_element[element] for element in elements_by_atom]
    total = int(sum(count * per_element[element] for element, count in zip(species, counts, strict=True)))
    return per_atom, per_element, total


def matrix_shape_match_report(
    signature: MatrixSignature,
    per_atom_orbitals: list[int],
    spin_factor: int,
) -> tuple[bool, str]:
    if spin_factor < 1:
        raise ValueError("spin_factor must be >= 1")
    if len(signature.atom_pairs) != len(signature.chunk_shapes):
        return False, "atom_pairs and chunk_shapes length mismatch"
    for pair_index, (atom_pair, shape) in enumerate(zip(signature.atom_pairs, signature.chunk_shapes, strict=True)):
        i_atom = int(atom_pair[3])
        j_atom = int(atom_pair[4])
        if i_atom < 0 or i_atom >= len(per_atom_orbitals) or j_atom < 0 or j_atom >= len(per_atom_orbitals):
            return False, f"pair {pair_index}: atom index outside POSCAR atom count"
        expected = (per_atom_orbitals[i_atom] * spin_factor, per_atom_orbitals[j_atom] * spin_factor)
        observed = (int(shape[0]), int(shape[1]))
        if observed != expected:
            return False, f"pair {pair_index}: observed {observed}, expected {expected}"
    return True, "ok"


def same_atom_pairs(left: MatrixSignature, right: MatrixSignature) -> bool:
    return left.atom_pairs.shape == right.atom_pairs.shape and np.array_equal(left.atom_pairs, right.atom_pairs)


def path_is_same_or_inside(path: Path, parent: Path) -> bool:
    try:
        normalize_path(path).relative_to(normalize_path(parent))
        return True
    except ValueError:
        return False


def find_dirs_with_file(
    root: Path,
    filename: str,
    exclude_dir: Path | None = None,
    exclude_prepared: bool = True,
) -> list[Path]:
    dirs: list[Path] = []
    for path in root.rglob(filename):
        if not path.is_file():
            continue
        candidate = path.parent
        if exclude_dir is not None and path_is_same_or_inside(candidate, exclude_dir):
            continue
        if exclude_prepared and (candidate / "band_prepare_manifest.json").is_file():
            continue
        dirs.append(candidate)
    return sorted(set(dirs), key=lambda item: item.stat().st_mtime, reverse=True)


def prepare_output_directory(out_dir: Path, overwrite: bool) -> None:
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        return
    if not out_dir.is_dir():
        raise NotADirectoryError(out_dir)
    if not overwrite:
        raise FileExistsError(f"{out_dir} already exists; pass --overwrite or choose another --out-dir")
    for filename in (*PREPARED_BAND_FILES, *STALE_BAND_OUTPUTS):
        target = out_dir / filename
        if target.exists() or target.is_symlink():
            target.unlink()


def materialize_file(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif link_mode == "symlink":
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {link_mode}")


def candidate_info_report(candidate_dir: Path, ham_sig: MatrixSignature) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dir": str(candidate_dir),
        "has_info": (candidate_dir / "info.json").is_file(),
        "has_poscar": (candidate_dir / "POSCAR").is_file(),
        "has_overlap": (candidate_dir / "overlap.h5").is_file(),
    }
    if not report["has_info"] or not report["has_poscar"]:
        report["status"] = "skip"
        report["reason"] = "missing info.json or POSCAR"
        return report
    try:
        info = json.loads((candidate_dir / "info.json").read_text(encoding="utf-8"))
        per_atom, per_element, spatial_total = orbital_counts_from_info(info, candidate_dir / "POSCAR")
        reported_orbits = int(info.get("orbits_quantity", -1))
        report.update(
            {
                "info_spinful": bool(info.get("spinful", False)),
                "reported_orbits_quantity": reported_orbits,
                "spatial_orbits_quantity": spatial_total,
                "per_element_spatial_orbitals": per_element,
            }
        )
        if reported_orbits != spatial_total:
            report["status"] = "bad"
            report["reason"] = (
                f"info.json orbits_quantity={reported_orbits}, but POSCAR/elements_orbital_map imply "
                f"{spatial_total} spatial orbitals"
            )
            return report
        h_spinless_ok, h_spinless_reason = matrix_shape_match_report(ham_sig, per_atom, spin_factor=1)
        h_spinful_ok, h_spinful_reason = matrix_shape_match_report(ham_sig, per_atom, spin_factor=2)
        report["hamiltonian_matches_spinless_info"] = h_spinless_ok
        report["hamiltonian_matches_spinful_info"] = h_spinful_ok
        if h_spinful_ok:
            report["hamiltonian_spin_mode"] = "spinful"
        elif h_spinless_ok:
            report["hamiltonian_spin_mode"] = "spinless"
        else:
            report["status"] = "bad"
            report["reason"] = f"H does not match this info.json as spinless ({h_spinless_reason}) or spinful ({h_spinful_reason})"
            return report

        if report["has_overlap"]:
            s_sig = read_matrix_signature(candidate_dir / "overlap.h5")
            report["overlap_unique_shapes"] = s_sig.unique_shapes
            if not same_atom_pairs(ham_sig, s_sig):
                report["overlap_matches_for_band"] = False
                report["overlap_reason"] = "atom_pairs differ between hamiltonian.h5 and overlap.h5"
            else:
                s_ok, s_reason = matrix_shape_match_report(s_sig, per_atom, spin_factor=1)
                report["overlap_matches_for_band"] = s_ok
                report["overlap_reason"] = s_reason
        report["status"] = "ok"
        return report
    except Exception as exc:
        report["status"] = "bad"
        report["reason"] = str(exc)
        return report


def choose_band_sources(
    case_root: Path,
    ham_dir: Path | None,
    overlap_dir: Path | None,
    out_dir: Path,
    requested_spin: str,
) -> dict[str, Any]:
    if ham_dir is not None:
        selected_ham_dir = normalize_path(ham_dir)
        if selected_ham_dir.is_file():
            selected_ham_dir = selected_ham_dir.parent
        if not (selected_ham_dir / "hamiltonian.h5").is_file():
            raise FileNotFoundError(f"{selected_ham_dir} does not contain hamiltonian.h5")
    else:
        candidates = find_dirs_with_file(case_root, "hamiltonian.h5", exclude_dir=out_dir)
        if not candidates:
            raise FileNotFoundError(f"No hamiltonian.h5 found under {case_root}")
        selected_ham_dir = candidates[0]

    ham_sig = read_matrix_signature(selected_ham_dir / "hamiltonian.h5")
    info_dirs = find_dirs_with_file(case_root, "info.json", exclude_dir=out_dir)
    reports = [candidate_info_report(candidate, ham_sig) for candidate in info_dirs]
    matching_info = [
        report for report in reports
        if report.get("status") == "ok"
        and report.get(f"hamiltonian_matches_{requested_spin}_info")
    ]
    if not matching_info:
        readable = "\n".join(
            f"- {report.get('dir')}: {report.get('reason', report.get('status'))}"
            for report in reports[:12]
        )
        raise ValueError(
            f"Could not find info.json/POSCAR matching hamiltonian.h5 as {requested_spin}.\n"
            "This usually means the Hamiltonian and overlap were generated with different basis labels.\n"
            f"Checked candidates:\n{readable}"
        )

    compatible: list[dict[str, Any]] = []
    for report in matching_info:
        candidate_dir = Path(str(report["dir"]))
        if overlap_dir is not None:
            requested_overlap_dir = normalize_path(overlap_dir)
            if requested_overlap_dir.is_file():
                requested_overlap_dir = requested_overlap_dir.parent
            if normalize_path(candidate_dir) != requested_overlap_dir:
                continue
            report = candidate_info_report(candidate_dir, ham_sig)
        if report.get("overlap_matches_for_band"):
            compatible.append(report)

    if not compatible:
        readable = "\n".join(
            f"- {report.get('dir')}: {report.get('overlap_reason', report.get('reason', report.get('status')))}"
            for report in matching_info[:12]
        )
        raise ValueError(
            "Found metadata matching hamiltonian.h5, but no spatial overlap.h5 compatible with dock calc-band.\n"
            "For spinful/SOC calc-band, overlap.h5 must stay spatial/spinless; do not feed a doubled spinful overlap.\n"
            f"Checked candidates:\n{readable}"
        )

    selected = compatible[0]
    selected_overlap_dir = Path(str(selected["dir"]))
    return {
        "ham_dir": selected_ham_dir,
        "overlap_dir": selected_overlap_dir,
        "ham_signature": ham_sig,
        "overlap_signature": read_matrix_signature(selected_overlap_dir / "overlap.h5"),
        "selected_report": selected,
        "candidate_reports": reports,
    }


def explain_band_prep_choice(choice: dict[str, Any], requested_spin: str) -> str:
    ham_sig: MatrixSignature = choice["ham_signature"]
    overlap_sig: MatrixSignature = choice["overlap_signature"]
    selected = choice["selected_report"]
    lines = [
        "检测结果 / Diagnosis:",
        f"- Hamiltonian source: {choice['ham_dir']}",
        f"- Overlap source: {choice['overlap_dir']}",
        f"- Band spin mode: {requested_spin}",
        f"- info.json spatial orbits_quantity: {selected.get('spatial_orbits_quantity')}",
        f"- per-element spatial orbitals: {selected.get('per_element_spatial_orbitals')}",
        f"- hamiltonian.h5 dtype: {ham_sig.entries_dtype}, max block shape: {ham_sig.max_shape}",
        f"- overlap.h5 dtype: {overlap_sig.entries_dtype}, max block shape: {overlap_sig.max_shape}",
    ]
    if requested_spin == "spinful":
        lines.extend(
            [
                "",
                "说明 / Note:",
                "- 对 SOC/spinful band 计算，hamiltonian.h5 是双倍 spinor block。",
                "- 但 overlap.h5 应保持 spatial/spinless，dock 会在内部扩展 S。",
                "- 不要把 orbits_quantity 改成两倍；它仍然是 spatial orbital 总数。",
            ]
        )
    return "\n".join(lines)



def validate_deeph_overlap_dir(
    output_dir: Path,
    expect_spin: str = "any",
    require_fermi_energy: bool = False,
    expect_fermi_energy_ev: float | None = None,
    fermi_tolerance_ev: float = 1.0e-6,
) -> dict[str, Any]:
    import h5py

    output_dir = normalize_path(output_dir)
    info_path = output_dir / "info.json"
    overlap_path = output_dir / "overlap.h5"
    poscar_path = output_dir / "POSCAR"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    if not overlap_path.is_file():
        raise FileNotFoundError(f"Missing overlap.h5: {overlap_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    manifest_path = output_dir / "direct_overlap_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    spinful = bool(info.get("spinful", False))
    spin_mode = "spinful" if spinful else "spinless"
    if expect_spin != "any" and spin_mode != expect_spin:
        raise ValueError(
            f"Spin mode mismatch: expected {expect_spin}, but {info_path} reports {spin_mode}. "
            "Regenerate overlap.h5 with or without --spinful to match the trained model."
        )

    orbital_map = info.get("elements_orbital_map")
    if not isinstance(orbital_map, dict) or not orbital_map:
        raise ValueError(f"{info_path} does not contain a valid elements_orbital_map")
    base_orbitals = {
        element: orbital_count_from_l_list(angular_momenta)
        for element, angular_momenta in orbital_map.items()
    }
    spin_factor = 2 if spinful else 1
    expected_dims = {
        element: count * spin_factor
        for element, count in base_orbitals.items()
    }
    fermi_energy_ev = info.get("fermi_energy_eV")
    if fermi_energy_ev is None:
        raise ValueError(f"{info_path} does not contain fermi_energy_eV")
    fermi_energy_ev = float(fermi_energy_ev)

    fermi_source = "unknown"
    fermi_provided = None
    if manifest is not None:
        fermi_report = manifest.get("fermi_energy", {})
        fermi_source = str(fermi_report.get("source", "unknown"))
        fermi_provided = bool(fermi_report.get("provided", False))

    if require_fermi_energy:
        if manifest is None:
            raise ValueError(
                "Cannot prove that fermi_energy_eV came from an SCF calculation because "
                "direct_overlap_manifest.json is missing. Regenerate with --fermi-energy-ev "
                "or --fermi-log, or provide a manifest."
            )
        if not fermi_provided:
            raise ValueError(
                "Fermi energy is required, but this overlap directory was generated without "
                "an SCF Fermi energy. Regenerate with --fermi-energy-ev VALUE or --fermi-log PATH."
            )
    if expect_fermi_energy_ev is not None:
        difference = abs(fermi_energy_ev - expect_fermi_energy_ev)
        if difference > fermi_tolerance_ev:
            raise ValueError(
                f"Fermi energy mismatch: info.json has {fermi_energy_ev:.12g} eV, "
                f"expected {expect_fermi_energy_ev:.12g} eV, diff {difference:.3e} eV."
            )

    with h5py.File(overlap_path, "r") as handle:
        for dataset in ["atom_pairs", "chunk_boundaries", "chunk_shapes", "entries"]:
            if dataset not in handle:
                raise ValueError(f"{overlap_path} is missing required dataset {dataset!r}")

        atom_pairs_shape = tuple(int(item) for item in handle["atom_pairs"].shape)
        chunk_boundaries = np.asarray(handle["chunk_boundaries"])
        chunk_shapes = np.asarray(handle["chunk_shapes"])
        entries_size = int(handle["entries"].shape[0])

    if len(atom_pairs_shape) != 2 or atom_pairs_shape[1] != 5:
        raise ValueError(f"atom_pairs must have shape (npairs, 5), got {atom_pairs_shape}")
    if chunk_shapes.ndim != 2 or chunk_shapes.shape[1] != 2:
        raise ValueError(f"chunk_shapes must have shape (npairs, 2), got {tuple(chunk_shapes.shape)}")
    if chunk_boundaries.ndim != 1 or chunk_boundaries.shape[0] != chunk_shapes.shape[0] + 1:
        raise ValueError(
            "chunk_boundaries must be one-dimensional with length npairs + 1; "
            f"got {tuple(chunk_boundaries.shape)} for npairs={chunk_shapes.shape[0]}"
        )
    if chunk_shapes.shape[0] != atom_pairs_shape[0]:
        raise ValueError(
            f"atom_pairs and chunk_shapes disagree on npairs: {atom_pairs_shape[0]} vs {chunk_shapes.shape[0]}"
        )
    if int(chunk_boundaries[0]) != 0:
        raise ValueError("chunk_boundaries must start from 0")
    if int(chunk_boundaries[-1]) != entries_size:
        raise ValueError(
            f"chunk_boundaries[-1] must equal entries length; got {chunk_boundaries[-1]} vs {entries_size}"
        )

    block_sizes = np.prod(chunk_shapes, axis=1, dtype=np.int64)
    boundary_steps = np.diff(chunk_boundaries)
    if not np.array_equal(boundary_steps, block_sizes):
        bad_index = int(np.nonzero(boundary_steps != block_sizes)[0][0])
        raise ValueError(
            f"chunk boundary mismatch at pair {bad_index}: "
            f"boundary step {boundary_steps[bad_index]}, shape product {block_sizes[bad_index]}"
        )

    observed_dims = sorted({int(item) for item in chunk_shapes.reshape(-1)})
    valid_dims = set(expected_dims.values())
    invalid_dims = [dimension for dimension in observed_dims if dimension not in valid_dims]
    if invalid_dims:
        raise ValueError(
            f"Observed block dimensions {invalid_dims} do not match {spin_mode} orbital dimensions "
            f"{expected_dims}. This often means spinful and spinless overlap/model data were mixed."
        )

    poscar_report: dict[str, Any] | None = None
    if poscar_path.is_file():
        poscar_summary = read_poscar_summary(poscar_path)
        missing_elements = [
            element for element in poscar_summary["composition"]
            if element not in base_orbitals
        ]
        if missing_elements:
            raise ValueError(f"POSCAR elements missing from elements_orbital_map: {missing_elements}")
        expected_spatial_orbitals = sum(
            count * base_orbitals[element]
            for element, count in poscar_summary["composition"].items()
        )
        expected_spinful_orbitals = expected_spatial_orbitals * spin_factor
        reported_total_orbitals = int(info.get("orbits_quantity", -1))
        if reported_total_orbitals != expected_spatial_orbitals:
            raise ValueError(
                f"orbits_quantity mismatch: info.json reports {reported_total_orbitals}, "
                f"but POSCAR and elements_orbital_map imply {expected_spatial_orbitals} spatial orbitals. "
                "In the DeepH/HPRO new interface, info.json orbits_quantity stores the spatial-orbital count; "
                "spinful doubling is represented in overlap.h5 block shapes, not in orbits_quantity."
            )
        poscar_report = {
            "formula": poscar_summary["formula"],
            "natoms": poscar_summary["natoms"],
            "expected_spatial_orbitals": expected_spatial_orbitals,
            "expected_matrix_orbitals": expected_spinful_orbitals,
        }

    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "spin_mode": spin_mode,
        "fermi_energy_eV": fermi_energy_ev,
        "fermi_energy_source": fermi_source,
        "fermi_energy_provided": fermi_provided,
        "base_orbitals_by_element": base_orbitals,
        "expected_block_dims_by_element": expected_dims,
        "observed_block_dims": observed_dims,
        "npairs": int(chunk_shapes.shape[0]),
        "entries": entries_size,
        "poscar": poscar_report,
    }


def ensure_output_targets(output_dir: Path, manifest_path: Path, overwrite: bool, input_poscar: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = [output_dir / "overlap.h5", output_dir / "info.json", manifest_path]
    output_poscar = output_dir / "POSCAR"
    if normalize_path(output_poscar) != normalize_path(input_poscar):
        protected.append(output_poscar)
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        existing_text = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing output files. Use --overwrite or choose a new --output-dir:\n"
            f"{existing_text}"
        )


def copy_poscar(input_poscar: Path, output_dir: Path) -> None:
    output_poscar = output_dir / "POSCAR"
    if normalize_path(output_poscar) != normalize_path(input_poscar):
        shutil.copy2(input_poscar, output_poscar)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DeepH-format overlap.h5 directly from a VASP POSCAR and numerical atomic orbitals. "
            "Supported basis backends are abacus (.orb) and openmx (.pao)."
        )
    )
    parser.add_argument("data_dir", type=Path, help="Directory containing POSCAR.")
    parser.add_argument("basis_dir", type=Path, help="Directory containing basis files.")
    parser.add_argument(
        "--basis-code",
        choices=["abacus", "openmx"],
        default="abacus",
        help="Basis file format/backend. Default: abacus.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: data_dir.")
    parser.add_argument(
        "--ecut",
        type=float,
        default=100.0,
        help="Cutoff parameter passed to HPRO calc_overlap. Use 100 for the SG15 DZP 100Ry basis.",
    )
    parser.add_argument(
        "--basis-map",
        action="append",
        default=[],
        metavar="ELEMENT=BASIS_PATH",
        help="Override automatic basis-file discovery. Can be repeated.",
    )
    parser.add_argument(
        "--openmx-basis",
        action="append",
        default=[],
        metavar="ELEMENT=BASIS_SPEC",
        help="OpenMX basis label, e.g. Au=Au7.0-s2p2d2f1. Required when --basis-code openmx.",
    )
    parser.add_argument("--expect-formula", default=None, help="Optional formula guard, e.g. Au16S8Mo4.")
    parser.add_argument("--expect-natoms", type=int, default=None, help="Optional atom-count guard.")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest path. Default: output_dir/direct_overlap_manifest.json.")
    parser.add_argument("--norm-tol", type=float, default=1.0e-5, help="Maximum allowed |orbital_norm - 1|.")
    parser.add_argument(
        "--allow-unnormalized-orbitals",
        action="store_true",
        help="Warn instead of failing when orbital normalization exceeds --norm-tol.",
    )
    parser.add_argument(
        "--strict-norm-check",
        action="store_true",
        help="Fail on normalization deviations for all basis backends. ABACUS is strict by default; OpenMX warns by default.",
    )
    parser.add_argument(
        "--spinful",
        action="store_true",
        help=(
            "Write a spinful overlap matrix by duplicating the spin-independent spatial overlap "
            "onto the spin-up and spin-down diagonal blocks. Use this for spinor/SOC DeepH data "
            "formats when the basis itself is spin independent."
        ),
    )
    parser.add_argument(
        "--expect-spin",
        choices=["any", "spinless", "spinful"],
        default="any",
        help="Optional guard for the requested output spin mode. Useful in production scripts.",
    )
    parser.add_argument(
        "--fermi-energy-ev",
        type=float,
        default=None,
        help="SCF Fermi energy / chemical potential in eV to write into info.json.",
    )
    parser.add_argument(
        "--fermi-log",
        type=Path,
        default=None,
        help="Parse SCF Fermi energy from a log/output file. OpenMX 'Chemical potential (Hartree)' is supported.",
    )
    parser.add_argument(
        "--require-fermi-energy",
        action="store_true",
        help="Fail unless --fermi-energy-ev or --fermi-log is provided.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing overlap.h5/info.json/manifest in output-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write manifest without computing overlap.h5.")
    parser.add_argument("--verbose", action="store_true", help="Print progress details.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(message)s")

    data_dir = normalize_path(args.data_dir)
    basis_dir = normalize_path(args.basis_dir)
    output_dir = normalize_path(args.output_dir) if args.output_dir else data_dir
    manifest_path = normalize_path(args.manifest) if args.manifest else output_dir / "direct_overlap_manifest.json"
    poscar_path = data_dir / "POSCAR"

    if not poscar_path.is_file():
        raise FileNotFoundError(f"Missing POSCAR: {poscar_path}")
    if not basis_dir.is_dir():
        raise FileNotFoundError(f"Missing basis directory: {basis_dir}")
    requested_spin = "spinful" if args.spinful else "spinless"
    if args.expect_spin != "any" and args.expect_spin != requested_spin:
        raise ValueError(
            f"--expect-spin {args.expect_spin} conflicts with requested output mode {requested_spin}. "
            "Add --spinful for spinful output or remove it for spinless output."
        )
    fermi_report = resolve_fermi_energy(args)

    poscar_summary = read_poscar_summary(poscar_path)
    if args.expect_natoms is not None and poscar_summary["natoms"] != args.expect_natoms:
        raise ValueError(f"Expected {args.expect_natoms} atoms, found {poscar_summary['natoms']} in {poscar_path}")
    if args.expect_formula is not None:
        expected_composition = parse_formula(args.expect_formula)
        if dict(poscar_summary["composition"]) != expected_composition:
            raise ValueError(
                f"Expected formula {args.expect_formula}, found {poscar_summary['formula']} "
                f"with composition {poscar_summary['composition']}"
            )

    ensure_output_targets(output_dir, manifest_path, args.overwrite, poscar_path)
    basis_map = parse_element_path_map(args.basis_map, "--basis-map")
    openmx_basis_specs = parse_openmx_basis_items(args.openmx_basis)
    if args.basis_code == "abacus" and openmx_basis_specs:
        raise ValueError("--openmx-basis is only valid with --basis-code openmx")
    runtime = load_runtime()

    with poscar_path.open("r", encoding="utf-8") as handle:
        structure = runtime.from_poscar(handle)
    aodata = build_aodata(structure, basis_dir, args.basis_code, basis_map, openmx_basis_specs, runtime)
    attach_calc_phiq(aodata, runtime)
    basis = basis_report(aodata, runtime)
    allow_unnormalized = args.allow_unnormalized_orbitals or (
        args.basis_code == "openmx" and not args.strict_norm_check
    )
    validate_basis_norms(basis, args.norm_tol, allow_unnormalized)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "method": f"direct HPRO/deepx-dock overlap integration from {args.basis_code} numerical orbitals",
        "notes": [
            "This command directly integrates numerical atomic orbitals and writes DeepH-format overlap.h5.",
            "It does not run ABACUS get_S or OpenMX SCF.",
        ],
        "inputs": {
            "data_dir": str(data_dir),
            "basis_dir": str(basis_dir),
            "poscar": poscar_summary,
            "basis": basis,
        },
        "parameters": {
            "basis_code": args.basis_code,
            "ecut": args.ecut,
            "spinful": bool(args.spinful),
            "expect_spin": args.expect_spin,
            "fermi_energy_eV": fermi_report["fermi_energy_eV"],
            "fermi_energy_source": fermi_report["source"],
            "norm_tolerance": args.norm_tol,
            "allow_unnormalized_orbitals": allow_unnormalized,
        },
        "fermi_energy": fermi_report,
        "software": software_versions(),
        "output": {
            "output_dir": str(output_dir),
            "manifest": str(manifest_path),
            "dry_run": bool(args.dry_run),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        LOGGER.info("Dry run completed; wrote %s", manifest_path)
        return

    LOGGER.info("Computing overlap for %s atoms with ecut=%s", poscar_summary["natoms"], args.ecut)
    overlaps = runtime.calc_overlap(aodata, Ecut=args.ecut)
    if args.spinful:
        overlaps.spinless_to_spinful()
    overlaps.structure.efermi = float(fermi_report["fermi_energy_eV"]) / HARTREE_TO_EV
    if not fermi_report["provided"]:
        LOGGER.warning("%s", fermi_report["note"])
    runtime.save_mat_deeph(output_dir, overlaps, "o")
    copy_poscar(poscar_path, output_dir)

    overlap_path = output_dir / "overlap.h5"
    info_path = output_dir / "info.json"
    if not overlap_path.is_file():
        raise FileNotFoundError(f"Expected overlap output was not created: {overlap_path}")
    if not info_path.is_file():
        raise FileNotFoundError(f"Expected DeepH metadata output was not created: {info_path}")

    manifest["output"]["overlap_h5"] = hdf5_summary(overlap_path)
    manifest["output"]["info_json_sha256"] = sha256_file(info_path)
    manifest["output"]["poscar_sha256"] = sha256_file(output_dir / "POSCAR")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["output"]["validation"] = validate_deeph_overlap_dir(
        output_dir,
        expect_spin=requested_spin,
        require_fermi_energy=args.require_fermi_energy,
        expect_fermi_energy_ev=float(fermi_report["fermi_energy_eV"]) if fermi_report["provided"] else None,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest["output"], indent=2))
    LOGGER.info("Wrote %s", overlap_path)
    LOGGER.info("Wrote %s", manifest_path)


def build_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a DeepH overlap directory before running deeph-infer."
    )
    parser.add_argument("output_dir", type=Path, help="Directory containing overlap.h5, info.json, and preferably POSCAR.")
    parser.add_argument(
        "--expect-spin",
        choices=["any", "spinless", "spinful"],
        default="any",
        help="Fail if info.json reports a different spin mode.",
    )
    parser.add_argument(
        "--require-fermi-energy",
        action="store_true",
        help="Fail unless the direct-overlap manifest proves an SCF Fermi energy was provided.",
    )
    parser.add_argument(
        "--expect-fermi-energy-ev",
        type=float,
        default=None,
        help="Fail unless info.json fermi_energy_eV matches this value.",
    )
    parser.add_argument(
        "--fermi-tol-ev",
        type=float,
        default=1.0e-6,
        help="Tolerance for --expect-fermi-energy-ev.",
    )
    return parser


def check_main() -> None:
    parser = build_check_parser()
    args = parser.parse_args()
    report = validate_deeph_overlap_dir(
        args.output_dir,
        expect_spin=args.expect_spin,
        require_fermi_energy=args.require_fermi_energy,
        expect_fermi_energy_ev=args.expect_fermi_energy_ev,
        fermi_tolerance_ev=args.fermi_tol_ev,
    )
    print(json.dumps(report, indent=2))


def build_band_prep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a DeepH-dock calc-band directory and prevent spinful/spinless "
            "hamiltonian/overlap metadata mismatches."
        )
    )
    parser.add_argument("case_root", type=Path, help="Case root containing inference outputs and/or dft directories.")
    parser.add_argument("--ham-dir", type=Path, default=None, help="Directory or hamiltonian.h5 file to use.")
    parser.add_argument("--overlap-dir", type=Path, default=None, help="Directory or overlap.h5 file to use.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: case_root/band_ready.")
    parser.add_argument(
        "--spin",
        choices=["auto", "spinless", "spinful"],
        default="auto",
        help="Band spin mode. auto currently chooses spinful when Hamiltonian matches spinful metadata.",
    )
    parser.add_argument("--fermi-energy-ev", type=float, default=None, help="Override fermi_energy_eV in output info.json.")
    parser.add_argument("--overwrite", action="store_true", help="Replace prepared files in --out-dir.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask for confirmation before writing files. Useful in batch scripts.",
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Only print the detected action; do not write files.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to place large HDF5 files in --out-dir. Falls back to copy when linking fails.",
    )
    parser.add_argument("--run-calc-band", action="store_true", help="Run dock compute eigen calc-band after preparation.")
    parser.add_argument("--dock-bin", default="dock", help="dock executable for --run-calc-band.")
    parser.add_argument("--parallel-num", "-p", default="5", help="parallel-num passed to calc-band.")
    parser.add_argument("--thread-num", default="1", help="thread-num passed to calc-band.")
    return parser


def band_prep_main() -> None:
    parser = build_band_prep_parser()
    args = parser.parse_args()
    case_root = normalize_path(args.case_root)
    if not case_root.is_dir():
        raise FileNotFoundError(f"Missing case root: {case_root}")
    out_dir = normalize_path(args.out_dir) if args.out_dir is not None else case_root / "band_ready"

    requested_spin = args.spin
    if requested_spin == "auto":
        requested_spin = "spinful"
    choice = choose_band_sources(
        case_root=case_root,
        ham_dir=args.ham_dir,
        overlap_dir=args.overlap_dir,
        out_dir=out_dir,
        requested_spin=requested_spin,
    )
    message = explain_band_prep_choice(choice, requested_spin)
    print(message)

    if args.diagnose_only:
        print("\n诊断模式：没有写入任何文件。")
        return

    if not args.yes and sys.stdin.isatty():
        answer = input(f"\n是否生成 calc-band 可用目录 {out_dir}? [Y/n] ").strip().lower()
        if answer not in {"", "y", "yes"}:
            print("已取消。")
            return

    selected = choice["selected_report"]
    selected_overlap_dir = Path(str(choice["overlap_dir"]))
    selected_ham_dir = Path(str(choice["ham_dir"]))
    info = json.loads((selected_overlap_dir / "info.json").read_text(encoding="utf-8"))
    info["spinful"] = requested_spin == "spinful"
    info["orbits_quantity"] = int(selected["spatial_orbits_quantity"])
    if args.fermi_energy_ev is not None:
        info["fermi_energy_eV"] = float(args.fermi_energy_ev)
    if "fermi_energy_eV" not in info:
        info["fermi_energy_eV"] = 0.0

    poscar_src = selected_overlap_dir / "POSCAR"
    k_path_src = selected_ham_dir / "K_PATH"
    if not k_path_src.is_file():
        raise FileNotFoundError(f"Missing K_PATH beside hamiltonian.h5: {k_path_src}")

    prepare_output_directory(out_dir, overwrite=args.overwrite)
    shutil.copy2(poscar_src, out_dir / "POSCAR")
    shutil.copy2(k_path_src, out_dir / "K_PATH")
    materialize_file(selected_ham_dir / "hamiltonian.h5", out_dir / "hamiltonian.h5", args.link_mode)
    materialize_file(selected_overlap_dir / "overlap.h5", out_dir / "overlap.h5", args.link_mode)
    (out_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ham_sig: MatrixSignature = choice["ham_signature"]
    overlap_sig: MatrixSignature = choice["overlap_signature"]
    manifest = {
        "schema_version": "direct-overlap-band-prep/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_root": str(case_root),
        "output_dir": str(out_dir),
        "hamiltonian_source": str(selected_ham_dir),
        "overlap_source": str(selected_overlap_dir),
        "spin_mode": requested_spin,
        "fermi_energy_eV": info.get("fermi_energy_eV"),
        "link_mode": args.link_mode,
        "validation": {
            "per_element_spatial_orbitals": selected["per_element_spatial_orbitals"],
            "spatial_orbits_quantity": selected["spatial_orbits_quantity"],
            "matrix_orbits_quantity": selected["spatial_orbits_quantity"] * (2 if requested_spin == "spinful" else 1),
            "hamiltonian_unique_shapes": ham_sig.unique_shapes,
            "overlap_unique_shapes": overlap_sig.unique_shapes,
            "hamiltonian_entries_dtype": ham_sig.entries_dtype,
            "overlap_entries_dtype": overlap_sig.entries_dtype,
            "atom_pairs": int(len(ham_sig.atom_pairs)),
        },
        "notes": [
            "For spinful/SOC calc-band, hamiltonian.h5 is spinful but overlap.h5 stays spatial/spinless.",
            "orbits_quantity remains the spatial-orbital count.",
        ],
        "next_commands": [
            f"cd {out_dir}",
            f"{args.dock_bin} compute eigen calc-band ./ --parallel-num {args.parallel_num} --thread-num {args.thread_num}",
        ],
    }
    (out_dir / "band_prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n已生成 / Prepared:")
    print(f"  {out_dir}")
    print("下一步 / Next:")
    print(f"  cd {out_dir}")
    print(f"  {args.dock_bin} compute eigen calc-band ./ --parallel-num {args.parallel_num} --thread-num {args.thread_num}")

    if args.run_calc_band:
        command = [
            args.dock_bin,
            "compute",
            "eigen",
            "calc-band",
            "./",
            "--parallel-num",
            str(args.parallel_num),
            "--thread-num",
            str(args.thread_num),
        ]
        print("+ " + " ".join(command))
        subprocess.run(command, cwd=str(out_dir), check=True)


if __name__ == "__main__":
    main()
