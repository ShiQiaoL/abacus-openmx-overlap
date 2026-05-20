#!/usr/bin/env python3
"""Compute DeepH overlap.h5 directly from POSCAR and numerical atomic orbitals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import platform
import re
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


LOGGER = logging.getLogger("direct-overlap")
SCHEMA_VERSION = "direct-overlap-basis/2"
ANGULAR_MOMENTA = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


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


def orbital_count_from_l_list(angular_momenta: list[int]) -> int:
    return int(sum(2 * int(angular_momentum) + 1 for angular_momentum in angular_momenta))


def validate_deeph_overlap_dir(output_dir: Path, expect_spin: str = "any") -> dict[str, Any]:
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
        expected_total_orbitals = sum(
            count * base_orbitals[element] * spin_factor
            for element, count in poscar_summary["composition"].items()
        )
        reported_total_orbitals = int(info.get("orbits_quantity", -1))
        if reported_total_orbitals != expected_total_orbitals:
            raise ValueError(
                f"orbits_quantity mismatch: info.json reports {reported_total_orbitals}, "
                f"but POSCAR and elements_orbital_map imply {expected_total_orbitals}"
            )
        poscar_report = {
            "formula": poscar_summary["formula"],
            "natoms": poscar_summary["natoms"],
            "expected_total_orbitals": expected_total_orbitals,
        }

    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "spin_mode": spin_mode,
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
            "norm_tolerance": args.norm_tol,
            "allow_unnormalized_orbitals": allow_unnormalized,
        },
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
    manifest["output"]["validation"] = validate_deeph_overlap_dir(output_dir, expect_spin=requested_spin)
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
    return parser


def check_main() -> None:
    parser = build_check_parser()
    args = parser.parse_args()
    report = validate_deeph_overlap_dir(args.output_dir, expect_spin=args.expect_spin)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
