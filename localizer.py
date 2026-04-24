#!/usr/bin/env python3
"""
Validator: Robot-agnostic Localization + Validation (ROS Package Standard)

Follows ROS package structure in conda:
  - Package: share/PACKAGE_NAME/ (e.g., ergoCub, iCub)
  - Variants: PACKAGE_NAME/robots/VARIANT_NAME/ (e.g., ergoCubGazeboSN001)
  - Meshes: any package-relative asset path referenced by the URDF

Features:
1. List available packages and variants
2. Auto-discovers URDF and referenced assets (standards-compliant)
3. Copies all referenced mesh files locally preserving relative subpaths
4. Rewrites URDF paths to relative localized paths

Result: Self-contained outputs/ folder ready for conversion (config design)

TRULY ROBOT-AGNOSTIC:
  - Works with any robot package (ergoCub, iCub, custom, etc.)
  - No code changes needed for new robots
  - Follows ROS/conda conventions explicitly
  - Scales across multiple packages and variants

Usage:
  python localizer.py --robot ergoCub/ergoCubGazeboSN001
  python localizer.py --robot iCub/iCub3
"""

import shutil
import sys
import os
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET


def discover_packages():
    """List available ROS packages in conda environment (share directory)."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return []

    share_path = Path(conda_prefix) / "share"
    packages = []

    for item in share_path.iterdir():
        if item.is_dir() and (item / "robots").exists():
            packages.append(item.name)

    return sorted(packages)


def discover_variants(package_name):
    """List available robot variants in a package."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return []

    robots_dir = Path(conda_prefix) / "share" / package_name / "robots"
    variants = []

    if robots_dir.exists():
        for variant_dir in robots_dir.iterdir():
            if variant_dir.is_dir() and (variant_dir / "model.urdf").exists():
                variants.append(variant_dir.name)

    return sorted(variants)


def find_robot_and_meshes(package_name, variant_name):
    """
    Find URDF and package root for given package/variant (ROS standard).

    Supports ROS package layout in conda:
    - share/PACKAGE_NAME/robots/VARIANT_NAME/model.urdf
    - share/PACKAGE_NAME/<package-relative asset paths referenced by the URDF>
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        raise EnvironmentError("CONDA_PREFIX not set. Activate conda environment first.")

    share_path = Path(conda_prefix) / "share"
    package_path = share_path / package_name

    if not package_path.exists():
        available = discover_packages()
        raise FileNotFoundError(
            f"Package '{package_name}' not found in {share_path}\n"
            f"Available packages: {', '.join(available) if available else 'none'}"
        )

    variant_path = package_path / "robots" / variant_name
    urdf_path = variant_path / "model.urdf"

    if not urdf_path.exists():
        available_variants = discover_variants(package_name)
        raise FileNotFoundError(
            f"Variant '{variant_name}' not found in package '{package_name}'\n"
            f"Available variants: {', '.join(available_variants) if available_variants else 'none'}"
        )

    return urdf_path, package_path


def parse_package_uri(uri):
    """Split package://PACKAGE/path/to/file into (PACKAGE, relative_path)."""
    if not uri.startswith("package://"):
        raise ValueError(f"Unsupported mesh URI: {uri}")

    package_spec = uri[len("package://"):]
    parts = package_spec.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Malformed package URI: {uri}")

    return parts[0], Path(parts[1])


def copy_meshes(urdf_path, package_name, package_path, dst_meshes_path):
    """Copy all mesh files referenced by the URDF into the localized mesh tree."""
    dst_meshes_path.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    mesh_refs = []
    seen_refs = set()
    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if filename and filename not in seen_refs:
            mesh_refs.append(filename)
            seen_refs.add(filename)

    print(f"Found {len(mesh_refs)} referenced mesh files")

    copied_files = []
    for mesh_ref in mesh_refs:
        if mesh_ref.startswith("package://"):
            ref_package_name, relative_path = parse_package_uri(mesh_ref)
            if ref_package_name != package_name:
                raise FileNotFoundError(
                    f"Referenced mesh belongs to package '{ref_package_name}', expected '{package_name}': {mesh_ref}"
                )
            src_file = package_path / relative_path
        else:
            relative_path = Path(mesh_ref)
            src_file = (Path(urdf_path).parent / relative_path).resolve()

        if not src_file.exists():
            raise FileNotFoundError(f"Referenced mesh not found: {src_file}")

        dst_file = dst_meshes_path / relative_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

        copied_files.append(relative_path.as_posix())
        if len(copied_files) <= 5:
            print(f"  ✓ {relative_path.as_posix()}")
        elif len(copied_files) == 6:
            print(f"  ... and {len(mesh_refs) - 5} more files")

    return copied_files


def rewrite_urdf_paths(urdf_path, meshes_local_name="meshes"):
    """Rewrite package mesh paths to localized relative paths."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    meshes_rewritten = 0

    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if filename and filename.startswith("package://"):
            _, relative_path = parse_package_uri(filename)
            new_filename = (Path(".") / meshes_local_name / relative_path).as_posix()
            mesh_elem.set("filename", new_filename)
            meshes_rewritten += 1
            print(f"  ✓ {filename} → {new_filename}")

    print(f"✓ Rewrote {meshes_rewritten} mesh paths to relative paths")

    return tree


def main():
    parser = argparse.ArgumentParser(description="Localize URDF and meshes for MuJoCo Conversion.")
    parser.add_argument("--robot", type=str, required=True, help="Robot as PACKAGE/VARIANT (e.g. ergoCub/ergoCubGazeboSN001)")
    args = parser.parse_args()

    if "/" not in args.robot:
        print("ERROR: Robot must be PACKAGE/VARIANT")
        sys.exit(1)

    package_name, variant_name = args.robot.split("/", 1)

    print(f"Localizing {args.robot}...")
    urdf_path, package_path = find_robot_and_meshes(package_name, variant_name)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    meshes_dir_name = f"meshes_{variant_name}"
    meshes_output_dir = output_dir / meshes_dir_name
    localized_urdf_path = output_dir / f"{variant_name}_localized.urdf"

    copy_meshes(urdf_path, package_name, package_path, meshes_output_dir)

    urdf_tree = rewrite_urdf_paths(urdf_path, meshes_local_name=meshes_dir_name)
    urdf_tree.write(localized_urdf_path, encoding="utf-8", xml_declaration=True)

    print("==================================================")
    print(f"✓ Localized URDF saved to: {localized_urdf_path}")
    print(f"✓ Meshes saved to: {meshes_output_dir}")
    print("==================================================")


if __name__ == "__main__":
    main()
