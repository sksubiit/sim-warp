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
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse
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

    return parts[0], sanitize_relative_path(parts[1], context=uri)


def sanitize_relative_path(path_str, context):
    """Normalize a mesh path and reject absolute or escaping paths."""
    normalized = PurePosixPath(path_str.replace("\\", "/"))
    if normalized.is_absolute():
        raise ValueError(f"Absolute mesh paths are not supported: {context}")

    parts = [part for part in normalized.parts if part not in ("", ".")]
    if not parts:
        raise ValueError(f"Empty mesh path is not supported: {context}")
    if any(part == ".." for part in parts):
        raise ValueError(f"Mesh path cannot escape its base directory: {context}")

    return Path(*parts)


def derive_file_uri_relative_path(src_file):
    """Map a file:// URI into a stable relative path under the localized mesh tree."""
    parts = [part for part in src_file.parts if part not in (src_file.anchor, "/")]
    if not parts:
        raise ValueError(f"Cannot derive output path for file URI: {src_file}")
    return Path("external") / Path(*parts)


def ensure_within_directory(path, root, context):
    """Ensure a resolved path remains within the expected root directory."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Mesh path escapes '{resolved_root}': {context}") from exc
    return resolved_path


def resolve_relative_mesh_reference(mesh_ref, urdf_dir, package_path):
    """Resolve a plain relative mesh reference and keep it within the package root."""
    normalized = PurePosixPath(mesh_ref.replace("\\", "/"))
    if normalized.is_absolute():
        raise ValueError(f"Absolute mesh paths are not supported: {mesh_ref}")

    parts = [part for part in normalized.parts if part not in ("", ".")]
    if not parts:
        raise ValueError(f"Empty mesh path is not supported: {mesh_ref}")

    src_file = ensure_within_directory(urdf_dir / Path(*parts), package_path, mesh_ref)
    return src_file, src_file.relative_to(package_path.resolve())


def resolve_mesh_reference(mesh_ref, urdf_path, package_name, package_path):
    """Resolve a mesh reference to its source file and localized relative path."""
    parsed = urlparse(mesh_ref)
    urdf_dir = Path(urdf_path).parent.resolve()

    if mesh_ref.startswith("package://"):
        ref_package_name, relative_path = parse_package_uri(mesh_ref)
        if ref_package_name != package_name:
            raise FileNotFoundError(
                f"Referenced mesh belongs to package '{ref_package_name}', expected '{package_name}': {mesh_ref}"
            )
        src_file = ensure_within_directory(package_path / relative_path, package_path, mesh_ref)
        return src_file, relative_path

    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"Unsupported file URI host in mesh reference: {mesh_ref}")
        src_file = Path(unquote(parsed.path)).resolve()
        relative_path = derive_file_uri_relative_path(src_file)
        return src_file, relative_path

    if parsed.scheme:
        raise ValueError(f"Unsupported mesh URI scheme in '{mesh_ref}'")

    return resolve_relative_mesh_reference(mesh_ref, urdf_dir, package_path)


def localized_mesh_path(mesh_ref, urdf_path, package_name, package_path, meshes_local_name):
    """Return the localized mesh path that should be written back into the URDF."""
    _, relative_path = resolve_mesh_reference(mesh_ref, urdf_path, package_name, package_path)
    return (Path(".") / meshes_local_name / relative_path).as_posix()


def copy_meshes(urdf_path, package_name, package_path, dst_meshes_path):
    """Copy all mesh files referenced by the URDF into the localized mesh tree."""
    dst_meshes_path = dst_meshes_path.resolve()
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
        src_file, relative_path = resolve_mesh_reference(mesh_ref, urdf_path, package_name, package_path)

        if not src_file.exists():
            raise FileNotFoundError(f"Referenced mesh not found: {src_file}")

        dst_file = ensure_within_directory(dst_meshes_path / relative_path, dst_meshes_path, mesh_ref)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

        copied_files.append(relative_path.as_posix())
        if len(copied_files) <= 5:
            print(f"  ✓ {relative_path.as_posix()}")
        elif len(copied_files) == 6:
            print(f"  ... and {len(mesh_refs) - 5} more files")

    return copied_files


def rewrite_urdf_paths(urdf_path, package_name, package_path, meshes_local_name="meshes"):
    """Rewrite mesh paths to localized relative paths."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    meshes_rewritten = 0

    for mesh_elem in root.findall(".//mesh"):
        filename = mesh_elem.get("filename")
        if filename:
            new_filename = localized_mesh_path(filename, urdf_path, package_name, package_path, meshes_local_name)
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

    urdf_tree = rewrite_urdf_paths(urdf_path, package_name, package_path, meshes_local_name=meshes_dir_name)
    urdf_tree.write(localized_urdf_path, encoding="utf-8", xml_declaration=True)

    print("==================================================")
    print(f"✓ Localized URDF saved to: {localized_urdf_path}")
    print(f"✓ Meshes saved to: {meshes_output_dir}")
    print("==================================================")


if __name__ == "__main__":
    main()
