"""
MCJF_wrap.py - Universal URDF to MuJoCo MJCF Converter

Pipeline:
  Phase 1: Config Loading        - Validate YAML config
  Phase 2: Model Processing      - Resolve package URIs
  Phase 2b: Asset Localization   - Copy meshes to ./assets/
  Phase 3: Scene Building        - Convert URDF → MJCF
  Phase 4: Validation            - MuJoCo load test
  Phase 5: Export                - Save MJCF with ./assets
  Phase 6: Summary               - Print results

Usage:
  python3 MCJF_warp.py --config ergocub.yaml

Outputs:
  robot_clean.xml     ← Final MJCF (meshdir="./assets")
  assets/             ← Self-contained mesh bundle
"""

import os
import sys
import re
import argparse
import yaml
import shutil
from pathlib import Path
from typing import Tuple, Set

try:
    import mujoco
except ImportError:
    print("[ERROR] MuJoCo not installed. Run: pip install mujoco")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

class ConfigLoader:
    """Load and validate YAML configuration"""

    REQUIRED_FIELDS = ['model_path', 'output_path', 'mesh_dir']

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.data = {}
        self.errors = []

    def _expand_path(self, path_str: str) -> str:
        """Expand env vars, ~, and relative paths"""
        expanded = os.path.expandvars(path_str)
        expanded = os.path.expanduser(expanded)
        if os.path.isabs(expanded):
            return expanded
        config_dir = os.path.dirname(os.path.abspath(self.yaml_path))
        return os.path.join(config_dir, expanded)

    def load(self) -> bool:
        print(f"\n[PHASE 1] Config Loading")
        print(f"  config: {self.yaml_path}")

        if not os.path.exists(self.yaml_path):
            self.errors.append(f"Config not found: {self.yaml_path}")
            return False
        print(f"  ✓ Config file exists")

        try:
            with open(self.yaml_path, 'r') as f:
                self.data = yaml.safe_load(f)
            print(f"  ✓ YAML parsed")
        except yaml.YAMLError as e:
            self.errors.append(f"YAML error: {e}")
            return False

        missing = [f for f in self.REQUIRED_FIELDS if f not in self.data]
        if missing:
            self.errors.append(f"Missing fields: {', '.join(missing)}")
            return False
        print(f"  ✓ Required fields present")

        if 'package_map' not in self.data:
            self.data['package_map'] = {}

        # Expand all paths
        self.data['model_path']  = self._expand_path(self.data['model_path'])
        self.data['mesh_dir']    = self._expand_path(self.data['mesh_dir'])
        self.data['output_path'] = self._expand_path(self.data['output_path'])
        self.data['package_map'] = {
            k: self._expand_path(v) for k, v in self.data['package_map'].items()
        }
        print(f"  ✓ Paths expanded")

        # Validate paths
        if not os.path.exists(self.data['model_path']):
            self.errors.append(f"Model not found: {self.data['model_path']}")
        else:
            ext = Path(self.data['model_path']).suffix.lower()
            if ext not in ['.urdf', '.xml']:
                self.errors.append(f"Unsupported format: {ext}")
            else:
                print(f"  ✓ Model file exists ({ext})")

        if not os.path.exists(self.data['mesh_dir']):
            self.errors.append(f"Mesh dir not found: {self.data['mesh_dir']}")
        else:
            print(f"  ✓ Mesh dir accessible")

        output_dir = os.path.dirname(self.data['output_path'])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.errors:
            return False

        print(f"\n[PHASE 1] Summary")
        print(f"  Model       : {os.path.basename(self.data['model_path'])}")
        print(f"  Output      : {os.path.basename(self.data['output_path'])}")
        print(f"  Mesh dir    : {os.path.basename(self.data['mesh_dir'])}")
        print(f"  Packages    : {list(self.data['package_map'].keys()) or 'none'}")
        return True

    def print_errors(self):
        for e in self.errors:
            print(f"[ERROR] {e}", file=sys.stderr)

    def get(self, key: str, default=None):
        return self.data.get(key, default)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MODEL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class ModelProcessor:
    """Resolve package:// URIs and save processed URDF"""

    def __init__(self, config: ConfigLoader):
        self.config = config

    def process(self) -> str:
        print(f"\n[PHASE 2] Model Processing")

        model_path = self.config.get('model_path')
        mesh_dir   = self.config.get('mesh_dir')

        # Read model
        print(f"\n  Reading model...")
        with open(model_path, 'r') as f:
            xml = f.read()
        print(f"  ✓ Read {len(xml):,} bytes")

        # Strip XML declaration
        xml = re.sub(r'<\?xml[^?]*\?>', '', xml).strip()
        print(f"  ✓ XML declaration stripped")

        # Resolve package:// URIs
        package_map = self.config.get('package_map', {})
        if package_map:
            print(f"\n  Resolving package:// URIs...")
            resolved_count = 0
            for pkg_name, pkg_path in package_map.items():
                old = f"package://{pkg_name}/"
                new = pkg_path.rstrip("/") + "/"
                count = xml.count(old)
                if count > 0:
                    xml = xml.replace(old, new)
                    print(f"    {pkg_name}: {count} resolved")
                    resolved_count += count

            unresolved = list(set(re.findall(r'package://([^/]+)/', xml)))
            if unresolved:
                print(f"  [WARN] Still unresolved: {unresolved}")
            else:
                print(f"  ✓ All URIs resolved ({resolved_count} total)")
        else:
            print(f"  ✓ No package_map - skipping URI resolution")

        # Save resolved model
        model_stem = Path(model_path).stem
        model_ext  = Path(model_path).suffix.lower()
        resolved_path = os.path.join(mesh_dir, f"{model_stem}_resolved{model_ext}")
        with open(resolved_path, 'w') as f:
            f.write(xml)
        print(f"  ✓ Processed model saved")

        return resolved_path


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2b: ASSET LOCALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class AssetLocalizer:
    """Extract and copy all mesh files to self-contained ./assets/ folder"""

    MESH_EXTENSIONS = {'.obj', '.stl', '.dae', '.ply'}

    def __init__(self, config: ConfigLoader, processed_urdf: str):
        self.config = config
        self.processed_urdf = processed_urdf
        self.assets_dir = None
        self.mesh_map = {}  # old_path → new_path

    def localize(self) -> str:
        """Extract mesh references and copy to assets folder"""
        print(f"\n[PHASE 2b] Asset Localization")

        output_path = self.config.get('output_path')
        output_dir = os.path.dirname(os.path.abspath(output_path))
        self.assets_dir = os.path.join(output_dir, 'assets')

        # Create assets folder
        os.makedirs(self.assets_dir, exist_ok=True)
        print(f"  ✓ Assets folder created: {self.assets_dir}")

        # Step 1: Extract all mesh filenames from URDF
        print(f"\n  Extracting mesh references...")
        meshes = self._extract_mesh_paths()
        print(f"  ✓ Found {len(meshes)} mesh files")

        # Step 2: Copy meshes to assets
        print(f"\n  Copying meshes to assets...")
        copied_count = 0
        for source_path in meshes:
            if self._copy_mesh(source_path):
                copied_count += 1

        print(f"  ✓ Copied {copied_count} / {len(meshes)} meshes")

        # Step 3: Update URDF with new mesh paths
        print(f"\n  Updating URDF with local mesh paths...")
        urdf_updated = self._update_urdf_paths()

        # Save updated URDF
        model_stem = Path(self.config.get('model_path')).stem
        mesh_dir = self.config.get('mesh_dir')
        localized_urdf = os.path.join(mesh_dir, f"{model_stem}_localized.urdf")
        with open(localized_urdf, 'w') as f:
            f.write(urdf_updated)
        print(f"  ✓ Localized URDF saved")

        return localized_urdf

    def _extract_mesh_paths(self) -> Set[str]:
        """Extract all mesh file paths from URDF"""
        with open(self.processed_urdf, 'r') as f:
            urdf_content = f.read()

        # Find all filename attributes
        # Matches: filename="/path/to/mesh.obj" or filename="mesh.obj"
        matches = re.findall(r'filename="([^"]+)"', urdf_content)

        mesh_files = set()
        for match in matches:
            # Only include actual mesh files (not other assets)
            if any(match.lower().endswith(ext) for ext in self.MESH_EXTENSIONS):
                mesh_files.add(match)

        return mesh_files

    def _copy_mesh(self, source_path: str) -> bool:
        """Copy a single mesh file to assets folder"""
        if not os.path.exists(source_path):
            print(f"    [WARN] Mesh not found: {source_path}")
            return False

        # Get basename and copy
        basename = os.path.basename(source_path)
        dest_path = os.path.join(self.assets_dir, basename)

        # Handle filename conflicts (rare, but possible)
        if os.path.exists(dest_path) and dest_path != source_path:
            existing_size = os.path.getsize(dest_path)
            source_size = os.path.getsize(source_path)
            if existing_size == source_size:
                # Same file, skip
                self.mesh_map[source_path] = dest_path
                return True
            else:
                # Different files with same name - add suffix
                stem, ext = os.path.splitext(basename)
                counter = 1
                while True:
                    dest_path = os.path.join(self.assets_dir, f"{stem}_{counter}{ext}")
                    if not os.path.exists(dest_path):
                        break
                    counter += 1

        try:
            shutil.copy2(source_path, dest_path)
            self.mesh_map[source_path] = dest_path
            print(f"    {basename}")
            return True
        except Exception as e:
            print(f"    [ERROR] Cannot copy {basename}: {e}")
            return False

    def _update_urdf_paths(self) -> str:
        """Replace absolute mesh paths with relative ./assets/ paths"""
        with open(self.processed_urdf, 'r') as f:
            urdf_content = f.read()

        # Replace all mesh paths with relative ./assets/ paths
        for old_path, new_path in self.mesh_map.items():
            basename = os.path.basename(new_path)
            relative_path = f"./assets/{basename}"
            urdf_content = urdf_content.replace(old_path, relative_path)

        return urdf_content

    def get_assets_dir(self) -> str:
        """Return absolute path to assets folder"""
        return self.assets_dir


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: SCENE BUILDING
# ══════════════════════════════════════════════════════════════════════════════

class SceneBuilder:
    """Convert URDF → MJCF and add scene elements"""

    def __init__(self, config: ConfigLoader, processed_model_path: str, assets_dir: str):
        self.config = config
        self.processed_model_path = processed_model_path
        self.assets_dir = assets_dir

    def build(self) -> Tuple[str, str]:
        print(f"\n[PHASE 3] Scene Building")

        mesh_dir   = self.config.get('mesh_dir')
        model_stem = Path(self.config.get('model_path')).stem

        # Load with MuJoCo (use absolute assets_dir for internal validation)
        print(f"\n  Loading with MuJoCo...")
        try:
            model = mujoco.MjModel.from_xml_path(self.processed_model_path)
            print(f"  ✓ Model loaded")
        except Exception as e:
            print(f"  [ERROR] MuJoCo load failed: {e}", file=sys.stderr)
            raise

        # Save to temp MJCF
        temp_path = os.path.join(mesh_dir, f"{model_stem}_temp.xml")
        mujoco.mj_saveLastXML(temp_path, model)
        with open(temp_path, 'r') as f:
            xml = f.read()
        print(f"  ✓ MJCF generated")

        # Set meshdir to absolute assets path for internal validation
        abs_assets = os.path.abspath(self.assets_dir)
        xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{abs_assets}"', xml)

        if '<compiler' in xml and 'meshdir=' not in xml:
            xml = xml.replace('<compiler', f'<compiler meshdir="{abs_assets}"')

        # Remove floating geoms from worldbody
        xml = re.sub(r'(<worldbody>)\s*<geom[^>]*mesh="[^"]*"[^>]*/>', r'\1', xml)

        # Add ground + lights
        ground = '    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.7 0.7 0.7 1" condim="3" friction="1 0.005 0.0001"/>'
        lights = '''\
    <light name="sun" pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="fill" pos="2 -2 2" dir="-1 1 -1" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>'''

        xml = re.sub(r'(<worldbody>)', f'\\1\n{lights}\n{ground}', xml)
        print(f"  ✓ Ground + lights added")

        # Save scene XML
        scene_path = os.path.join(mesh_dir, f"{model_stem}_scene.xml")
        with open(scene_path, 'w') as f:
            f.write(xml)
        print(f"  ✓ Scene XML saved (internal)")

        return xml, scene_path


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class MuJoCoValidator:
    """Load and validate final scene with MuJoCo"""

    def __init__(self, scene_path: str):
        self.scene_path = scene_path
        self.model = None
        self.stats = {}

    def validate(self) -> bool:
        print(f"\n[PHASE 4] Validation")
        try:
            self.model = mujoco.MjModel.from_xml_path(self.scene_path)
            self.stats = {
                'nbody': self.model.nbody,
                'njnt' : self.model.njnt,
                'ngeom': self.model.ngeom,
                'nmesh': self.model.nmesh,
                'nu'   : self.model.nu,
            }
            print(f"  ✓ Validated")
            print(f"  Bodies    : {self.stats['nbody']}")
            print(f"  Joints    : {self.stats['njnt']}")
            print(f"  Geoms     : {self.stats['ngeom']}")
            print(f"  Meshes    : {self.stats['nmesh']}")
            print(f"  Actuators : {self.stats['nu']}")
            return True
        except Exception as e:
            print(f"  [ERROR] Validation failed: {e}", file=sys.stderr)
            return False


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: EXPORT
# ══════════════════════════════════════════════════════════════════════════════

class MJCFExporter:
    """Export final MJCF with relative ./assets/ path"""

    def __init__(self, config: ConfigLoader, model):
        self.config = config
        self.model = model

    def export(self) -> bool:
        print(f"\n[PHASE 5] Export")

        output_path = self.config.get('output_path')

        try:
            # Save model to temp
            temp_path = output_path + ".tmp"
            mujoco.mj_saveLastXML(temp_path, self.model)

            with open(temp_path, 'r') as f:
                xml = f.read()
            os.remove(temp_path)

            # Replace meshdir with relative path to ./assets
            xml = re.sub(r'meshdir="[^"]*"', 'meshdir="./assets"', xml)

            # Write final MJCF
            with open(output_path, 'w') as f:
                f.write(xml)

            file_size = os.path.getsize(output_path)
            print(f"  ✓ MJCF exported with relative meshdir: ./assets")
            print(f"    → {output_path}")
            return True
        except Exception as e:
            print(f"  [ERROR] Export failed: {e}", file=sys.stderr)
            return False


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(config: ConfigLoader, validator: MuJoCoValidator):
    print(f"\n[PHASE 6] Summary")
    print(f"  Input   : {config.get('model_path')}")
    print(f"  Output  : {config.get('output_path')}")
    print(f"  Assets  : ./assets/ (self-contained)")
    print(f"  Bodies  : {validator.stats['nbody']}")
    print(f"  Joints  : {validator.stats['njnt']}")
    print(f"  Meshes  : {validator.stats['nmesh']}")
    print(f"  Status  : ✓ SUCCESS")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MCJF_wrap - Convert robot URDF to MuJoCo MJCF",
        epilog="Example: python3 MCJF_warp.py --config ergocub.yaml"
    )
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  MCJF_wrap.py - URDF to MuJoCo MJCF Converter")
    print("=" * 70)

    # Phase 1
    config = ConfigLoader(args.config)
    if not config.load():
        config.print_errors()
        sys.exit(1)

    # Phase 2
    try:
        processor = ModelProcessor(config)
        processed_model = processor.process()
    except Exception:
        sys.exit(1)

    # Phase 2b: Asset Localization
    try:
        localizer = AssetLocalizer(config, processed_model)
        localized_urdf = localizer.localize()
        assets_dir = localizer.get_assets_dir()
    except Exception as e:
        print(f"\n[ERROR] Asset localization failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Phase 3
    try:
        builder = SceneBuilder(config, localized_urdf, assets_dir)
        _, scene_path = builder.build()
    except Exception:
        sys.exit(1)

    # Phase 4
    validator = MuJoCoValidator(scene_path)
    if not validator.validate():
        sys.exit(1)

    # Phase 5
    exporter = MJCFExporter(config, validator.model)
    if not exporter.export():
        sys.exit(1)

    # Phase 6
    print_summary(config, validator)

    print("\n" + "=" * 70)
    print("  ✓ Conversion Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()