"""
MCJF_warp.py - Universal URDF/SDF to MuJoCo MJCF Converter

A simple, robust tool for converting robot models to MuJoCo MJCF format.

Complete pipeline in 6 phases:
  Phase 1: Config Loading       - Load and validate YAML configuration
  Phase 2: Model Processing     - Clean and resolve package URIs
  Phase 3: Scene Building       - Convert model to MJCF via MuJoCo
  Phase 4: Validation           - Load and validate with MuJoCo
  Phase 5: Export               - Save canonical MJCF file
  Phase 6: Summary              - Print conversion results

Usage:
  python3 MCJF_warp.py --config ergocub.yaml
  python3 MCJF_warp.py --config my_robot.yaml

Requirements:
  - PyYAML: pip install pyyaml
  - MuJoCo: pip install mujoco
"""

import os
import sys
import re
import argparse
import yaml
from pathlib import Path
from typing import Tuple

try:
    import mujoco
except ImportError:
    print("[ERROR] MuJoCo not installed. Run: pip install mujoco")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

class ConfigLoader:
    """Load and validate YAML configuration file"""

    REQUIRED_FIELDS = ['model_path', 'output_path', 'mesh_dir', 'package_map']

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.data = {}
        self.errors = []

    def _expand_path(self, path_str: str) -> str:
        """Expand paths with environment variables and relative paths"""
        expanded = os.path.expandvars(path_str)
        expanded = os.path.expanduser(expanded)
        
        if os.path.isabs(expanded):
            return expanded
        
        config_dir = os.path.dirname(os.path.abspath(self.yaml_path))
        return os.path.join(config_dir, expanded)

    def load(self) -> bool:
        """Load, validate, and expand configuration"""
        print(f"\n[PHASE 1] Config Loading")
        print(f"  config file: {self.yaml_path}")

        if not os.path.exists(self.yaml_path):
            self.errors.append(f"Config file not found: {self.yaml_path}")
            return False
        print(f"  ✓ Config file exists")

        try:
            with open(self.yaml_path, 'r') as f:
                self.data = yaml.safe_load(f)
            print(f"  ✓ YAML parsed")
        except yaml.YAMLError as e:
            self.errors.append(f"YAML syntax error: {e}")
            return False

        missing = [f for f in self.REQUIRED_FIELDS if f not in self.data]
        if missing:
            self.errors.append(f"Missing required fields: {', '.join(missing)}")
            return False
        print(f"  ✓ All required fields present")

        print(f"\n[PHASE 1] Expanding Paths")
        self.data['model_path'] = self._expand_path(self.data['model_path'])
        self.data['mesh_dir'] = self._expand_path(self.data['mesh_dir'])
        self.data['output_path'] = self._expand_path(self.data['output_path'])
        
        expanded_packages = {}
        for pkg_name, pkg_path in self.data['package_map'].items():
            expanded_packages[pkg_name] = self._expand_path(pkg_path)
        self.data['package_map'] = expanded_packages
        print(f"  ✓ Paths expanded")

        print(f"\n[PHASE 1] Validating File Paths")
        if not os.path.exists(self.data['model_path']):
            self.errors.append(f"Model file not found: {self.data['model_path']}")
        else:
            model_ext = Path(self.data['model_path']).suffix.lower()
            print(f"  ✓ Model file exists ({model_ext})")
        
        if not os.path.exists(self.data['mesh_dir']):
            self.errors.append(f"Mesh dir not found: {self.data['mesh_dir']}")
        else:
            print(f"  ✓ Mesh dir accessible")

        output_dir = os.path.dirname(self.data['output_path'])
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"  ✓ Output dir created")
            except Exception as e:
                self.errors.append(f"Cannot create output dir: {e}")

        if self.errors:
            return False

        self._print_summary()
        return True

    def _print_summary(self):
        """Print loaded configuration"""
        print(f"\n[PHASE 1] Configuration Summary")
        model_name = os.path.basename(self.data['model_path'])
        print(f"  Model file  : {model_name}")
        print(f"  Output MJCF : {os.path.basename(self.data['output_path'])}")
        print(f"  Mesh dir    : {os.path.basename(self.data['mesh_dir'])}")
        print(f"  Packages    : {list(self.data['package_map'].keys())}")

    def print_errors(self):
        """Print all validation errors"""
        for error in self.errors:
            print(f"[ERROR] {error}", file=sys.stderr)

    def get(self, key: str, default=None):
        """Get config value"""
        return self.data.get(key, default)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MODEL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class ModelProcessor:
    """Process URDF/SDF: strip declaration, resolve package URIs"""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.raw_xml = ""
        self.processed_xml = ""

    def process(self) -> str:
        """Process model file and return path"""
        print(f"\n[PHASE 2] Model Processing")
        
        # Step 1: Read model file
        print(f"\n  Reading model file...")
        model_path = self.config.get('model_path')
        try:
            with open(model_path, 'r') as f:
                self.raw_xml = f.read()
            print(f"  ✓ Read {len(self.raw_xml):,} bytes")
        except Exception as e:
            print(f"  [ERROR] Cannot read model: {e}", file=sys.stderr)
            raise
        
        self.processed_xml = self.raw_xml

        # Step 2: Strip XML declaration
        print(f"\n  Stripping XML declaration...")
        self.processed_xml = re.sub(r'<\?xml[^?]*\?>', '', self.processed_xml).strip()
        print(f"  ✓ Declaration removed")

        # Step 3: Resolve package:// URIs
        print(f"\n  Resolving package:// URIs...")
        package_map = self.config.get('package_map', {})
        resolved_count = 0
        
        for pkg_name, pkg_path in package_map.items():
            old = f"package://{pkg_name}/"
            new = pkg_path.rstrip("/") + "/"
            count = self.processed_xml.count(old)
            if count > 0:
                self.processed_xml = self.processed_xml.replace(old, new)
                print(f"    {pkg_name}: {count} URIs resolved")
                resolved_count += count

        unresolved = list(set(re.findall(r'package://([^/]+)/', self.processed_xml)))
        if unresolved:
            print(f"  [WARN] Unresolved packages: {unresolved}")
        else:
            print(f"  ✓ All URIs resolved ({resolved_count} total)")

        # Step 4: Write processed model
        print(f"\n  Saving processed model...")
        mesh_dir = self.config.get('mesh_dir')
        model_stem = Path(model_path).stem
        model_ext = Path(model_path).suffix.lower()
        processed_path = os.path.join(mesh_dir, f"{model_stem}_resolved{model_ext}")
        
        try:
            with open(processed_path, 'w') as f:
                f.write(self.processed_xml)
            print(f"  ✓ Processed model saved")
        except Exception as e:
            print(f"  [ERROR] Cannot write processed model: {e}", file=sys.stderr)
            raise

        return processed_path


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: SCENE BUILDING
# ══════════════════════════════════════════════════════════════════════════════

class SceneBuilder:
    """Convert URDF → MJCF and add scene elements"""

    def __init__(self, config: ConfigLoader, processed_model_path: str):
        self.config = config
        self.processed_model_path = processed_model_path

    def build(self) -> Tuple[str, str]:
        print(f"\n[PHASE 3] Scene Building")

        mesh_dir   = self.config.get('mesh_dir')
        model_stem = Path(self.config.get('model_path')).stem

        # Load with MuJoCo
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

        # ── INTERNAL VALIDATION USE ABSOLUTE PATH ──
        # During Phase 3 and 4, we use absolute meshdir so MuJoCo doesn't get lost
        abs_mesh_dir = os.path.abspath(mesh_dir)
        xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{abs_mesh_dir}"', xml)
        
        if '<compiler' in xml and 'meshdir=' not in xml:
            xml = xml.replace('<compiler', f'<compiler meshdir="{abs_mesh_dir}"')

        # Remove floating geoms from worldbody
        xml = re.sub(r'(<worldbody>)\s*<geom[^>]*mesh="[^"]*"[^>]*/>', r'\1', xml)

        # Add ground + lights
        ground = '    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.7 0.7 0.7 1" condim="3" friction="1 0.005 0.0001"/>'
        lights = '''\
    <light name="sun" pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" castshadow="true"/>
    <light name="fill" pos="2 -2 2" dir="-1 1 -1" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>'''

        xml = re.sub(r'(<worldbody>)', f'\\1\n{lights}\n{ground}', xml)
        print(f"  ✓ Ground + lights added")

        # Save scene XML for Phase 4 (Validation)
        scene_path = os.path.join(mesh_dir, f"{model_stem}_scene.xml")
        with open(scene_path, 'w') as f:
            f.write(xml)
        print(f"  ✓ Scene XML saved (Internal)")

        return xml, scene_path


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class MuJoCoValidator:
    """Load and validate model with MuJoCo"""

    def __init__(self, scene_path: str):
        self.scene_path = scene_path
        self.model = None
        self.stats = {}

    def validate(self) -> bool:
        """Load scene and validate"""
        print(f"\n[PHASE 4] Validation (MuJoCo Load Test)")
        
        try:
            self.model = mujoco.MjModel.from_xml_path(self.scene_path)
            print(f"  ✓ Model loaded and validated")
            
            self.stats = {
                'nbody': self.model.nbody,
                'njnt': self.model.njnt,
                'ngeom': self.model.ngeom,
                'nmesh': self.model.nmesh,
                'nu': self.model.nu,
            }
            
            self._print_stats()
            return True
        except Exception as e:
            print(f"  [ERROR] MuJoCo validation failed: {e}", file=sys.stderr)
            return False

    def _print_stats(self):
        """Print robot statistics"""
        print(f"\n[PHASE 4] Robot Statistics")
        print(f"  Bodies    : {self.stats['nbody']}")
        print(f"  Joints    : {self.stats['njnt']}")
        print(f"  Geoms     : {self.stats['ngeom']}")
        print(f"  Meshes    : {self.stats['nmesh']}")
        print(f"  Actuators : {self.stats['nu']}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: EXPORT
# ══════════════════════════════════════════════════════════════════════════════

class MJCFExporter:
    """Export final MJCF with clean relative paths"""

    def __init__(self, config: ConfigLoader, model):
        self.config = config
        self.model = model

    def export(self) -> bool:
        print(f"\n[PHASE 5] Export (MJCF)")
        
        output_path = self.config.get('output_path')
        mesh_dir = self.config.get('mesh_dir')
        
        try:
            # Save the model to memory first
            temp_path = output_path + ".tmp"
            mujoco.mj_saveLastXML(temp_path, self.model)
            
            with open(temp_path, 'r') as f:
                xml = f.read()
            os.remove(temp_path)

            # ── FINAL EXPORT USE RELATIVE PATH ──
            output_dir = os.path.dirname(os.path.abspath(output_path))
            try:
                rel_mesh_dir = os.path.relpath(os.path.abspath(mesh_dir), output_dir)
            except ValueError:
                rel_mesh_dir = mesh_dir

            xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{rel_mesh_dir}"', xml)
            
            with open(output_path, 'w') as f:
                f.write(xml)

            file_size = os.path.getsize(output_path)
            print(f"  ✓ MJCF exported with relative meshdir: {rel_mesh_dir}")
            print(f"    → {output_path}")
            return True
        except Exception as e:
            print(f"  [ERROR] Export failed: {e}", file=sys.stderr)
            return False


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(config: ConfigLoader, validator: MuJoCoValidator):
    """Print conversion summary"""
    print(f"\n[PHASE 6] Summary")
    print(f"\n  Input:")
    print(f"    Model: {config.get('model_path')}")
    print(f"\n  Output:")
    print(f"    MJCF: {config.get('output_path')}")
    print(f"\n  Robot Model:")
    print(f"    Bodies    : {validator.stats['nbody']}")
    print(f"    Joints    : {validator.stats['njnt']}")
    print(f"    Geoms     : {validator.stats['ngeom']}")
    print(f"    Meshes    : {validator.stats['nmesh']}")
    print(f"    Actuators : {validator.stats['nu']}")
    print(f"\n  Status    : ✓ SUCCESS")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main pipeline: execute all 6 phases"""

    parser = argparse.ArgumentParser(
        description="MCJF_wrap - Convert robot models (URDF/SDF) to MuJoCo MJCF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 MCJF_warp.py --config ergocub.yaml
  python3 MCJF_warp.py --config my_robot.yaml

Supports: URDF, SDF, MJCF input files
See README.md for configuration instructions.
        """
    )
    parser.add_argument('--config', required=True, help='Path to robot YAML config')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  MCJF_wrap.py - URDF/SDF to MuJoCo MJCF Converter")
    print("=" * 70)

    # ── Phase 1: Config Loading ────────────────────────────────────────────
    config = ConfigLoader(args.config)
    if not config.load():
        config.print_errors()
        sys.exit(1)

    # ── Phase 2: Model Processing ──────────────────────────────────────────
    try:
        processor = ModelProcessor(config)
        processed_model = processor.process()
    except Exception as e:
        sys.exit(1)

    # ── Phase 3: Scene Building ────────────────────────────────────────────
    try:
        builder = SceneBuilder(config, processed_model)
        scene_xml, scene_path = builder.build()
    except Exception as e:
        sys.exit(1)

    # ── Phase 4: Validation ────────────────────────────────────────────────
    validator = MuJoCoValidator(scene_path)
    if not validator.validate():
        sys.exit(1)

    # ── Phase 5: Export ────────────────────────────────────────────────────
    exporter = MJCFExporter(config, validator.model)
    if not exporter.export():
        sys.exit(1)

    # ── Phase 6: Summary ───────────────────────────────────────────────────
    print_summary(config, validator)

    print("\n" + "=" * 70)
    print("  ✓ Conversion Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()