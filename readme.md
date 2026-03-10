# MCJF_warp — URDF to MuJoCo MJCF Converter

Converts robot URDF files to MuJoCo MJCF format. Adds ground plane and lighting.

## Requirements

### Install MuJoCo (Linux — minimal steps)
Create and activate a Python venv, install the Python package and system GL deps:

```bash
python3 -m venv ~/mujoco_env
source ~/mujoco_env/bin/activate
pip install --upgrade pip
pip install mujoco pyyaml
```

## Usage

```bash
python3 MCJF_warp.py --config sample.yaml
```

## Config (YAML)

| Field | Required | Description |
|-------|----------|-------------|
| `model_path` | ✓ | Path to robot `.urdf` or `.xml` |
| `output_path` | ✓ | Output `.xml` path |
| `mesh_dir` | ✓ | Directory containing mesh files |
| `package_map` | ✗ | Maps `package://` URIs to filesystem paths |

```yaml
model_path:  "/path/to/robot.urdf"
output_path: "./robot_clean.xml"
mesh_dir:    "/path/to/meshes"

package_map:                          # optional
  my_package: "/path/to/my_package"
```

### Path formats supported

| Format | Example |
|--------|---------|
| Absolute | `/home/user/robot.urdf` |
| Relative | `./robot.urdf` |
| Env var | `${CONDA_PREFIX}/share/robot.urdf` |
| Home dir | `~/robot.urdf` |

## Pipeline

```
Phase 1 → Load & validate YAML config
Phase 2 → Resolve package:// URIs → save resolved URDF
Phase 3 → MuJoCo converts URDF → MJCF, adds ground + lights
Phase 4 → Validate with MuJoCo
Phase 5 → Export final MJCF
Phase 6 → Print summary
```

## Intermediate Files (in mesh_dir)

| File | Description |
|------|-------------|
| `*_resolved.urdf` | URDF after URI resolution |
| `*_temp.xml` | Raw MuJoCo output |
| `*_scene.xml` | MJCF with ground + lights |

## View Output

```bash
python3 -m mujoco.viewer robot_clean.xml
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Config not found` | Check YAML path |
| `Model not found` | Check `model_path` in YAML |
| `Mesh dir not found` | Check `mesh_dir` in YAML |
| `Unresolved packages` | Add missing package to `package_map` |
| `MuJoCo load failed` | Check all meshes exist in `mesh_dir` |

## Supported Formats

| Format | Input | Notes |
|--------|-------|-------|
| URDF | ✓ | Primary format |
| MJCF | ✓ | MuJoCo native |
| SDF | ✗ | Not supported by MuJoCo directly |

## Adding a New Robot

```bash
cp sample.yaml my_robot.yaml
# edit my_robot.yaml
python3 MCJF_warp.py --config my_robot.yaml
```