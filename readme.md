# Creo2Robot: CAD → iDynTree → MuJoCo Pipeline

Automatic conversion of CAD models to physics-enabled MuJoCo simulations via explicit API-to-API synthesis.

## Architecture: The Hourglass

```
Creo CAD Model
    ↓
[Phase 1] iDynTree::Model (In-Memory Graph)
    ↓ (Topology, Joints, Mass, Inertia)
[Phase 2] MuJoCo MjSpec (Explicit Synthesis)
    ↓
MJCF XML → MuJoCo Viewer / Simulator
```

## Prerequisites

### Install Miniforge/Miniconda/Anaconda

If you don't have a conda distribution installed, download one first:
- **Miniforge** (lightweight, recommended): https://github.com/conda-forge/miniforge
- **Miniconda** (official minimal): https://docs.conda.io/projects/miniconda/
- **Anaconda** (full distribution): https://www.anaconda.com/

Verify installation:
```bash
conda --version
```

---

## Setup

### 1. Create Conda Environment with Models

```bash
conda create -n ergocubenv -c conda-forge python=3.11 ergocub-models
conda activate ergocubenv
```

pin the python for stability

This installs **only the URDF models and meshes** (lightweight). You must also install dependencies:

### 2. Install Dependencies

```bash
conda install -c conda-forge lxml numpy scipy pyyaml
conda install -c conda-forge idyntree
conda install -c conda-forge mujoco
```

Or install all in one step:
```bash
conda create -n ergocubenv -c conda-forge python=3.11 ergocub-models idyntree mujoco lxml numpy scipy pyyaml
```

### 3. Verify Installation

```bash
conda run -n ergocubenv python -c "import idyntree.swig; import mujoco; print('OK')"
```

## Usage

### Step 1: Localize Assets (Copy & Rewrite Paths)

```bash
conda run -n ergocubenv python localizer.py --robot robot_name/robot_version
```

**Output:** 
- `outputs/robot_localized.urdf` 
- `outputs/meshes/` (all mesh files)

Example variant:
```bash
conda run -n ergocubenv python localizer.py --robot ergoCub/ergoCubGazeboSN001
```

### Step 2: Synthesize MuJoCo Model

```bash
conda run -n ergocubenv python urdf_to_mjcf_explicit.py \
  --urdf outputs/robot_localized.urdf \
  --output outputs/robot_synthesis.xml
```

**Output:** `outputs/robot_synthesis.xml` (native MuJoCo format)


Example variant:
```bash
conda run -n ergocubenv python urdf_to_mjcf_explicit.py \
  --urdf outputs/ergoCubGazeboSN001_localized.urdf \
  --output outputs/ergoCubGazeboSN001_synthesis.xml
```

### Step 3: Visualize

```bash
conda run -n ergocubenv python -m mujoco.viewer --mjcf outputs/robot_synthesis.xml
```


Example variant:
```bash
conda run -n ergocubenv python -m mujoco.viewer --mjcf outputs/ergoCubGazeboSN001_synthesis.xml
```

---

## Dependencies Matrix

| Package | Source | Purpose | Required For |
|---------|--------|---------|--------------|
| `ergocub-models` | conda-forge | URDF + mesh files (ROS standard layout) | `localizer.py` |
| `idyntree` | conda-forge | Extract topology, joints, mass, inertia from URDF | Both scripts |
| `mujoco` | conda-forge | Create MjSpec, matrix↔quaternion conversion, export MJCF | `urdf_to_mjcf_explicit.py` |
| `numpy` | conda-forge | Matrix/quaternion math operations | `urdf_to_mjcf_explicit.py` |
| `lxml` | conda-forge | XML parsing (ElementTree) | Both scripts |
| `scipy` | conda-forge | Scientific computing utilities | Optional (future use) |
| `pyyaml` | conda-forge | YAML parsing | Optional (future config files) |

**Install all at once:**
```bash
conda create -n ergocubenv -c conda-forge ergocub-models lxml numpy scipy pyyaml idyntree mujoco
```

---

## Quick Start (After Conda Setup)

From the `creo2robot/` directory, run (line-by-line):

```bash
# Activate environment
conda activate ergocubenv

# Step 1: Localize robot models
python localizer.py --robot ergoCub/ergoCubGazeboSN001

# Step 2: Synthesize to MuJoCo
python urdf_to_mjcf_explicit.py \
  --urdf outputs/ergoCubGazeboSN001_localized.urdf \
  --output outputs/ergoCubGazeboSN001_synthesis.xml

# Step 3: Visualize
python -m mujoco.viewer --mjcf outputs/ergoCubGazeboSN001_synthesis.xml
```

**Expected outputs:**
- `outputs/ergoCubGazeboSN001_localized.urdf` — Localized URDF
- `outputs/meshes/` — Local mesh directory
- `outputs/ergoCubGazeboSN001_synthesis.xml` — MuJoCo-native MJCF
