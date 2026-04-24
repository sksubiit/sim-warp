#!/usr/bin/env python3
"""
Explicit API-to-API Conversion (The Correct Architecture)
Simulates extracting data from the CAD plugin by loading into iDynTree::Model,
then programmatically synthesizing the MjSpec completely from scratch.

This script proves the articulated-core path today while organizing the code as
explicit compiler passes that can evolve toward the roadmap architecture.
"""

import sys
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    print("ERROR: mujoco not found. Please install it.")
    sys.exit(1)

try:
    import idyntree.swig as iDynTree
except ImportError as e:
    print(f"ERROR: iDynTree bindings missing: {e}")
    sys.exit(1)


SUPPORTED_MUJOCO_MESH_SUFFIXES = {".stl", ".obj", ".msh"}


@dataclass
class InertialSpec:
    mass: float
    com: list[float]
    fullinertia: list[float]


@dataclass
class GeometrySpec:
    kind: str
    source_file: str
    pos: list[float]
    rpy: list[float]
    scale: list[float]
    backend_file: str | None = None
    backend_status: str = "pending"


@dataclass
class JointSpec:
    name: str
    parent_link: str
    child_link: str
    pos: list[float]
    quat: list[float]
    mj_type: str | None
    axis: list[float] | None
    source_dofs: int

    @property
    def is_fixed(self) -> bool:
        return self.mj_type is None


@dataclass
class LinkNode:
    name: str
    inertia: InertialSpec
    visuals: list[GeometrySpec] = field(default_factory=list)


@dataclass
class AssetResolutionReport:
    total_visuals: int = 0
    exact_matches: int = 0
    substitutions: int = 0
    skipped: int = 0
    skipped_files: list[str] = field(default_factory=list)
    substituted_files: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class RobotModel:
    model_name: str
    root_link_name: str
    floating_base: bool = True
    links: dict[str, LinkNode] = field(default_factory=dict)
    joints: list[JointSpec] = field(default_factory=list)
    children_by_link: dict[str, list[JointSpec]] = field(default_factory=lambda: defaultdict(list))
    source_only_visual_links: set[str] = field(default_factory=set)

    def add_link(self, link: LinkNode) -> None:
        self.links[link.name] = link

    def add_joint(self, joint: JointSpec) -> None:
        self.joints.append(joint)
        self.children_by_link[joint.parent_link].append(joint)


@dataclass
class SemanticMergeReport:
    articulated_links_with_visuals: int
    source_only_visual_links: int
    total_visual_geometries: int


def rpy_to_quat(r, p, y):
    """Convert Roll-Pitch-Yaw to MuJoCo's [w, x, y, z] quaternion."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([r, p, y]), "xyz")
    return list(q)


def mat33_to_quat(H):
    """Convert the rotation block of an iDynTree homogeneous transform to a quaternion."""
    mat = np.zeros(9)
    for r in range(3):
        for c in range(3):
            mat[r * 3 + c] = H.getVal(r, c)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return list(quat)


def derive_model_name(urdf_path: str) -> str:
    stem = Path(urdf_path).stem
    if stem.endswith("_localized"):
        stem = stem[: -len("_localized")]
    return stem


def derive_output_path(urdf_path, output_mjcf_path=None):
    """Derive a robot-specific MJCF path from the localized URDF when not provided."""
    if output_mjcf_path:
        return output_mjcf_path

    urdf_file = Path(urdf_path)
    stem = derive_model_name(urdf_path)
    return str(urdf_file.with_name(f"{stem}_synthesis.xml"))


def build_inertial_spec(link) -> InertialSpec:
    inertia = link.getInertia()
    com = inertia.getCenterOfMass()
    rot = inertia.getRotationalInertiaWrtCenterOfMass()
    return InertialSpec(
        mass=float(inertia.getMass()),
        com=[com.getVal(0), com.getVal(1), com.getVal(2)],
        fullinertia=[
            rot.getVal(0, 0),
            rot.getVal(1, 1),
            rot.getVal(2, 2),
            rot.getVal(0, 1),
            rot.getVal(0, 2),
            rot.getVal(1, 2),
        ],
    )


def extract_joint_spec(model, joint_idx: int) -> JointSpec:
    joint = model.getJoint(joint_idx)
    parent_idx = joint.getFirstAttachedLink()
    child_idx = joint.getSecondAttachedLink()
    parent_link = model.getLinkName(parent_idx)
    child_link = model.getLinkName(child_idx)

    transform = joint.getRestTransform(parent_idx, child_idx)
    H = transform.asHomogeneousTransform()
    pos = [H.getVal(0, 3), H.getVal(1, 3), H.getVal(2, 3)]
    quat = mat33_to_quat(H)

    dofs = joint.getNrOfDOFs()
    mj_type = None
    axis = None

    if dofs == 1:
        rev_joint = joint.asRevoluteJoint()
        if rev_joint:
            mj_type = "hinge"
            axis_obj = rev_joint.getAxis(child_idx)
            axis_dir = axis_obj.getDirection()
            axis = [axis_dir.getVal(0), axis_dir.getVal(1), axis_dir.getVal(2)]
        else:
            prism_joint = joint.asPrismaticJoint()
            if prism_joint:
                mj_type = "slide"
                axis_obj = prism_joint.getAxis(child_idx)
                axis_dir = axis_obj.getDirection()
                axis = [axis_dir.getVal(0), axis_dir.getVal(1), axis_dir.getVal(2)]

    return JointSpec(
        name=model.getJointName(joint_idx),
        parent_link=parent_link,
        child_link=child_link,
        pos=pos,
        quat=quat,
        mj_type=mj_type,
        axis=axis,
        source_dofs=dofs,
    )


def load_articulated_core(urdf_path: str) -> RobotModel:
    """Pass 1: load articulated mechanics through iDynTree."""
    loader = iDynTree.ModelLoader()
    if not loader.loadModelFromFile(urdf_path):
        print("ERROR: Failed to load into iDynTree")
        sys.exit(1)

    model = loader.model()

    children = set()
    for joint_idx in range(model.getNrOfJoints()):
        children.add(model.getJoint(joint_idx).getSecondAttachedLink())
    root_idx = next(i for i in range(model.getNrOfLinks()) if i not in children)

    robot = RobotModel(
        model_name=derive_model_name(urdf_path),
        root_link_name=model.getLinkName(root_idx),
    )

    for link_idx in range(model.getNrOfLinks()):
        link_name = model.getLinkName(link_idx)
        robot.add_link(LinkNode(name=link_name, inertia=build_inertial_spec(model.getLink(link_idx))))

    for joint_idx in range(model.getNrOfJoints()):
        robot.add_joint(extract_joint_spec(model, joint_idx))

    print(f"  ✓ iDynTree Graph ready: {model.getNrOfLinks()} Links, {model.getNrOfJoints()} Joints")
    print(f"  ✓ Root link identified dynamically: '{robot.root_link_name}'")
    return robot


def parse_visual_supplement(urdf_path: str) -> dict[str, list[GeometrySpec]]:
    """Pass 2: parse source visuals from the localized URDF XML."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    visuals: dict[str, list[GeometrySpec]] = {}

    for link in root.findall("link"):
        link_name = link.get("name")
        geometries: list[GeometrySpec] = []
        for vis in link.findall("visual"):
            origin = vis.find("origin")
            geom = vis.find("geometry/mesh")
            if geom is None:
                continue

            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            scale = [1.0, 1.0, 1.0]

            if geom.get("scale"):
                scale = [float(x) for x in geom.get("scale").split()]
            if origin is not None:
                if origin.get("xyz"):
                    xyz = [float(x) for x in origin.get("xyz").split()]
                if origin.get("rpy"):
                    rpy = [float(x) for x in origin.get("rpy").split()]

            geometries.append(
                GeometrySpec(
                    kind="visual",
                    source_file=geom.get("filename"),
                    pos=xyz,
                    rpy=rpy,
                    scale=scale,
                )
            )

        visuals[link_name] = geometries

    return visuals


def merge_semantic_supplement(robot: RobotModel, urdf_path: str) -> SemanticMergeReport:
    """Pass 3: merge source semantic supplements into one in-memory robot model."""
    visuals_by_link = parse_visual_supplement(urdf_path)

    articulated_links_with_visuals = 0
    total_visual_geometries = 0
    source_only_links = 0

    for link_name, visuals in visuals_by_link.items():
        if not visuals:
            continue
        total_visual_geometries += len(visuals)
        if link_name in robot.links:
            robot.links[link_name].visuals.extend(visuals)
            articulated_links_with_visuals += 1
        else:
            robot.source_only_visual_links.add(link_name)
            source_only_links += 1

    return SemanticMergeReport(
        articulated_links_with_visuals=articulated_links_with_visuals,
        source_only_visual_links=source_only_links,
        total_visual_geometries=total_visual_geometries,
    )


def normalize_mesh_file(filename: str) -> str:
    mesh_file = Path(filename).as_posix()
    if mesh_file.startswith("./"):
        return mesh_file[2:]
    return mesh_file


def resolve_backend_assets(robot: RobotModel, urdf_path: str) -> AssetResolutionReport:
    """Pass 4a: resolve source visuals into MuJoCo-compatible backend assets."""
    urdf_dir = Path(urdf_path).resolve().parent
    report = AssetResolutionReport()

    for link in robot.links.values():
        for geom in link.visuals:
            if geom.kind != "visual":
                continue

            report.total_visuals += 1
            source_file = normalize_mesh_file(geom.source_file)
            source_path = Path(source_file)

            if source_path.suffix.lower() in SUPPORTED_MUJOCO_MESH_SUFFIXES:
                geom.backend_file = source_file
                geom.backend_status = "exact"
                report.exact_matches += 1
                continue

            substituted = None
            for suffix in (".stl", ".obj", ".msh"):
                candidate = source_path.with_suffix(suffix)
                if (urdf_dir / candidate).exists():
                    substituted = candidate.as_posix()
                    break

            if substituted is not None:
                geom.backend_file = substituted
                geom.backend_status = "substituted"
                report.substitutions += 1
                report.substituted_files.append((source_file, substituted))
            else:
                geom.backend_file = None
                geom.backend_status = "skipped"
                report.skipped += 1
                report.skipped_files.append(source_file)

    return report


def emit_mjcf(robot: RobotModel, urdf_path: str, output_mjcf_path: str) -> tuple[int, int]:
    """Pass 4b: emit backend-native MuJoCo XML from the merged robot model."""
    spec = mujoco.MjSpec()
    spec.modelname = Path(output_mjcf_path).stem
    spec.compiler.autolimits = True
    spec.compiler.meshdir = str(Path(urdf_path).resolve().parent)

    mesh_assets: dict[tuple[str, tuple[float, float, float]], str] = {}
    bodies_added = 0
    joints_added = 0

    def get_or_create_mesh_name(geom: GeometrySpec) -> str:
        key = (geom.backend_file, tuple(geom.scale))
        mesh_name = mesh_assets.get(key)
        if mesh_name is not None:
            return mesh_name

        mesh_name = f"mesh_{len(mesh_assets)}_{Path(geom.backend_file).stem}"
        mesh = spec.add_mesh()
        mesh.name = mesh_name
        mesh.file = geom.backend_file
        mesh.scale = geom.scale
        mesh_assets[key] = mesh_name
        return mesh_name

    def synthesize_body(parent_mjbody, link_name: str, incoming_joint: JointSpec | None = None):
        nonlocal bodies_added, joints_added

        link = robot.links[link_name]
        body = parent_mjbody.add_body()
        body.name = link.name

        if incoming_joint is None:
            body.pos = [0.0, 0.0, 0.0]
            if robot.floating_base:
                free_joint = body.add_freejoint()
                free_joint.name = "root_freejoint"
        else:
            body.pos = incoming_joint.pos
            body.quat = incoming_joint.quat

            if incoming_joint.mj_type == "hinge":
                joint = body.add_joint()
                joint.name = incoming_joint.name
                joint.type = mujoco.mjtJoint.mjJNT_HINGE
                joint.axis = incoming_joint.axis
                joints_added += 1
            elif incoming_joint.mj_type == "slide":
                joint = body.add_joint()
                joint.name = incoming_joint.name
                joint.type = mujoco.mjtJoint.mjJNT_SLIDE
                joint.axis = incoming_joint.axis
                joints_added += 1

        body.explicitinertial = True
        body.ipos = link.inertia.com
        body.mass = link.inertia.mass
        body.fullinertia = link.inertia.fullinertia
        bodies_added += 1

        for geom in link.visuals:
            if geom.backend_status == "skipped" or geom.backend_file is None:
                continue

            mj_geom = body.add_geom()
            mj_geom.type = mujoco.mjtGeom.mjGEOM_MESH
            mj_geom.meshname = get_or_create_mesh_name(geom)
            mj_geom.pos = geom.pos
            mj_geom.quat = rpy_to_quat(*geom.rpy)

        for child_joint in robot.children_by_link.get(link_name, []):
            synthesize_body(body, child_joint.child_link, incoming_joint=child_joint)

    synthesize_body(spec.worldbody, robot.root_link_name)

    with open(output_mjcf_path, "w") as f:
        f.write(spec.to_xml())

    return bodies_added, joints_added


def report_compiler_state(robot: RobotModel, merge_report: SemanticMergeReport, asset_report: AssetResolutionReport) -> None:
    articulated_visual_links = sum(1 for link in robot.links.values() if link.visuals)
    print("\n[2] Parsing semantic supplement from localized URDF...")
    print(f"  ✓ Attached visual semantics to {articulated_visual_links} articulated links")
    print(f"  ✓ Parsed {merge_report.total_visual_geometries} source visual geometries")
    if merge_report.source_only_visual_links:
        print(f"  ! Found {merge_report.source_only_visual_links} visual links outside the articulated core")

    print("\n[3] Resolving backend assets for MuJoCo...")
    print(f"  ✓ Exact supported assets: {asset_report.exact_matches}")
    print(f"  ✓ Substituted assets: {asset_report.substitutions}")
    print(f"  ! Skipped unsupported assets: {asset_report.skipped}")
    if asset_report.substituted_files:
        for src, dst in asset_report.substituted_files[:5]:
            print(f"    - substitute: {src} -> {dst}")
        if len(asset_report.substituted_files) > 5:
            print(f"    - ... and {len(asset_report.substituted_files) - 5} more substitutions")
    if asset_report.skipped_files:
        for filename in sorted(set(asset_report.skipped_files))[:5]:
            print(f"    - skipped: {filename}")
        if len(set(asset_report.skipped_files)) > 5:
            print(f"    - ... and {len(set(asset_report.skipped_files)) - 5} more skipped assets")


def run_explicit_synthesis(urdf_path: str, output_mjcf_path: str | None):
    print("==================================================")
    print("EXPLICIT API-TO-API SYNTHESIS (iDynTree -> MjSpec)")
    print("==================================================")

    output_mjcf_path = derive_output_path(urdf_path, output_mjcf_path)

    print("\n[1] Loading articulated mechanical core through iDynTree...")
    robot = load_articulated_core(urdf_path)
    merge_report = merge_semantic_supplement(robot, urdf_path)
    asset_report = resolve_backend_assets(robot, urdf_path)
    report_compiler_state(robot, merge_report, asset_report)

    print("\n[4] Emitting explicit MuJoCo MjSpec...")
    bodies_added, joints_added = emit_mjcf(robot, urdf_path, output_mjcf_path)
    print(f"  ✓ Synthesized {bodies_added} bodies and {joints_added} explicit joints")

    print("\n[5] Saved explicit MJCF")
    print(f"  ✓ Output XML: {output_mjcf_path}")
    print("==================================================")
    print("SUCCESS: Explicit API Synthesis Complete")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Input URDF used to initialize iDynTree and recover source semantics")
    parser.add_argument("--output", help="Output MJCF generated by API compilation. Defaults to <robot>_synthesis.xml next to the URDF.")
    args = parser.parse_args()

    run_explicit_synthesis(args.urdf, args.output)
