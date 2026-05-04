#!/usr/bin/env python3
"""
Atomic PoC: load a localized URDF into iDynTree and emit the articulated robot
explicitly to MuJoCo through MjSpec.

This script uses iDynTree as the only robot-model source. It emits:
- articulated body tree
- movable joints represented by iDynTree
- inertial properties carried by iDynTree
- joint limits carried by iDynTree
- joint damping and static friction carried by iDynTree
- additional frames as MuJoCo sites

It does not supplement the model with URDF-side visuals, collisions,
actuators, or sensors.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

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


def rotation_to_quat(rotation) -> list[float]:
    quat = iDynTree.Vector4()
    rotation.getQuaternion(quat)
    return [quat.getVal(i) for i in range(4)]


def transform_to_pose(transform) -> tuple[list[float], list[float]]:
    H = transform.asHomogeneousTransform()
    pos = [H.getVal(0, 3), H.getVal(1, 3), H.getVal(2, 3)]
    quat = rotation_to_quat(transform.getRotation())
    return pos, quat


def derive_model_name(urdf_path: str) -> str:
    stem = Path(urdf_path).stem
    if stem.endswith("_localized"):
        stem = stem[: -len("_localized")]
    return stem


def derive_output_path(urdf_path: str, output_mjcf_path: str | None = None) -> str:
    if output_mjcf_path:
        return output_mjcf_path
    urdf_file = Path(urdf_path)
    return str(urdf_file.with_name(f"{derive_model_name(urdf_path)}_skeleton.xml"))


def load_idyntree_model(urdf_path: str):
    loader = iDynTree.ModelLoader()
    if not loader.loadModelFromFile(urdf_path):
        print("ERROR: Failed to load URDF into iDynTree")
        sys.exit(1)

    model = loader.model()
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    child_links = set()

    for joint_idx in range(model.getNrOfJoints()):
        joint = model.getJoint(joint_idx)
        parent_idx = joint.getFirstAttachedLink()
        child_idx = joint.getSecondAttachedLink()
        children_by_parent[parent_idx].append(joint_idx)
        child_links.add(child_idx)

    root_idx = next(link_idx for link_idx in range(model.getNrOfLinks()) if link_idx not in child_links)

    print(f"  ✓ iDynTree Graph ready: {model.getNrOfLinks()} Links, {model.getNrOfJoints()} Joints")
    print(f"  ✓ Root link identified dynamically: '{model.getLinkName(root_idx)}'")
    return loader, model, root_idx, children_by_parent


def get_inertia_data(link):
    inertia = link.getInertia()
    com = inertia.getCenterOfMass()
    rot = inertia.getRotationalInertiaWrtCenterOfMass()
    return {
        "mass": float(inertia.getMass()),
        "com": [com.getVal(0), com.getVal(1), com.getVal(2)],
        "fullinertia": [
            rot.getVal(0, 0),
            rot.getVal(1, 1),
            rot.getVal(2, 2),
            rot.getVal(0, 1),
            rot.getVal(0, 2),
            rot.getVal(1, 2),
        ],
    }


def get_joint_limits(joint) -> list[float] | None:
    if not joint.hasPosLimits():
        return None

    lower = float(joint.getMinPosLimit(0))
    upper = float(joint.getMaxPosLimit(0))
    return [lower, upper]


def get_joint_scalar(joint, getter_name: str) -> float | None:
    getter = getattr(joint, getter_name, None)
    if getter is None:
        return None

    try:
        return float(getter(0))
    except TypeError:
        return float(getter())


def get_joint_export_data(model, joint_idx: int):
    joint = model.getJoint(joint_idx)
    parent_idx = joint.getFirstAttachedLink()
    child_idx = joint.getSecondAttachedLink()
    transform = joint.getRestTransform(parent_idx, child_idx)

    pos, quat = transform_to_pose(transform)
    dofs = joint.getNrOfDOFs()
    mj_type = None
    axis = None
    limits = None
    damping = None
    frictionloss = None

    if dofs == 1:
        rev_joint = joint.asRevoluteJoint()
        if rev_joint:
            motion = rev_joint.getMotionSubspaceVector(0, 0)
            mj_type = mujoco.mjtJoint.mjJNT_HINGE
            axis = [motion.getVal(3), motion.getVal(4), motion.getVal(5)]
            limits = get_joint_limits(rev_joint)
            damping = get_joint_scalar(rev_joint, "getDamping")
            frictionloss = get_joint_scalar(rev_joint, "getStaticFriction")
        else:
            prism_joint = joint.asPrismaticJoint()
            if prism_joint:
                motion = prism_joint.getMotionSubspaceVector(0, 0)
                mj_type = mujoco.mjtJoint.mjJNT_SLIDE
                axis = [motion.getVal(0), motion.getVal(1), motion.getVal(2)]
                limits = get_joint_limits(prism_joint)
                damping = get_joint_scalar(prism_joint, "getDamping")
                frictionloss = get_joint_scalar(prism_joint, "getStaticFriction")

    return {
        "name": model.getJointName(joint_idx),
        "child_idx": child_idx,
        "pos": pos,
        "quat": quat,
        "mj_type": mj_type,
        "axis": axis,
        "limits": limits,
        "damping": damping,
        "frictionloss": frictionloss,
        "dofs": dofs,
    }


def collect_frame_sites(model):
    sites_by_link: dict[int, list[dict]] = defaultdict(list)

    for frame_idx in range(model.getNrOfFrames()):
        link_idx = model.getFrameLink(frame_idx)
        if link_idx < 0:
            continue

        frame_name = model.getFrameName(frame_idx)

        # Skip the canonical frame that coincides with the link name itself.
        if frame_name == model.getLinkName(link_idx):
            continue

        transform = model.getFrameTransform(frame_idx)
        pos, quat = transform_to_pose(transform)
        sites_by_link[link_idx].append(
            {
                "name": frame_name,
                "pos": pos,
                "quat": quat,
            }
        )

    return sites_by_link


def emit_mjcf(model, root_idx: int, children_by_parent, output_mjcf_path: str):
    spec = mujoco.MjSpec()
    spec.modelname = Path(output_mjcf_path).stem
    spec.compiler.degree = 0

    frame_sites = collect_frame_sites(model)

    bodies_added = 0
    joints_added = 0
    unsupported_joints = 0
    sites_added = 0

    def synthesize_link(parent_mjbody, link_idx: int, incoming_joint: dict | None = None):
        nonlocal bodies_added, joints_added, unsupported_joints, sites_added

        link_name = model.getLinkName(link_idx)
        link = model.getLink(link_idx)
        body = parent_mjbody.add_body()
        body.name = link_name

        if incoming_joint is None:
            body.pos = [0.0, 0.0, 0.0]
            free_joint = body.add_freejoint()
            free_joint.name = "root_freejoint"
        else:
            body.pos = incoming_joint["pos"]
            body.quat = incoming_joint["quat"]

            if incoming_joint["mj_type"] is not None:
                joint = body.add_joint()
                joint.name = incoming_joint["name"]
                joint.type = incoming_joint["mj_type"]
                joint.axis = incoming_joint["axis"]

                if incoming_joint["limits"] is not None:
                    joint.limited = True
                    joint.range = incoming_joint["limits"]

                if incoming_joint["damping"] is not None:
                    joint.damping = incoming_joint["damping"]

                if incoming_joint["frictionloss"] is not None:
                    joint.frictionloss = incoming_joint["frictionloss"]

                joints_added += 1
            elif incoming_joint["dofs"] > 0:
                unsupported_joints += 1

        inertia = get_inertia_data(link)
        body.explicitinertial = True
        body.ipos = inertia["com"]
        body.mass = inertia["mass"]
        body.fullinertia = inertia["fullinertia"]
        bodies_added += 1

        for site_spec in frame_sites.get(link_idx, []):
            site = body.add_site()
            site.name = site_spec["name"]
            site.pos = site_spec["pos"]
            site.quat = site_spec["quat"]
            sites_added += 1

        for joint_idx in children_by_parent.get(link_idx, []):
            child_joint = get_joint_export_data(model, joint_idx)
            synthesize_link(body, child_joint["child_idx"], child_joint)

    synthesize_link(spec.worldbody, root_idx)

    with open(output_mjcf_path, "w") as f:
        f.write(spec.to_xml())

    return {
        "bodies_added": bodies_added,
        "joints_added": joints_added,
        "unsupported_joints": unsupported_joints,
        "sites_added": sites_added,
    }


def run_explicit_synthesis(urdf_path: str, output_mjcf_path: str | None):
    print("==================================================")
    print("ATOMIC iDynTree -> MjSpec SYNTHESIS")
    print("==================================================")

    output_mjcf_path = derive_output_path(urdf_path, output_mjcf_path)

    print("\n[1] Loading articulated model through iDynTree...")
    loader, model, root_idx, children_by_parent = load_idyntree_model(urdf_path)

    print("\n[2] Emitting articulated MuJoCo MjSpec...")
    stats = emit_mjcf(
        model,
        root_idx,
        children_by_parent,
        output_mjcf_path,
    )
    print(f"  ✓ Synthesized {stats['bodies_added']} bodies")
    print(f"  ✓ Synthesized {stats['joints_added']} explicit 1-DOF joints")
    print(f"  ✓ Added {stats['sites_added']} frame sites")
    print(f"  ! Unsupported non-1-DOF joints skipped as MuJoCo joints: {stats['unsupported_joints']}")

    print("\n[3] Saved explicit MJCF")
    print(f"  ✓ Output XML: {output_mjcf_path}")
    print("==================================================")
    print("SUCCESS: Atomic iDynTree API Synthesis Complete")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Localized URDF used only to populate iDynTree")
    parser.add_argument("--output", help="Output MJCF generated through MuJoCo MjSpec")
    args = parser.parse_args()

    run_explicit_synthesis(args.urdf, args.output)
