#!/usr/bin/env python3
"""
Atomic PoC: load a localized URDF into iDynTree and emit the robot explicitly
to MuJoCo through MjSpec.

The scope is intentionally narrow:
- iDynTree is the only robot-model source
- MjSpec is only the MuJoCo emitter
- no URDF-side supplementation
- no actuator or control-layer inference
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
except ImportError as exc:
    print(f"ERROR: iDynTree bindings missing: {exc}")
    sys.exit(1)


# Common transforms and paths
def rotation_to_quat(rotation):
    quat = iDynTree.Vector4()
    rotation.getQuaternion(quat)
    return [quat.getVal(i) for i in range(4)]


def transform_to_pose(transform):
    matrix = transform.asHomogeneousTransform()
    pos = [matrix.getVal(0, 3), matrix.getVal(1, 3), matrix.getVal(2, 3)]
    quat = rotation_to_quat(transform.getRotation())
    return pos, quat


def vector3_to_list(vec):
    return [float(vec.getVal(i)) for i in range(3)]


def container_size(container):
    try:
        return int(container.size())
    except AttributeError:
        return len(container)


def pose_key(pos, quat, digits=9):
    return tuple(round(value, digits) for value in (*pos, *quat))


def derive_model_name(urdf_path):
    return Path(urdf_path).stem.removesuffix("_localized")


def derive_output_path(urdf_path, output_mjcf_path):
    if output_mjcf_path:
        return output_mjcf_path
    urdf_file = Path(urdf_path)
    return str(urdf_file.with_name(f"{derive_model_name(urdf_path)}_synthesis.xml"))


# iDynTree model loading
def load_idyntree_model(urdf_path):
    loader = iDynTree.ModelLoader()
    if not loader.loadModelFromFile(urdf_path):
        print("ERROR: Failed to load URDF into iDynTree")
        sys.exit(1)

    model = loader.model()
    children_by_parent = defaultdict(list)
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


# Joint extraction
def get_joint_limits(joint):
    if not joint.hasPosLimits():
        return None
    return [float(joint.getMinPosLimit(0)), float(joint.getMaxPosLimit(0))]


def get_joint_scalar(joint, getter_name):
    getter = getattr(joint, getter_name, None)
    if getter is None:
        return None
    try:
        return float(getter(0))
    except TypeError:
        return float(getter())


def get_joint_export_data(model, joint_idx):
    joint = model.getJoint(joint_idx)
    parent_idx = joint.getFirstAttachedLink()
    child_idx = joint.getSecondAttachedLink()
    pos, quat = transform_to_pose(joint.getRestTransform(parent_idx, child_idx))

    joint_spec = {
        "name": model.getJointName(joint_idx),
        "child_idx": child_idx,
        "pos": pos,
        "quat": quat,
        "dofs": joint.getNrOfDOFs(),
        "mj_type": None,
        "axis": None,
        "limits": None,
        "damping": None,
        "frictionloss": None,
    }

    if joint_spec["dofs"] != 1:
        return joint_spec

    revolute = joint.asRevoluteJoint()
    if revolute:
        motion = revolute.getMotionSubspaceVector(0, child_idx, parent_idx)
        joint_spec["mj_type"] = mujoco.mjtJoint.mjJNT_HINGE
        joint_spec["axis"] = [float(motion.getVal(i)) for i in range(3, 6)]
        joint_spec["limits"] = get_joint_limits(revolute)
        joint_spec["damping"] = get_joint_scalar(revolute, "getDamping")
        joint_spec["frictionloss"] = get_joint_scalar(revolute, "getStaticFriction")
        return joint_spec

    prismatic = joint.asPrismaticJoint()
    if prismatic:
        motion = prismatic.getMotionSubspaceVector(0, child_idx, parent_idx)
        joint_spec["mj_type"] = mujoco.mjtJoint.mjJNT_SLIDE
        joint_spec["axis"] = [float(motion.getVal(i)) for i in range(3)]
        joint_spec["limits"] = get_joint_limits(prismatic)
        joint_spec["damping"] = get_joint_scalar(prismatic, "getDamping")
        joint_spec["frictionloss"] = get_joint_scalar(prismatic, "getStaticFriction")

    return joint_spec


def add_mjcf_joint(body, joint_spec, stats):
    if joint_spec is None:
        body.pos = [0.0, 0.0, 0.0]
        body.add_freejoint().name = "root_freejoint"
        return

    body.pos = joint_spec["pos"]
    body.quat = joint_spec["quat"]

    if joint_spec["mj_type"] is None:
        if joint_spec["dofs"] > 0:
            stats["unsupported_joints"] += 1
        return

    joint = body.add_joint()
    joint.name = joint_spec["name"]
    joint.type = joint_spec["mj_type"]
    joint.axis = joint_spec["axis"]

    if joint_spec["limits"] is not None:
        joint.limited = True
        joint.range = joint_spec["limits"]
    if joint_spec["damping"] is not None:
        joint.damping = joint_spec["damping"]
    if joint_spec["frictionloss"] is not None:
        joint.frictionloss = joint_spec["frictionloss"]

    stats["joints_added"] += 1


# Link inertials
def get_inertia_data(link):
    inertia = link.getInertia()
    com = inertia.getCenterOfMass()
    rot = inertia.getRotationalInertiaWrtCenterOfMass()
    return {
        "mass": float(inertia.getMass()),
        "com": vector3_to_list(com),
        "fullinertia": [
            rot.getVal(0, 0),
            rot.getVal(1, 1),
            rot.getVal(2, 2),
            rot.getVal(0, 1),
            rot.getVal(0, 2),
            rot.getVal(1, 2),
        ],
    }


def add_link_inertial(body, link):
    inertia = get_inertia_data(link)
    body.explicitinertial = True
    body.ipos = inertia["com"]
    body.mass = inertia["mass"]
    body.fullinertia = inertia["fullinertia"]


# Sites and sensors
def collect_frame_sites(model):
    sites_by_link = defaultdict(list)
    site_lookup = defaultdict(dict)

    for frame_idx in range(model.getNrOfFrames()):
        link_idx = model.getFrameLink(frame_idx)
        if link_idx < 0:
            continue

        frame_name = model.getFrameName(frame_idx)
        if frame_name == model.getLinkName(link_idx):
            continue

        pos, quat = transform_to_pose(model.getFrameTransform(frame_idx))
        site_spec = {"name": frame_name, "pos": pos, "quat": quat}
        sites_by_link[link_idx].append(site_spec)
        site_lookup[link_idx][pose_key(pos, quat)] = frame_name

    return sites_by_link, site_lookup


def collect_sensor_sites(loader):
    sensors_by_link = defaultdict(list)
    sensors = loader.sensors()

    for sensor_idx in range(sensors.getNrOfSensors(iDynTree.ACCELEROMETER)):
        sensor = sensors.getAccelerometerSensor(sensor_idx)
        pos, quat = transform_to_pose(sensor.getLinkSensorTransform())
        sensors_by_link[sensor.getParentLinkIndex()].append(
            {
                "site_name": f"{sensor.getName()}_site",
                "pos": pos,
                "quat": quat,
                "sensors": [{"name": sensor.getName(), "type": mujoco.mjtSensor.mjSENS_ACCELEROMETER}],
            }
        )

    for sensor_idx in range(sensors.getNrOfSensors(iDynTree.SIX_AXIS_FORCE_TORQUE)):
        sensor = sensors.getSixAxisForceTorqueSensor(sensor_idx)
        link_idx = sensor.getAppliedWrenchLink()
        transform = iDynTree.Transform()
        if not sensor.getLinkSensorTransform(link_idx, transform):
            continue

        pos, quat = transform_to_pose(transform)
        sensors_by_link[link_idx].append(
            {
                "site_name": f"{sensor.getName()}_site",
                "pos": pos,
                "quat": quat,
                "sensors": [
                    {"name": f"{sensor.getName()}_force", "type": mujoco.mjtSensor.mjSENS_FORCE},
                    {"name": f"{sensor.getName()}_torque", "type": mujoco.mjtSensor.mjSENS_TORQUE},
                ],
            }
        )

    return sensors_by_link


def add_frame_sites(body, site_specs, stats):
    for site_spec in site_specs:
        site = body.add_site()
        site.name = site_spec["name"]
        site.pos = site_spec["pos"]
        site.quat = site_spec["quat"]
        stats["sites_added"] += 1


def ensure_sensor_site(body, link_idx, site_spec, site_lookup):
    key = pose_key(site_spec["pos"], site_spec["quat"])
    existing_name = site_lookup[link_idx].get(key)
    if existing_name is not None:
        return existing_name, False

    site = body.add_site()
    site.name = site_spec["site_name"]
    site.pos = site_spec["pos"]
    site.quat = site_spec["quat"]
    site_lookup[link_idx][key] = site_spec["site_name"]
    return site_spec["site_name"], True


def add_sensor_sites(spec, body, link_idx, sensor_specs, site_lookup, stats):
    for site_spec in sensor_specs:
        site_name, created = ensure_sensor_site(body, link_idx, site_spec, site_lookup)
        if created:
            stats["sites_added"] += 1

        for sensor_def in site_spec["sensors"]:
            sensor = spec.add_sensor()
            sensor.name = sensor_def["name"]
            sensor.type = sensor_def["type"]
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = site_name
            stats["sensors_added"] += 1


# Visual and collision shapes
def get_shape_spec(shape):
    pos, quat = transform_to_pose(shape.getLink_H_geometry())

    if shape.isExternalMesh():
        mesh = shape.asExternalMesh()
        mesh_file = str(mesh.getFileLocationOnLocalFileSystem())
        if not mesh_file:
            return None
        return {
            "geom_type": "mesh",
            "mesh_file": mesh_file,
            "mesh_scale": vector3_to_list(mesh.getScale()),
            "pos": pos,
            "quat": quat,
        }

    if shape.isSphere():
        sphere = shape.asSphere()
        return {"geom_type": "sphere", "size": [float(sphere.getRadius())], "pos": pos, "quat": quat}

    if shape.isBox():
        box = shape.asBox()
        return {
            "geom_type": "box",
            "size": [0.5 * float(box.getX()), 0.5 * float(box.getY()), 0.5 * float(box.getZ())],
            "pos": pos,
            "quat": quat,
        }

    if shape.isCylinder():
        cylinder = shape.asCylinder()
        return {
            "geom_type": "cylinder",
            "size": [float(cylinder.getRadius()), 0.5 * float(cylinder.getLength())],
            "pos": pos,
            "quat": quat,
        }

    return None


def collect_link_shapes(model, link_shape_sets):
    shapes_by_link = defaultdict(list)

    for link_idx in range(model.getNrOfLinks()):
        link_shapes = link_shape_sets[link_idx]
        for shape_idx in range(container_size(link_shapes)):
            shape_spec = get_shape_spec(link_shapes[shape_idx])
            if shape_spec is not None:
                shapes_by_link[link_idx].append(shape_spec)

    return shapes_by_link


def add_shape_geom(body, spec, mesh_assets, shape_spec, collision):
    geom = body.add_geom()

    if shape_spec["geom_type"] == "mesh":
        mesh_file = shape_spec["mesh_file"]
        mesh_name = mesh_assets.get(mesh_file)
        if mesh_name is None:
            mesh_name = f"mesh_{len(mesh_assets)}"
            mesh = spec.add_mesh()
            mesh.name = mesh_name
            mesh.file = mesh_file
            mesh.scale = shape_spec["mesh_scale"]
            mesh_assets[mesh_file] = mesh_name
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh_name
    else:
        geom.type = getattr(mujoco.mjtGeom, f"mjGEOM_{shape_spec['geom_type'].upper()}")
        geom.size = shape_spec["size"]

    geom.pos = shape_spec["pos"]
    geom.quat = shape_spec["quat"]

    if collision:
        geom.contype = 1
        geom.conaffinity = 1
        geom.group = 3
        geom.rgba = [0.8, 0.2, 0.2, 0.35]
    else:
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 1


def add_link_shapes(body, spec, mesh_assets, shape_specs, collision, stats):
    for shape_spec in shape_specs:
        add_shape_geom(body, spec, mesh_assets, shape_spec, collision)
        if collision:
            stats["collision_geoms_added"] += 1
        else:
            stats["visual_geoms_added"] += 1


# Final MJCF emission
def emit_mjcf(loader, model, root_idx, children_by_parent, output_mjcf_path):
    output_path = Path(output_mjcf_path).resolve()
    spec = mujoco.MjSpec()
    spec.modelname = output_path.stem
    spec.compiler.degree = 0
    spec.compiler.meshdir = str(output_path.parent)

    frame_sites, site_lookup = collect_frame_sites(model)
    visual_shapes = collect_link_shapes(model, model.visualSolidShapes().getLinkSolidShapes())
    collision_shapes = collect_link_shapes(model, model.collisionSolidShapes().getLinkSolidShapes())
    sensor_sites = collect_sensor_sites(loader)

    mesh_assets = {}
    stats = {
        "bodies_added": 0,
        "joints_added": 0,
        "unsupported_joints": 0,
        "sites_added": 0,
        "visual_geoms_added": 0,
        "collision_geoms_added": 0,
        "mesh_assets_added": 0,
        "sensors_added": 0,
    }

    def synthesize_link(parent_body, link_idx, incoming_joint=None):
        link = model.getLink(link_idx)
        body = parent_body.add_body()
        body.name = model.getLinkName(link_idx)

        add_mjcf_joint(body, incoming_joint, stats)
        add_link_inertial(body, link)
        add_frame_sites(body, frame_sites.get(link_idx, []), stats)
        add_sensor_sites(spec, body, link_idx, sensor_sites.get(link_idx, []), site_lookup, stats)
        add_link_shapes(body, spec, mesh_assets, visual_shapes.get(link_idx, []), False, stats)
        add_link_shapes(body, spec, mesh_assets, collision_shapes.get(link_idx, []), True, stats)
        stats["bodies_added"] += 1

        for joint_idx in children_by_parent.get(link_idx, []):
            child_joint = get_joint_export_data(model, joint_idx)
            synthesize_link(body, child_joint["child_idx"], child_joint)

    synthesize_link(spec.worldbody, root_idx)
    output_path.write_text(spec.to_xml(), encoding="utf-8", newline="\n")
    stats["mesh_assets_added"] = len(mesh_assets)
    return stats


def run_explicit_synthesis(urdf_path, output_mjcf_path=None):
    print("==================================================")
    print("ATOMIC iDynTree -> MjSpec SYNTHESIS")
    print("==================================================")

    output_mjcf_path = derive_output_path(urdf_path, output_mjcf_path)

    print("\n[1] Loading articulated model through iDynTree...")
    loader, model, root_idx, children_by_parent = load_idyntree_model(urdf_path)

    print("\n[2] Emitting articulated MuJoCo MjSpec...")
    stats = emit_mjcf(loader, model, root_idx, children_by_parent, output_mjcf_path)
    print(f"  ✓ Synthesized {stats['bodies_added']} bodies")
    print(f"  ✓ Synthesized {stats['joints_added']} explicit 1-DOF joints")
    print(f"  ✓ Added {stats['sites_added']} frame and sensor sites")
    print(f"  ✓ Added {stats['visual_geoms_added']} visual geoms")
    print(f"  ✓ Added {stats['collision_geoms_added']} collision geoms")
    print(f"  ✓ Added {stats['mesh_assets_added']} mesh assets")
    print(f"  ✓ Added {stats['sensors_added']} sensors")
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
