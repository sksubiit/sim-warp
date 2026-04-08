#!/usr/bin/env python3
"""
Explicit API-to-API Conversion (The Correct Architecture)
Simulates extracting data from the CAD plugin by loading into iDynTree::Model,
then programmatically synthesizing the MjSpec completely from scratch.

This script mathematically proves Phase 2 of the Hourglass Architecture.
"""

import sys
import os
import argparse
import math
import xml.etree.ElementTree as ET
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


def rpy_to_quat(r, p, y):
    """Convert Roll-Pitch-Yaw to MuJoCo's [w, x, y, z] Quaternion using native math."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([r, p, y]), 'xyz')
    return list(q)


def parse_visuals(urdf_path):
    """
    Since iDynTree is a pure physics engine and intentionally drops visual geometries,
    we parse the URDF as a generic dictionary just to get the visual assets.
    Returns: { 'link_name': [ {filename, xyz, rpy}, ... ] }
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    visuals = {}
    
    for link in root.findall("link"):
        name = link.get("name")
        visuals[name] = []
        for vis in link.findall("visual"):
            origin = vis.find("origin")
            geom = vis.find("geometry/mesh")
            
            if geom is not None:
                filename = geom.get("filename")
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
                        
                visuals[name].append({"filename": filename, "xyz": xyz, "rpy": rpy, "scale": scale})
    return visuals


def run_explicit_synthesis(urdf_path, output_mjcf_path):
    print("==================================================")
    print("EXPLICIT API-TO-API SYNTHESIS (iDynTree -> MjSpec)")
    print("==================================================")
    
    # 1. Initialize The Source (iDynTree)
    print("\n[1] Loading Model into iDynTree (Emulating CAD Extractor)...")
    loader = iDynTree.ModelLoader()
    if not loader.loadModelFromFile(urdf_path):
        print("ERROR: Failed to load into iDynTree")
        sys.exit(1)
    
    model = loader.model()
    print(f"  ✓ iDynTree Graph ready: {model.getNrOfLinks()} Links, {model.getNrOfJoints()} Joints")
    
    # Pre-parse visuals from URDF to attach later
    visuals = parse_visuals(urdf_path)
    
    # 2. Initialize The Target (MuJoCo MjSpec)
    print("\n[2] Initializing explicit MjSpec compiler...")
    spec = mujoco.MjSpec()
    spec.modelname = "ergocub_synthesized"
    spec.compiler.autolimits = True
    spec.compiler.meshdir = os.path.join(os.path.dirname(os.path.abspath(urdf_path)), "meshes")
    
    # Track added meshes to avoid duplicates
    added_meshes = set()

    # 3. Analyze Graph Topology (Find parent-child relationships)
    children_map = {}
    for i in range(model.getNrOfJoints()):
        joint = model.getJoint(i)
        parent_idx = joint.getFirstAttachedLink()
        child_idx = joint.getSecondAttachedLink()
        if parent_idx not in children_map:
            children_map[parent_idx] = []
        children_map[parent_idx].append((child_idx, i))

    # Identify Root Link (Link with no parent joint)
    all_children = set(c for p in children_map for c, _ in children_map.get(p, []))
    root_idx = next(i for i in range(model.getNrOfLinks()) if i not in all_children)
    print(f"  ✓ Root link identified dynamically: '{model.getLinkName(root_idx)}'")

    # 4. Graph Traversal Synthesis
    print("\n[3] Synthesizing MuJoCo Tree explicitly from iDynTree Math...")
    
    bodies_added = 0
    joints_added = 0
    
    def synthesize_body(parent_mjbody, link_idx, is_root=False, parent_joint_idx=None, parent_link_idx=None):
        nonlocal bodies_added, joints_added
        name = model.getLinkName(link_idx)
        link = model.getLink(link_idx)
        
        # Create Body
        body = parent_mjbody.add_body()
        body.name = name
        
        if is_root:
            body.pos = [0, 0, 0]
            # Add free joint for floating base
            free_j = body.add_freejoint()
            free_j.name = "root_freejoint"
        else:
            # --- TRANSLATE TOPOLOGY ---
            joint = model.getJoint(parent_joint_idx)
            
            # Extract SE(3) Transform matrix from iDynTree
            transform = joint.getRestTransform(parent_link_idx, link_idx)
            H = transform.asHomogeneousTransform()
            
            # Apply Translation
            body.pos = [H.getVal(0,3), H.getVal(1,3), H.getVal(2,3)]
            
            # Apply Rotation (Convert 3x3 to Quaternion via MuJoCo native math)
            mat = np.zeros(9)
            for r in range(3):
                for c in range(3):
                    mat[r*3 + c] = H.getVal(r, c)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat)
            body.quat = quat
            
            # --- TRANSLATE KINEMATICS ---
            dofs = joint.getNrOfDOFs()
            if dofs == 1:
                j = body.add_joint()
                j.name = model.getJointName(parent_joint_idx)
                joints_added += 1
                
                rev_joint = joint.asRevoluteJoint()
                if rev_joint:
                    j.type = mujoco.mjtJoint.mjJNT_HINGE
                    axis_obj = rev_joint.getAxis(link_idx)
                    axis_dir = axis_obj.getDirection()
                    j.axis = [axis_dir.getVal(0), axis_dir.getVal(1), axis_dir.getVal(2)]
                else:
                    prism_joint = joint.asPrismaticJoint()
                    if prism_joint:
                        j.type = mujoco.mjtJoint.mjJNT_SLIDE
                        axis_obj = prism_joint.getAxis(link_idx)
                        axis_dir = axis_obj.getDirection()
                        j.axis = [axis_dir.getVal(0), axis_dir.getVal(1), axis_dir.getVal(2)]

        # --- TRANSLATE PHYSICS (Inertia/Mass/CoM) ---
        inertia = link.getInertia()
        com = inertia.getCenterOfMass()
        body.ipos = [com.getVal(0), com.getVal(1), com.getVal(2)]
        body.mass = inertia.getMass()
        
        rot = inertia.getRotationalInertiaWrtCenterOfMass()
        body.fullinertia = [
            rot.getVal(0,0), rot.getVal(1,1), rot.getVal(2,2),
            rot.getVal(0,1), rot.getVal(0,2), rot.getVal(1,2)
        ]
        
        bodies_added += 1

        # --- ATTACH VISUAL ASSETS ---
        for v in visuals.get(name, []):
            mesh_filename = os.path.basename(v["filename"])
            mesh_name = f"{name}_mesh_{mesh_filename}"
            
            if mesh_name not in added_meshes:
                m = spec.add_mesh()
                m.name = mesh_name
                m.file = mesh_filename
                m.scale = v["scale"]
                added_meshes.add(mesh_name)
                
            geom = body.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh_name
            geom.pos = v["xyz"]
            geom.quat = rpy_to_quat(*v["rpy"])

        # Recurse
        for child_idx, joint_idx in children_map.get(link_idx, []):
            synthesize_body(body, child_idx, is_root=False, parent_joint_idx=joint_idx, parent_link_idx=link_idx)

    # Begin Graph Traversal
    synthesize_body(spec.worldbody, root_idx, is_root=True)
    
    print(f"  ✓ Synthesized {bodies_added} bodies and {joints_added} explicit joints")

    # 5. Export The Synthesized Spec
    print("\n[4] Compiling and saving explicit MJCF...")
    with open(output_mjcf_path, "w") as f:
        f.write(spec.to_xml())
        
    print(f"==================================================")
    print(f"SUCCESS: Explicit API Synthesis Complete")
    print(f"Output XML: {output_mjcf_path}")
    print(f"==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Input URDF (only used to initialize iDynTree and steal visual asset strings)")
    parser.add_argument("--output", default="synthesized_robot.xml", help="Output MJCF generated by API compilation")
    args = parser.parse_args()
    
    run_explicit_synthesis(args.urdf, args.output)
