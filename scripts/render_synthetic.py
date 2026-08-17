import blenderproc as bproc

# Render a domain-randomised synthetic training set from the CAD alone.
# (BlenderProc insists its import is the first statement of the script.)
#
# Run with BlenderProc's launcher (it hosts its own Blender):
#
#     .venv/bin/blenderproc run scripts/render_synthetic.py -- \
#         --cad model/3d_model.ply --out seg_data/synthetic --frames 1200
#
# Every frame drops 3-12 instances of the part into a tray (or onto a bare
# surface) with physics, then randomises everything a real cell could
# change: part material colour (orange only ~35% of the time), lighting
# count/position/energy/colour, floor and tray colours, camera height and
# tilt. Labels are written straight in YOLO-seg polygon format, so the
# output directory can be handed to `yolo segment train` after adding a
# data.yaml. Nothing here is specific to this part: pass any CAD to
# onboard a new object with zero hand labels.

import argparse
import os
import random

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--cad", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--frames", type=int, default=1200)
parser.add_argument("--start", type=int, default=0)
args = parser.parse_args()

W, H = 960, 640
os.makedirs(os.path.join(args.out, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(args.out, "labels", "train"), exist_ok=True)

bproc.init()
bproc.renderer.set_max_amount_of_samples(32)
# Instance masks are computed by hand below (project the CAD at each
# part's simulated pose and keep pixels that agree with the rendered
# depth) -- Blender's segmentation pass returns empty maps for objects
# duplicated after it is enabled.
bproc.renderer.enable_depth_output(activate_antialiasing=False)

base_part = bproc.loader.load_obj(args.cad)[0]
base_part.set_name("part_base")
base_part.hide(True)

_mesh = base_part.mesh_as_trimesh()
_samples, _ = _mesh.sample(25000, return_index=True)
_samples = np.asarray(_samples)
#: Blender cameras look along -Z with +Y up; OpenCV looks along +Z with
#: +Y down. Right-multiplying the camera-to-world matrix by this flip
#: converts it to the OpenCV convention the projection below assumes.
BLENDER_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def visible_mask(points_world, cam2world_cv, K, scene_depth,
                 tol: float = 0.0025):
    """Pixels where this instance's surface IS the rendered depth."""
    world2cam = np.linalg.inv(cam2world_cv)
    pc = points_world @ world2cam[:3, :3].T + world2cam[:3, 3]
    front = pc[:, 2] > 1e-6
    pc = pc[front]
    if not len(pc):
        return np.zeros((H, W), np.uint8)
    u = np.rint(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(np.int64)
    v = np.rint(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], pc[ok, 2]
    zbuf = np.full(H * W, np.inf)
    np.minimum.at(zbuf, v * W + u, z)
    zbuf = zbuf.reshape(H, W)
    import cv2
    mask = (np.abs(zbuf - scene_depth) < tol).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

floor = bproc.object.create_primitive("PLANE", scale=[2, 2, 1])
floor.enable_rigidbody(active=False, friction=0.8)


def random_material(obj, orange_bias=False):
    mat = bproc.material.create("m_%d" % random.randint(0, 10**9))
    if orange_bias and random.random() < 0.35:
        hue = random.uniform(0.01, 0.05)          # the real part's orange
        sat = random.uniform(0.75, 0.95)
    else:
        hue = random.random()                     # any colour at all
        sat = random.uniform(0.2, 0.95)
    val = random.uniform(0.15, 0.9)
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    mat.set_principled_shader_value("Base Color", [r, g, b, 1.0])
    mat.set_principled_shader_value("Roughness", random.uniform(0.3, 0.95))
    obj.replace_materials(mat)


def build_tray():
    """Four walls around the drop zone; sometimes no tray at all."""
    if random.random() < 0.3:
        return []
    size = random.uniform(0.11, 0.16)
    height = random.uniform(0.03, 0.06)
    thick = 0.008
    walls = []
    for dx, dy, sx, sy in ((size, 0, thick, size), (-size, 0, thick, size),
                           (0, size, size, thick), (0, -size, size, thick)):
        wall = bproc.object.create_primitive(
            "CUBE", scale=[sx, sy, height], location=[dx, dy, height])
        wall.enable_rigidbody(active=False, friction=0.8)
        walls.append(wall)
    return walls


frame = args.start
while frame < args.start + args.frames:
    parts = []
    for _ in range(random.randint(3, 12)):
        p = base_part.duplicate()
        p.set_name("part_copy")
        p.hide(False)
        p.enable_rigidbody(active=True, friction=0.6,
                           collision_shape="CONVEX_HULL")
        parts.append(p)
    same_colour = random.random() < 0.8
    random_material(parts[0], orange_bias=True)
    shared = parts[0].get_materials()[0]
    for p in parts[1:]:
        if same_colour:
            p.replace_materials(shared)
        else:
            random_material(p, orange_bias=True)

    random_material(floor)
    tray = build_tray()
    for wall in tray:
        random_material(wall)

    def sampler(obj):
        obj.set_location(np.random.uniform([-0.07, -0.07, 0.08],
                                           [0.07, 0.07, 0.25]))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    bproc.object.sample_poses(parts, sample_pose_func=sampler)
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=1.5, max_simulation_time=3.0,
        check_object_interval=0.5)

    lights = []
    for _ in range(random.randint(1, 3)):
        light = bproc.types.Light()
        light.set_type(random.choice(["POINT", "AREA", "SUN"]))
        light.set_location(np.random.uniform([-1, -1, 0.5], [1, 1, 1.5]))
        light.set_energy(random.uniform(20, 400)
                         if light.get_type() != "SUN"
                         else random.uniform(1, 6))
        warm = random.uniform(0.85, 1.0)
        light.set_color([1.0, warm, random.uniform(0.7, 1.0)])
        lights.append(light)

    fx = random.uniform(1600, 2200)
    K = np.array([[fx, 0, W / 2 + random.uniform(-260, 260)],
                  [0, fx, H / 2 + random.uniform(-160, 160)],
                  [0, 0, 1.0]])
    bproc.camera.set_intrinsics_from_K_matrix(K, W, H)
    cam_location = np.array([random.uniform(-0.06, 0.06),
                             random.uniform(-0.06, 0.06),
                             random.uniform(0.55, 0.9)])
    rotation = bproc.camera.rotation_from_forward_vec(
        np.array([0.0, 0.0, -0.03]) - cam_location,
        inplane_rot=random.uniform(0, 2 * np.pi))
    cam2world = bproc.math.build_transformation_mat(cam_location, rotation)
    bproc.camera.add_camera_pose(cam2world, frame=0)

    data = bproc.renderer.render()
    rgb = data["colors"][0]
    scene_depth = data["depth"][0]

    import cv2
    cam2world_cv = cam2world @ BLENDER_TO_CV
    lines = []
    for p in parts:
        T = np.asarray(p.get_local2world_mat())
        pts_world = _samples @ T[:3, :3].T + T[:3, 3]
        mask = visible_mask(pts_world, cam2world_cv, K, scene_depth)
        if mask.sum() < 400:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 300:
                continue
            poly = (c.reshape(-1, 2) / [W, H]).reshape(-1)
            lines.append("0 " + " ".join("%.5f" % v for v in poly))
    if lines:
        name = "syn_%06d" % frame
        cv2.imwrite(os.path.join(args.out, "images", "train", name + ".png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        with open(os.path.join(args.out, "labels", "train",
                               name + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        frame += 1
        if frame % 50 == 0:
            print("rendered %d frames" % frame, flush=True)

    for obj in parts + tray + lights:
        obj.delete()
    bproc.utility.reset_keyframes()

print("done: %d frames in %s" % (args.frames, args.out))
