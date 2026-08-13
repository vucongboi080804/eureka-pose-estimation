# 6-DoF pose estimation assignment

## Background

A robot cell picks parts one at a time out of a tray. A camera mounted above the
tray captures a colour image and a depth map, a vision program locates every
part and estimates its pose, and the robot uses that pose to approach and grasp
one part. The parts are tipped into the tray loose, so they come to rest in
arbitrary orientations, lie against one another and partially occlude one
another, and no two trays are alike.

A pick point on its own is not enough. The gripper has to be oriented to the
part it is picking, and the part may be lying on any of its faces, so the vision
program has to report a full 6-DoF pose rather than a position. This assignment
is that vision program, developed and evaluated offline on recorded captures.

## Task

Each scene is one RGB-D capture of a tray holding multiple instances of a single
rigid part. Given the RGB image, the depth map, the camera intrinsics and the CAD
model, detect every instance of the part and estimate its 6-DoF pose.

`train/` contains scenes with ground truth. `test/` contains scenes with images
only. Submit one `submission.json` covering every test scene.

Poses are required only for instances at least 80% visible, where visibility is
the area of an instance's visible region divided by the area it would cover
unoccluded. Instances below that threshold do not affect the score.

Any method and any library may be used. Include a short write-up of the approach
and its known limitations.

## Contents

```
model/3d_model.glb        CAD model, metres, in the object frame
model/3d_model.ply        the same model
train/<scene>/
  rgb.png                 8-bit colour
  depth.png               16-bit, millimetres, 0 = no measurement
  camera.json             intrinsics, image size, depth scale
  masks/000.png           binary 0/255, one file per labelled instance
  ignore/000.png          see "Ignore regions" (may be absent)
  poses.json              ground truth
test/<scene>/
  rgb.png  depth.png  camera.json
submission_template.json  every test scene id, with empty lists
visualize.py              overlays ground truth on the images
score.py                  the scoring script used for grading
```

The two scripts require `numpy`, `opencv-python`, `matplotlib` and `trimesh`.

## Conventions

* A pose is `T_camera_object`, mapping a point in the object frame to the camera
  frame:

  ```
  p_camera = R @ p_cad + t
  ```

* `R` is a row-major 3x3 rotation. `t` is in metres, as is the CAD model, so the
  mesh requires no rescaling.
* The camera frame follows the OpenCV convention: +X right, +Y down, +Z forward
  along the optical axis.
* `K` is a standard pinhole matrix. Images are rectified and lens distortion is
  zero, so projection is `K @ p_camera` followed by a perspective divide.
* `Z_metres = depth_png_value * camera.json["depth_scale"]`. Depth is registered
  to the colour image: pixel `(u, v)` is the same surface point in both. Pixels
  with no measurement are set to 0.
* Masks cover the visible region only: an occluded instance's mask covers just
  the part of it that is visible.
* The part is asymmetric, so each instance has exactly one correct pose.
* Intrinsics and image size vary between scenes.

## Visualisation

```bash
python visualize.py --root . --split train
python visualize.py --root . --split train --save overlays/
```

Draws each mask and each ground-truth pose, as axes at the object origin, over
the scene image.

## Ignore regions

`ignore/` contains masks of visible instances that carry no ground-truth pose.
They are neither positives nor negatives: predictions landing on them are
discarded rather than counted as errors. Not every scene has them.

## Submission

Return two things:

* `submission.json`, in the format below.
* A folder of the test images with the predicted poses drawn on them, one image
  per test scene, in the style of `visualize.py`.

`submission.json` is keyed by test scene id, with a list of predictions per
scene. Copy `submission_template.json` and fill in the lists.

```json
{
  "000003": [
    {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.01, -0.02, 0.55], "score": 0.94},
    {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.03, 0.00, 0.56], "score": 0.71}
  ],
  "000008": []
}
```

`R` row-major 3x3 and `t` in metres, following the conventions above.
Predictions may be listed in any order; each is matched to ground truth by pose,
not by position in the list. `score` is the prediction's confidence and sets its
rank within the scene. A prediction without a `score` ranks below every scored
prediction.

## Evaluation

`score.py` is the script used for grading. It runs against `train/`, where the
labels ship with the images:

```bash
python score.py --release . --split train --submission my_train_poses.json
python score.py --release . --split train --selftest
```

The metric is MSSD (maximum symmetry-aware surface distance): the largest
deviation of any point on the model, taken between corresponding vertices and
minimised over the object's symmetry transforms. For this asymmetric object that
set is the identity, so

```
e = max over model vertices x of  || (R_est - R_gt) x + (t_est - t_gt) ||
```

* Recall and precision are reported at MSSD thresholds of 2, 4, 6, 8 and 10 mm,
  and averaged into a single AR.
* Predictions are matched in order of descending `score`, each claiming the
  closest unclaimed instance within the threshold. Each instance is claimed at
  most once.
* top-1 per scene: the fraction of scenes whose highest-scoring prediction is
  within 5 mm of an instance. A scene with no prediction counts against it.
* Instances less than 80% visible are optional, as stated under Task.
* Predictions landing on an `ignore` region are discarded rather than counted as
  false positives.
