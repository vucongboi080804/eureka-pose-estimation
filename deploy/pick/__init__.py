"""Robot-cell integration: from an estimated pose to an executed pick.

The pose pipeline (``src/``) and the pose service (``deploy/pose/``)
answer where the parts are. This package is what a cell needs on top of
that to actually pick one, and it is deliberately separable from both:
nothing here estimates a pose, and nothing here moves a robot.

    frames       rigid transforms with their frame names attached, so
                 T_base_object = T_base_camera @ T_camera_object is the
                 only composition that type-checks
    calibration  the camera-to-robot (hand-eye) solve, with the residual
                 and uncertainty evidence that says whether to install it
    grasp        measured grasps on the part, ranked per frame by score,
                 top-of-pile and approach clearance
    drift        the watch on calibration drift -- the field failure that
                 degrades slowly and raises no software fault
    policy       the pick cycle as an explicit state machine
    runner       the loop over the camera and pose services: one cycle
                 per frame, one JSON line per cycle

Units are metres and seconds throughout, poses are ``T_camera_object``
in the OpenCV camera convention, and every library module runs its own
verification under ``python -m deploy.pick.<module>`` (``runner`` is the
loop itself).
"""

from __future__ import annotations

#: Bumped when anything in this package's public surface changes.
CELL_VERSION = "1.0"
