"""Helpers that two or more parts of the cell share.

Nothing here knows about frames, poses or grasps, and nothing here imports
``src/`` or another ``deploy`` package: the camera has to be deployable on a
machine that carries no estimator, and the estimator on a board with no
camera, so the only direction a dependency may point is *into* this folder.
One file per helper: ``jsonlog`` (one JSON object per log line), ``stats``
(percentiles), ``host`` (what machine this is), ``clock`` (elapsed time).
"""
