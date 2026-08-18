"""Record a release split as a replayable session, then prove it survived.

    .venv/bin/python -m deploy.camera.record \
        --root . --split test --out sessions/test40 --fps 2

The frames come out of :class:`~.sources.SceneFolderSource` rather than
off the disk directly, so a recording sees the split exactly as the
camera service does -- same ordering, same validation, same "one bad
capture is counted and skipped". That also means the day this is pointed
at a sensor instead of a folder, only the constructed source changes.

The last thing it does is read its own output back and compare it,
frame by frame, against the folder it came from: depth bit-identical,
colour bit-identical, ``K`` and ``depth_scale`` equal to what
``camera.json`` said. A recorder that does not check its own output is
a recorder that will one day hand a customer a session of frames that
never happened, and nothing in the replay will look wrong.

Exit codes: 0 the session was written and verified, 1 it was written and
does not verify, 2 the arguments or the split are unusable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .session import (DEFAULT_FPS, SIDECAR_NAME, SessionError,
                      SessionReader, SessionWriter)
from .sources import (FrameReadError, FrameSourceError, SceneFolderSource,
                      SourceExhausted)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".",
                        help="release folder holding the split directories")
    parser.add_argument("--split", default="test",
                        help="split to record (default: test)")
    parser.add_argument("--out", required=True,
                        help="session directory to write")
    parser.add_argument("--scenes", default="",
                        help="comma-separated scene ids, in this order; "
                             "empty records the whole split, sorted")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help="playback rate stamped into the containers "
                             "(default: %g)" % DEFAULT_FPS)
    parser.add_argument("--no-preview", action="store_true",
                        help="skip the lossy preview.mp4; the two lossless "
                             "streams are what a replay reads")
    parser.add_argument("--force", action="store_true",
                        help="overwrite a session directory that already "
                             "holds one")
    args = parser.parse_args(argv)

    out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out, SIDECAR_NAME)) and not args.force:
        print("%s already holds a session; pass --force to replace it"
              % out, file=sys.stderr)
        return EXIT_USAGE
    split_dir = os.path.join(os.path.abspath(args.root), args.split)
    try:
        scenes = _scene_ids(split_dir, args.scenes)
        canvas, geometries = _canvas(split_dir, scenes)
    except FrameSourceError as exc:
        print("cannot record: %s" % exc, file=sys.stderr)
        return EXIT_USAGE
    source = SceneFolderSource(root=args.root, split=args.split,
                               scenes=scenes, loop=False, fps=0.0)
    try:
        source.open()
    except FrameSourceError as exc:
        print("cannot record: %s" % exc, file=sys.stderr)
        return EXIT_USAGE
    print("%d scene(s), %s" % (len(scenes), ", ".join(
        "%dx%d x%d" % (w, h, n) for (w, h), n in sorted(geometries.items()))))
    if len(geometries) > 1:
        print("  streams are %dx%d; the smaller frames are zero-padded into "
              "that canvas and cropped back on read" % canvas)

    try:
        recorded, skipped, elapsed = _record(source, out, args.fps,
                                             preview=not args.no_preview,
                                             canvas=canvas)
    except SessionError as exc:
        print("recording failed: %s" % exc, file=sys.stderr)
        return EXIT_USAGE
    finally:
        source.close()

    _report_sizes(out, recorded, elapsed)
    if skipped:
        print("  %d frame(s) were unreadable and are not in the session"
              % skipped)
    return _verify(out, os.path.join(os.path.abspath(args.root), args.split))


def _record(source: SceneFolderSource, out: str, fps: float, preview: bool,
            canvas: Tuple[int, int]) -> Tuple[List[str], int, float]:
    """Drive the source to exhaustion, writing every frame it delivers."""
    print("recording %s -> %s" % (os.path.join(source.root, source.split),
                                  out))
    started = time.perf_counter()
    recorded = []           # type: List[str]
    skipped = 0
    with SessionWriter(out, fps=fps, preview=preview,
                       canvas=canvas) as writer:
        while True:
            try:
                frame = source.read()
            except SourceExhausted:
                break
            except FrameReadError as exc:
                # One unreadable capture is a hole in the recording, not
                # the end of it: the session says so and carries on.
                skipped += 1
                print("  skipped: %s" % exc, file=sys.stderr)
                continue
            scene_id = _scene_id(frame)
            writer.append(rgb=frame.rgb, depth_raw=frame.depth_raw,
                          K=frame.K, depth_scale=frame.depth_scale,
                          scene_id=scene_id,
                          timestamp_ns=frame.timestamp_ns)
            recorded.append(scene_id)
            print("  [%3d] %s  %dx%d  %.0f%% depth"
                  % (len(recorded), scene_id, frame.width, frame.height,
                     100.0 * frame.valid_depth_fraction()))
    return recorded, skipped, time.perf_counter() - started


def _scene_id(frame: Any) -> str:
    """The id this frame is recorded and replayed under.

    A folder source names itself ``scene_folder:test/000003``, whose last
    component is the scene id the rest of this repository uses. A sensor
    names itself by serial, which is the same string for every frame, so
    the frame id is appended -- the label deploy/pick/runner.py gives the
    same frames, and for the same reason: a session addresses frames by
    id, and a session of forty frames all called ``realsense:923322071127``
    has thirty-nine nobody can ask for.
    """
    source = frame.source_name
    if "/" in source:
        return source.rsplit("/", 1)[-1]
    return "%s#%d" % (source, frame.frame_id)


def _scene_ids(split_dir: str, requested: str) -> Tuple[str, ...]:
    """The scenes to record, in the order they will be recorded."""
    if not os.path.isdir(split_dir):
        raise FrameSourceError("no such split directory: %s" % split_dir)
    ids = tuple(s.strip() for s in requested.split(",") if s.strip())
    if ids:
        missing = [s for s in ids
                   if not os.path.isdir(os.path.join(split_dir, s))]
        if missing:
            raise FrameSourceError("scenes not in %s: %s"
                                   % (split_dir, ", ".join(missing)))
        repeated = sorted(set(s for s in ids if ids.count(s) > 1))
        if repeated:
            # Refused here rather than at the encoder so that nothing has
            # been written yet: a session addresses frames by id, so the
            # second copy of a scene would be a frame no replay can reach.
            raise FrameSourceError("--scenes repeats %s; a session "
                                   "addresses frames by scene id, so each "
                                   "one can appear once"
                                   % ", ".join(repeated))
        return ids
    ids = tuple(sorted(name for name in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, name))))
    if not ids:
        raise FrameSourceError("no scene folders in %s" % split_dir)
    return ids


def _canvas(split_dir: str,
            scenes: Sequence[str]) -> Tuple[Tuple[int, int], Dict[Any, int]]:
    """Stream geometry big enough for every scene, and what the sizes are.

    A stream's frame size is fixed when its encoder opens, so the largest
    frame has to be known before the first one is written. The size comes
    from ``camera.json`` -- the release's own statement of the image its
    ``K`` was calibrated for -- and a picture that turns out bigger than
    that stops the recording in :meth:`SessionWriter.append`, where both
    numbers can be printed. Only a release that omits the fields pays for
    a decode here.
    """
    geometries = {}         # type: Dict[Any, int]
    for scene_id in scenes:
        scene_dir = os.path.join(split_dir, scene_id)
        camera = _camera_json(scene_dir) or {}
        if camera.get("width") and camera.get("height"):
            size = (int(camera["width"]), int(camera["height"]))
        else:
            rgb = cv2.imread(os.path.join(scene_dir, "rgb.png"),
                             cv2.IMREAD_COLOR)
            if rgb is None:
                raise FrameSourceError("cannot size %s/rgb.png" % scene_dir)
            size = (int(rgb.shape[1]), int(rgb.shape[0]))
        geometries[size] = geometries.get(size, 0) + 1
    canvas = (max(w for w, _ in geometries), max(h for _, h in geometries))
    return canvas, geometries


def _report_sizes(out: str, recorded: List[str], elapsed: float) -> None:
    """What was written, and what it costs per frame."""
    frames = max(1, len(recorded))
    total = 0
    print("wrote %s" % out)
    for name in sorted(os.listdir(out)):
        size = os.path.getsize(os.path.join(out, name))
        total += size
        print("  %-14s %10s  %10s/frame" % (name, _human(size),
                                            _human(size / frames)))
    print("  %-14s %10s  %10s/frame" % ("total", _human(total),
                                        _human(total / frames)))
    print("  %d frames in %.1f s  (%.0f ms/frame)"
          % (len(recorded), elapsed, 1000.0 * elapsed / frames))


def _verify(out: str, split_dir: str) -> int:
    """Read the session back and compare it with the folder it came from.

    Two different questions, both worth asking: the streams are the ones
    recorded (their checksums), and the pixels are the ones on disk in
    the release (compared here against a fresh ``cv2.imread``, not
    against anything the recorder kept in memory).
    """
    print("verifying %s" % out)
    started = time.perf_counter()
    failures = []           # type: List[str]
    worst_mm = 0.0
    with SessionReader(out) as reader:
        report = reader.verify()
        for name in sorted(report["streams"]):
            entry = report["streams"][name]
            print("  %-8s %-12s %s  %s" % (name, entry["file"],
                                           entry["sha1"][:12],
                                           "ok" if entry["ok"] else "CHANGED"))
            if not entry["ok"]:
                failures.append("%s does not match its recorded checksum"
                                % entry["file"])
        frames = reader.frame_count
        for index, scene_id in enumerate(reader.scene_ids()):
            problems, error_mm = _compare(reader.read(index), index,
                                          scene_id, split_dir)
            failures.extend(problems)
            worst_mm = max(worst_mm, error_mm)
        elapsed = time.perf_counter() - started
        print("  %d frames read back in %.1f s  (%.0f ms/frame)"
              % (frames, elapsed, 1000.0 * elapsed / max(1, frames)))
    if failures:
        for line in failures[:20]:
            print("  FAIL %s" % line, file=sys.stderr)
        print("  %d check(s) failed; worst depth deviation %.4f mm"
              % (len(failures), worst_mm), file=sys.stderr)
        return EXIT_FAILED
    print("  depth bit-identical, colour bit-identical, K and depth_scale "
          "exact, on every frame")
    return EXIT_OK


def _compare(captured: Any, index: int, scene_id: str,
             split_dir: str) -> Tuple[List[str], float]:
    """Every way this frame could differ from the release folder.

    Returns the problems found and the worst depth deviation in
    millimetres -- the unit the accuracy budget is written in, because
    "1834 ticks differ" says nothing about whether a pose moved.
    """
    scene_dir = os.path.join(split_dir, scene_id)
    problems = []           # type: List[str]
    error_mm = 0.0
    rgb = cv2.imread(os.path.join(scene_dir, "rgb.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(os.path.join(scene_dir, "depth.png"),
                       cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        return (["frame %d (%s): the source scene is no longer readable"
                 % (index, scene_id)], error_mm)
    if captured.rgb.shape != rgb.shape or captured.rgb.dtype != rgb.dtype:
        problems.append("frame %d (%s): colour decoded as %r %s, recorded "
                        "from %r %s" % (index, scene_id, captured.rgb.shape,
                                        captured.rgb.dtype, rgb.shape,
                                        rgb.dtype))
    elif not np.array_equal(captured.rgb, rgb):
        differing = int(np.count_nonzero(captured.rgb != rgb))
        problems.append("frame %d (%s): colour differs in %d of %d samples"
                        % (index, scene_id, differing, rgb.size))
    if captured.depth_raw.shape != depth.shape:
        problems.append("frame %d (%s): depth decoded as %r, recorded from "
                        "%r" % (index, scene_id, captured.depth_raw.shape,
                                depth.shape))
    elif not np.array_equal(captured.depth_raw, depth):
        delta = np.abs(captured.depth_raw.astype(np.int64) -
                       depth.astype(np.int64))
        differing = int(np.count_nonzero(delta))
        error_mm = float(delta.max()) * captured.depth_scale * 1000.0
        problems.append("frame %d (%s): depth differs in %d of %d pixels, "
                        "max %d ticks (%.4f mm), mean %.4f ticks over those "
                        "pixels" % (index, scene_id, differing, depth.size,
                                    int(delta.max()), error_mm,
                                    float(delta.sum()) / differing))
    camera = _camera_json(scene_dir)
    if camera is None:
        problems.append("frame %d (%s): camera.json is no longer readable"
                        % (index, scene_id))
        return problems, error_mm
    expected_K = np.asarray(camera["K"], dtype=np.float64)
    if not np.array_equal(captured.K, expected_K):
        problems.append("frame %d (%s): K is %r, camera.json says %r"
                        % (index, scene_id, captured.K.tolist(),
                           expected_K.tolist()))
    if captured.depth_scale != float(camera["depth_scale"]):
        problems.append("frame %d (%s): depth_scale is %r, camera.json says "
                        "%r" % (index, scene_id, captured.depth_scale,
                                camera["depth_scale"]))
    return problems, error_mm


def _camera_json(scene_dir: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(scene_dir, "camera.json")) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _human(size: float) -> str:
    """Bytes as something a size report can be read off at a glance."""
    for unit in ("B", "kB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024.0
    return "%.1f GB" % size


if __name__ == "__main__":
    sys.exit(main())
