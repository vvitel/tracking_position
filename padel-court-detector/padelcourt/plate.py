"""Build a clean plate - a still photo of the empty court - from a video.

A pixel-wise median over frames spread across the whole recording. Anything that
moves is somewhere different in every frame and loses the median; the court is
identical in all of them and survives. Players and ball disappear.

Two things here are not incidental.

SPREAD OVER THE WHOLE RECORDING, not a window. Measured on a 30s clip: five
plates each built from a 15-frame window, and one of them kept a player who
happened to stand still - a ghost sitting in the far half, which is the part of
the court that constrains the fit most weakly. That plate produced a calibration
50px out in mid-court. Widening the window from 15 frames to 41 did not help,
because the player was stationary for the whole window; spreading 45 frames over
the entire clip dropped the largest ghost from 5943px to 80px and the fit passed.

KEYFRAMES, decoded with `-skip_frame nokey`. These recordings are raw H.264
muxed into MKV with no index, no duration and no usable timestamps - ffprobe
reports duration N/A, `ffmpeg -ss 600` returns nothing, and OpenCV's seek to
600s lands at 1.9s. So there is no way to ask for "the frame at 12 minutes".
Keyframes are the way in: they are self-contained, ffmpeg can walk them without
decoding anything else (a 21-minute file in ~25s), and they are already evenly
spaced through the recording, which is exactly the sampling wanted.
"""
import os
import shutil
import subprocess

import cv2
import numpy as np

#: No ffmpeg on PATH on this machine; the AgentDVR build is the one that works.
FFMPEG_DIRS = ["/opt/AgentDVR/ffmpeg7/bin"]
VIDEO_EXT = (".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".webm")


def find_ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p, {}
    for d in FFMPEG_DIRS:
        p = os.path.join(d, "ffmpeg")
        if os.path.exists(p):
            lib = os.path.join(os.path.dirname(d), "lib")
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
            return p, env
    raise SystemExit("no ffmpeg found (looked on PATH and in %s)" % ", ".join(FFMPEG_DIRS))


def plate_path(video):
    return os.path.splitext(video)[0] + "_plate.png"


def keyframes(video, target=45, cap=None, verbose=True):
    """Sample ~`target` keyframes spread evenly over the whole video.

    One pass, bounded memory, and no need to know the length in advance - which
    matters because these files do not report one. Keep every frame while the
    buffer is small; when it overflows, throw away every second frame and double
    the stride. That is systematic sampling with an adaptive stride, so the
    result stays evenly spread over whatever the recording turns out to be,
    rather than clustered the way a random reservoir sample can be.
    """
    ff, env = find_ffmpeg()
    cap = cap or 2 * target
    probe = cv2.VideoCapture(video)
    w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    probe.release()
    n_bytes = w * h * 3

    cmd = [ff, "-v", "error", "-skip_frame", "nokey", "-i", video,
           "-an", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            env=env, bufsize=n_bytes)
    buf, stride, seen = [], 1, 0
    try:
        while True:
            raw = proc.stdout.read(n_bytes)
            if len(raw) < n_bytes:
                break
            if seen % stride == 0:
                buf.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
                if len(buf) > cap:
                    buf = buf[::2]
                    stride *= 2
            seen += 1
    finally:
        proc.stdout.close()
        proc.wait()
    if not buf:
        raise SystemExit("no keyframes decoded from %s" % os.path.basename(video))
    if len(buf) > target:                      # trim to exactly `target`, evenly
        idx = np.linspace(0, len(buf) - 1, target).round().astype(int)
        buf = [buf[i] for i in idx]
    if verbose:
        print("      %d keyframes seen, %d kept (stride %d)" % (seen, len(buf), stride))
    return buf


def ghost_score(frames):
    """How much of a moving object survived the median, with no other plate.

    Median the odd-numbered frames and the even-numbered ones separately and
    difference them. Anything genuinely static cancels; a player who stood still
    through part of the recording survives in one half and not the other, and
    shows up as a large connected blob. Measured on known cases: a clean plate
    scores under ~150px, a plate whose calibration failed scored 5943.
    """
    if len(frames) < 8:
        return 0
    a = np.median(np.array(frames[0::2], np.float32), axis=0)
    b = np.median(np.array(frames[1::2], np.float32), axis=0)
    m = (np.linalg.norm(a - b, axis=2) > 25).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


def build(video, target=45, force=False, verbose=True):
    """Return (plate BGR, path, ghost or None). Cached: an existing plate is reused."""
    path = plate_path(video)
    if os.path.exists(path) and not force:
        img = cv2.imread(path)
        if img is not None:
            if verbose:
                print("      plate cached: %s" % os.path.basename(path))
            return img, path, None
    frames = keyframes(video, target=target, verbose=verbose)
    plate = np.median(np.array(frames, np.float32), axis=0).astype(np.uint8)
    g = ghost_score(frames)
    cv2.imwrite(path, plate)
    if verbose:
        print("      wrote %s   (ghost %d px%s)"
              % (os.path.basename(path), g, ", SUSPECT" if g > 3000 else ""))
    return plate, path, g


