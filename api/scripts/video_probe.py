"""Video mit Grifferkennungs-Overlay rendern (CLI-Hülle).

Die eigentliche Logik liegt in vision/video_overlay.py und wird auch vom
/process_video-Endpunkt genutzt.

Aufruf aus api/:
    python scripts/video_probe.py --in video.mp4 --out annotiert.mp4 --shapes
"""

import argparse
import os
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))


def main():
    parser = argparse.ArgumentParser(description="Video-Overlay-Renderer")
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", dest="dst", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--shapes", action="store_true")
    parser.add_argument("--keyframe", type=int, default=20,
                        help="SAM-Schlüsselbild alle n Frames (Vielfaches von stride)")
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug", default=None,
                        help="Debug-JSON mit Tracks, Motion und Maskenmetriken")
    parser.add_argument("--legacy", action="store_true",
                        help="alten Ein-Pass-Renderer fuer A/B-Vergleich nutzen")
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from vision.video_overlay import render_video, render_video_legacy

    renderer = render_video_legacy if args.legacy else render_video
    render_kwargs = {}
    if not args.legacy:
        render_kwargs.update(debug_path=args.debug, device=args.device)
    frames, dauer = renderer(
        args.src, args.dst, weights=args.weights, conf=args.conf,
        stride=args.stride, shapes=args.shapes, keyframe=args.keyframe,
        max_seconds=args.max_seconds or 90.0,
        progress_cb=lambda p: print(f"  {p*100:.0f}%", flush=True)
        if int(p * 100) % 20 == 0 else None,
        **render_kwargs,
    )
    print(f"fertig: {args.dst} ({frames} Frames in {dauer:.0f}s)")


if __name__ == "__main__":
    main()
