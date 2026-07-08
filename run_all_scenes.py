"""Convenience driver: process + run empirical_outcomes for every M3N-VC
scene in one command, instead of running process_data.py and
empirical_outcomes.py by hand per scene.

Usage
-----
    python run_all_scenes.py                     # all 6 scenes
    python run_all_scenes.py --scenes h08 s31     # just these two
    python run_all_scenes.py --skip-process       # outcomes only, if
                                                   # processed arrays already exist

Requires each scene's raw data at datasets/<scene>/<scene>/ (mic/geo/dis
parquet files + run_ids.parquet + sensor_location.parquet), matching the
M3N-VC release layout. h24 should already be there from prior work; the
other five need to be downloaded and unzipped into that same layout first.
"""

from __future__ import annotations

import argparse
import traceback

from empirical_outcomes import collect_empirical_outcomes
from process_data import save_scene_paired_arrays

ALL_SCENES = ["h24", "h08", "s31", "a06", "i29", "i22"]


def run_all(
    scenes: list[str],
    skip_process: bool = False,
    batch_size: int = 64,
    datasets_root: str | Path = "datasets",
) -> dict[str, str]:
    """Returns {scene: "ok" | "<error message>"} so one scene's missing
    data or a mid-run crash doesn't stop the others from being attempted.
    """
    results: dict[str, str] = {}
    for scene in scenes:
        print(f"\n{'=' * 60}\nSCENE: {scene}\n{'=' * 60}")
        try:
            if not skip_process:
                print(f"[{scene}] processing raw parquet -> spectrogram arrays...")
                save_scene_paired_arrays(scene, datasets_root=datasets_root)
            print(f"[{scene}] running empirical_outcomes (frozen h24 models, zero-shot)...")
            collect_empirical_outcomes(scene=scene, batch_size=batch_size)
            results[scene] = "ok"
        except FileNotFoundError as e:
            print(f"[{scene}] SKIPPED -- data not found: {e}")
            results[scene] = f"missing data: {e}"
        except Exception as e:
            print(f"[{scene}] FAILED: {e}")
            traceback.print_exc()
            results[scene] = f"error: {e}"

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for scene, status in results.items():
        print(f"  {scene:>6}: {status}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES,
                        help=f"Which scenes to run (default: all of {ALL_SCENES})")
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip process_data.py step (use if <scene>_paired_*.npy already exist)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--data-root",
        default="datasets",
        help="Root folder containing scene subdirs (default: datasets/)",
    )
    args = parser.parse_args()

    run_all(
        args.scenes,
        skip_process=args.skip_process,
        batch_size=args.batch_size,
        datasets_root=args.data_root,
    )
