"""One-shot diagnostic for the process_data.py failures on h08/s31/a06/i29.
Doesn't fix anything -- just reports exactly what's wrong so we know which
fix to write. Run this, paste the output.

Usage: python diagnose_scene_issues.py
"""

from pathlib import Path
import pandas as pd


def check_h08():
    print("=== h08: missing mic parquet files ===")
    d = Path("datasets/h08/h08")
    if not d.exists():
        print(f"  {d} does not exist at all.")
        return
    all_files = sorted(f.name for f in d.iterdir())
    print(f"  {d} contains {len(all_files)} files:")
    for f in all_files[:20]:
        print(f"    {f}")
    if len(all_files) > 20:
        print(f"    ... and {len(all_files) - 20} more")


def check_a06_nan():
    print("\n=== a06: NaN causing int conversion failure ===")
    d = Path("datasets/a06/a06")
    mic_path = d / "run0_rs1_mic.parquet"
    if not mic_path.exists():
        print(f"  {mic_path} not found")
        return
    df = pd.read_parquet(mic_path)
    print(f"  {mic_path.name}: {len(df)} rows, columns: {list(df.columns)}")
    if "timestamp" in df.columns:
        n_nan = df["timestamp"].isna().sum()
        print(f"  NaN timestamps: {n_nan} / {len(df)}")
        print(f"  timestamp dtype: {df['timestamp'].dtype}")
        print(f"  timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print("  NO 'timestamp' column found -- that's the real problem")


def check_segment_mismatch(scene: str, mic_filename: str):
    print(f"\n=== {scene}: mic/geo segment mismatch ({mic_filename}) ===")
    d = Path(f"datasets/{scene}/{scene}")
    mic_path = d / mic_filename
    geo_path = d / mic_filename.replace("_mic.parquet", "_geo.parquet")
    if not mic_path.exists() or not geo_path.exists():
        print(f"  missing file: mic={mic_path.exists()} geo={geo_path.exists()}")
        return

    mic_df = pd.read_parquet(mic_path)
    geo_df = pd.read_parquet(geo_path)
    print(f"  mic rows: {len(mic_df)}, geo rows: {len(geo_df)}")

    for name, df in [("mic", mic_df), ("geo", geo_df)]:
        if "timestamp" not in df.columns:
            print(f"  {name}: NO 'timestamp' column, has: {list(df.columns)}")
            continue
        n_nan = df["timestamp"].isna().sum()
        first_ts = df["timestamp"].min()
        last_ts = df["timestamp"].max()
        print(f"  {name}: {n_nan} NaN timestamps, range [{first_ts}, {last_ts}], "
              f"span={last_ts - first_ts if pd.notna(first_ts) and pd.notna(last_ts) else 'N/A'}")


def check_segment_sizes(scene: str, mic_filename: str, segment_seconds: float = 2.0):
    """Diagnose whether a specific file has a pathologically oversized
    segment -- e.g. from a timestamp gap or corrupted data -- that would
    make target_samples = grouped.size().max() blow up memory for every
    other segment in the file too (every segment gets padded to that max)."""
    print(f"\n=== {scene}: segment size distribution ({mic_filename}) ===")
    d = Path(f"datasets/{scene}/{scene}")
    mic_path = d / mic_filename
    if not mic_path.exists():
        print(f"  {mic_path} not found")
        return

    df = pd.read_parquet(mic_path)
    if "timestamp" not in df.columns:
        print(f"  NO 'timestamp' column, has: {list(df.columns)}")
        return

    first_ts = df["timestamp"].min()
    segment_number = ((df["timestamp"] - first_ts) // segment_seconds).astype(int)
    sizes = segment_number.value_counts()

    print(f"  total rows: {len(df)}, distinct segments: {len(sizes)}")
    print(f"  segment size: min={sizes.min()}, median={sizes.median():.0f}, "
          f"mean={sizes.mean():.0f}, max={sizes.max()}")
    print(f"  target_samples would be set to: {sizes.max()} "
          f"(every segment gets padded/truncated to this)")
    if sizes.max() > 10 * sizes.median():
        biggest_seg = sizes.idxmax()
        seg_rows = df[segment_number == biggest_seg]
        print(f"  ANOMALY: max segment ({sizes.max()} rows) is >10x the median "
              f"({sizes.median():.0f} rows) -- this is very likely the cause of "
              f"the memory blowup. Largest segment's timestamp range: "
              f"[{seg_rows['timestamp'].min()}, {seg_rows['timestamp'].max()}]")
        # Check for a timestamp gap: is this "segment" actually one real
        # 2-second window, or did a jump in timestamps make many real rows
        # collapse into what // segment_seconds treats as a single bucket?
        ts_diffs = seg_rows["timestamp"].diff().dropna()
        if len(ts_diffs) > 0:
            print(f"  within that segment, timestamp diffs: "
                  f"min={ts_diffs.min()}, max={ts_diffs.max()}, "
                  f"median={ts_diffs.median()}")


if __name__ == "__main__":
    check_h08()
    check_a06_nan()
    check_segment_mismatch("s31", "run0_rs5_mic.parquet")
    check_segment_mismatch("i29", "run0_rs2_mic.parquet")
    check_segment_sizes("s31", "run8_rs1_mic.parquet")