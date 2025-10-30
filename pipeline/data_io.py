from pathlib import Path
from typing import Union, Optional
import polars as pl
from io import StringIO


def read_device_csv(
    filepath: Union[str, Path], 
    columns: list[str] = ["z"], 
    logging: bool = False
) -> pl.DataFrame:
    """
    Reads a CSV file from the device containing accelerometer data.
    Skips malformed lines with wrong column count or non-integer values.
    
    The CSV format is expected to be:
    - Header line (skipped)
    - Data lines: x,y,z (comma-separated integers)
    
    Args:
        filepath: Path to CSV file.
        columns: List of columns to return. Options: ["x"], ["y"], ["z"], or combinations.
        logging: If True, writes a log file about read/skipped lines.
    
    Returns:
        Polars DataFrame with requested columns (dtype Int64).
    """
    info = {"file_path": str(filepath), "skipped_lines": 0}
    clean_lines = []

    with open(filepath, "r") as f:
        next(f, None)  # skip header line
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                info["skipped_lines"] += 1
                continue
            try:
                int(parts[0])
                int(parts[1])
                int(parts[2])
                clean_lines.append(line)
            except ValueError:
                info["skipped_lines"] += 1

    if not clean_lines:
        raise ValueError(f"No valid data lines found in {filepath}")

    buffer = StringIO("".join(clean_lines))

    df = pl.read_csv(
        buffer,
        has_header=False,
        schema={"x": pl.Int64, "y": pl.Int64, "z": pl.Int64}
    )

    if logging:
        info["line_count"] = df.height
        log_path = Path(filepath).with_suffix(".LOG.txt")
        log_path.write_text(str(info))

    return df.select(columns)


class ContinuousCSVReader:
    """
    Reads newly appended lines from a CSV file containing accelerometer data (integer x, y, z columns).
    Can be polled periodically for new data.
    """

    def __init__(self, infile: Union[str, Path]):
        self.infile = Path(infile)
        self._last_pos = 0

    def read_new(self) -> Optional[pl.DataFrame]:
        """
        Reads new lines appended since last read.

        Returns:
            Polars DataFrame of new rows, or None if no new valid rows.
        """

        with self.infile.open("r") as f:
            f.seek(self._last_pos)
            lines = f.readlines()
            self._last_pos = f.tell()

        if not lines:
            return None

        # Only keep lines with exactly 3 integers to avoid bad lines
        data = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 3:
                try:
                    data.append([int(p) for p in parts])
                except ValueError:
                    continue

        if not data:
            return None

        return pl.DataFrame(data, schema={"x": pl.Int64, "y": pl.Int64, "z": pl.Int64})

    def set_delay(self, delay: float):
        """Update the read delay."""
        self.delay = delay

    def reset_position(self):
        """Reset internal file pointer to start of file."""
        self._last_pos = 0
