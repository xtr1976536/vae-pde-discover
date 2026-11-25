"""Read 40-frame windows directly from memory-mapped Cylinder Arrow shards."""

import hashlib
import json

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

from .splits import HISTORY, ROLES, SPAN, validate


class CylinderStore:
    def __init__(self, shards, splits):
        audit = json.loads((splits / "split_audit.json").read_text())
        self.splits = {}
        for role in ROLES:
            path = splits / f"{role}_index_real.json"
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != audit["prepared_sha256"][path.name]:
                raise ValueError(f"Index hash mismatch: {path}")
            self.splits[role] = json.loads(raw)
        validate(self.splits)
        wanted = {row["sim_id"] for rows in self.splits.values() for row in rows}
        self.trajectories = {}
        for path in sorted(shards.glob("*.arrow")):
            with pa.memory_map(str(path)) as source:
                for batch in pa.ipc.open_stream(source):
                    for i, scalar in enumerate(batch["sim_id"]):
                        sim = scalar.as_py()
                        if sim not in wanted:
                            continue
                        if sim in self.trajectories:
                            raise ValueError(f"Duplicate Arrow trajectory: {sim}")
                        shape = tuple(batch[f"shape_{axis}"][i].as_py() for axis in "thw")
                        if shape[1:] != (64, 128):
                            raise ValueError(f"Unexpected grid for {sim}: {shape}")
                        times = np.frombuffer(batch["t"][i].as_buffer(), dtype="<f8")
                        if len(times) != shape[0] or not np.allclose(np.diff(times), .005, atol=5e-7):
                            raise ValueError(f"Unexpected time grid: {sim}")
                        self.trajectories[sim] = tuple(
                            np.frombuffer(batch[field][i].as_buffer(), dtype="<f4").reshape(shape)
                            for field in ("u", "v")
                        )
        if missing := wanted - self.trajectories.keys():
            raise ValueError(f"Missing Arrow trajectories: {sorted(missing)}")
        for rows in self.splits.values():
            for row in rows:
                if row["time_id"] + SPAN > len(self.trajectories[row["sim_id"]][0]):
                    raise ValueError(f"Incomplete window: {row}")
        y, x = np.ogrid[:64, :128]
        self.fluid = (x - 32)**2 + (y - 32)**2 > 9**2
        self.audit = audit

    def read(self, row):
        start = row["time_id"]
        fields = self.trajectories[row["sim_id"]]
        return np.stack([field[start:start + SPAN] for field in fields], axis=-1)

    def normalize(self, train_rows):
        # Count each distinct observed training frame once, even when windows overlap.
        by_sim = {}
        for row in train_rows:
            sim, start = row["sim_id"], row["time_id"]
            by_sim.setdefault(sim, set()).update(range(start, start + SPAN))
        sums, squares, count = np.zeros(2), np.zeros(2), 0
        for sim, indices in by_sim.items():
            frames = np.fromiter(sorted(indices), dtype=np.intp)
            for batch in np.array_split(frames, (len(frames) + 63) // 64):
                for i, field in enumerate(self.trajectories[sim]):
                    values = field[batch][:, self.fluid].astype(np.float64)
                    sums[i] += values.sum()
                    squares[i] += np.square(values).sum()
                count += values.size
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - mean**2, 1e-12))
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Nonfinite training normalization")
        return mean.astype(np.float32), std.astype(np.float32)


class Windows(Dataset):
    def __init__(self, store, rows, mean, scale):
        self.store, self.rows, self.mean, self.scale = store, rows, mean, scale

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        normalized = (self.store.read(self.rows[index]) - self.mean) / self.scale
        if not np.isfinite(normalized[:, self.store.fluid]).all():
            raise ValueError(f"Nonfinite fluid measurements: {self.rows[index]}")
        normalized[:, ~self.store.fluid] = 0.
        windows = torch.from_numpy(normalized)
        return index, windows[:HISTORY], windows[HISTORY:]
