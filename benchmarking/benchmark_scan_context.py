import csv
import glob
import numpy as np
import argparse
from scipy.spatial import cKDTree
from pathlib import Path

class ScanContext:
    """
    Generates Scan Context descriptors and computes distances between them.
    Reference: Scan Context: Egocentric Spatial Descriptor for Place Recognition (IROS 2018)
    """
    def __init__(self, num_rings=20, num_sectors=60, max_radius=5.75):
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.max_radius = max_radius

    def generate(self, points):
        """Convert a 3D point cloud into a 2D Scan Context matrix and a 1D Ring Key"""
        sc = np.zeros((self.num_rings, self.num_sectors))
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        radii = np.linalg.norm(np.column_stack((x, y)), axis=1)
        angles = np.degrees(np.arctan2(y, x)) + 180.0

        valid = radii < self.max_radius
        radii, angles, z = radii[valid], angles[valid], z[valid]

        ring_idx = np.clip(
            np.floor((radii / self.max_radius) * self.num_rings).astype(int),
            0,
            self.num_rings - 1,
        )
        sector_idx = np.clip(
            np.floor((angles / 360.0) * self.num_sectors).astype(int),
            0,
            self.num_sectors - 1,
        )

        for r, s, h in zip(ring_idx, sector_idx, z):
            if h > sc[r, s]:
                sc[r, s] = h

        ring_key = np.count_nonzero(sc, axis=1) / self.num_sectors
        return sc, ring_key

    def distance(self, sc1, sc2):
        """Calculate the column-shifted cosine distance between two Scan Context matrices"""
        min_dist = float("inf")
        norm1, norm2 = np.linalg.norm(sc1), np.linalg.norm(sc2)
        if norm1 == 0 or norm2 == 0:
            return 1.0

        for shift in range(self.num_sectors):
            sc1_shifted = np.roll(sc1, shift, axis=1)
            sim = np.sum(sc1_shifted * sc2) / (norm1 * norm2)
            dist = 1.0 - sim
            if dist < min_dist:
                min_dist = dist
        return min_dist

def load_kitti_bin(path):
    scan = np.fromfile(path, dtype=np.float32)
    return scan.reshape(-1, 4)[:, :3]

def load_kitti_poses(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            data = np.array([float(x) for x in line.strip().split()])
            poses.append(np.array([data[3], data[7], data[11]]))
    return np.array(poses)

def load_start_time(sequence_path):
    times_file = sequence_path / "times.txt"
    with open(times_file, "r") as f:
        return float(f.readline().strip())

def run_scan_context_benchmark(dataset_root: Path, scenario: str):
    """
    Run Scan Context benchmark and output recall stats to a csv.
    The benchmark uses the original Autowalk as the base map and queries it with all other sequences.
    """
    
    distance_threshold = 2.5
    csv_output_file = Path.cwd() / f"{scenario}_scan_context_benchmark_results.csv"
    max_radius = 5.75 if scenario == "indoor" else 50.0

    base_sequence = dataset_root / scenario / "autowalk" / "pseudo_gt" / "sequences" / "00"
    query_sequences_dir = dataset_root / scenario / "kitti" / "sequences"

    # build base database
    base_bins = sorted(glob.glob(str(base_sequence / "velodyne" / "*.bin")))
    base_poses_path = base_sequence.parent.parent / "poses" / f"{base_sequence.name}.txt"

    base_poses = load_kitti_poses(base_poses_path)
    base_start_time = load_start_time(base_sequence)

    sc_generator = ScanContext(num_rings=20, num_sectors=60, max_radius=max_radius)
    db_matrices = []
    db_ring_keys = []

    for path in base_bins:
        points = load_kitti_bin(path)
        sc, ring_key = sc_generator.generate(points)
        db_matrices.append(sc)
        db_ring_keys.append(ring_key)

    tree = cKDTree(db_ring_keys)

    # evaluate queries
    query_sequences = [d for d in query_sequences_dir.iterdir() if d.is_dir()]
    results = []

    for q_seq in query_sequences:
        seq_name = q_seq.name
        
        q_bins = sorted(glob.glob(str(q_seq / "velodyne" / "*.bin")))
        q_poses_path = q_seq.parent.parent / "poses" / f"{seq_name}.txt"

        if not q_bins or not q_poses_path.exists():
            continue

        query_poses = load_kitti_poses(q_poses_path)
        query_start_time = load_start_time(q_seq)
        
        time_gap_days = abs(query_start_time - base_start_time) / 86400.0
        
        total_queries = len(q_bins)
        r1_correct = 0
        r5_correct = 0

        for i, path in enumerate(q_bins):
            q_points = load_kitti_bin(path)
            q_sc, q_ring_key = sc_generator.generate(q_points)

            # retrieve top 10 candidates using ring keys
            _, indices = tree.query(q_ring_key, k=10)

            # re-ranking using exact distance
            candidates = []
            for idx in indices:
                dist = sc_generator.distance(q_sc, db_matrices[idx])
                candidates.append((dist, idx))

            candidates.sort(key=lambda x: x[0])
            top_5_indices = [c[1] for c in candidates[:5]]

            # recall@1 
            rank1_pose = base_poses[top_5_indices[0]]
            if np.linalg.norm(query_poses[i] - rank1_pose) <= distance_threshold:
                r1_correct += 1

            # recall@5 
            for idx in top_5_indices:
                match_pose = base_poses[idx]
                if np.linalg.norm(query_poses[i] - match_pose) <= distance_threshold:
                    r5_correct += 1
                    break

        recall_at_1 = (r1_correct / total_queries) * 100
        recall_at_5 = (r5_correct / total_queries) * 100

        results.append({
            "sequence": seq_name,
            "r1": recall_at_1,
            "r5": recall_at_5,
            "time_gap_days": time_gap_days,
            "total": total_queries,
        })
        
        print(f"Finished {seq_name} | Recall@1: {recall_at_1:.1f}% | Recall@5: {recall_at_5:.1f}%")

    # calculate final averages
    if results:
        avg_r1 = np.mean([r["r1"] for r in results])
        avg_r5 = np.mean([r["r5"] for r in results])
        print(f"\nFinal Average | Recall@1: {avg_r1:.1f}% | Recall@5: {avg_r5:.1f}%")

    # export summary
    results.sort(key=lambda x: x["time_gap_days"])

    with open(csv_output_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Sequence", "Time_Gap_Days", "Recall@1", "Recall@5", "Total_Scans"])
        for res in results:
            writer.writerow([
                res["sequence"],
                f"{res['time_gap_days']:.2f}",
                f"{res['r1']:.2f}",
                f"{res['r5']:.2f}",
                res["total"],
            ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Scan Context benchmark on the SALT dataset.")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Path to the SALT dataset root directory.")
    parser.add_argument("--scenario", type=str, required=True, choices=["indoor", "indoor_lidar", "outdoor"], help="The scenario to evaluate.")
    args = parser.parse_args()
    
    run_scan_context_benchmark(args.dataset_root, args.scenario)
