import os
import sys
import csv
import glob
import argparse
import torch
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from models.model_factory import model_factory
from misc.utils import ModelParams

class MinkLoc3DExtractor:
    """
    Loads a pretrained MinkLoc3Dv2 model and generates global descriptors from point clouds.
    Reference: MinkLoc3Dv2: Point Cloud Based Place Recognition (Komorowski 2022)
    """
    def __init__(self, weights_path, repo_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if repo_path not in sys.path:
            sys.path.append(repo_path)
            
        config_path = os.path.join(repo_path, "models", "minkloc3dv2.txt")
        model_params = ModelParams(config_path)
        
        self.model = model_factory(model_params)
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        
        self.voxel_size = 0.05 

    def generate(self, points):
        """Quantizes the point cloud and passes it through the MinkLoc3D sparse tensor network"""
        quantized_coords = np.floor(points[:, :3] / self.voxel_size).astype(np.int32)
        unique_coords = np.unique(quantized_coords, axis=0)
        
        features = np.ones((len(unique_coords), 1), dtype=np.float32)
        
        coords_tensor = torch.tensor(unique_coords, dtype=torch.int32)
        batch_indices = torch.zeros((len(coords_tensor), 1), dtype=torch.int32)
        batched_coords = torch.cat([batch_indices, coords_tensor], dim=1)
        
        features_tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            feed_dict = {
                'coords': batched_coords.to(self.device),
                'features': features_tensor.to(self.device)
            }
            descriptor = self.model(feed_dict)
            
        return descriptor['global'].cpu().numpy().flatten()
    
def load_kitti_bin(path):
    scan = np.fromfile(path, dtype=np.float32)
    return scan.reshape(-1, 4)[:, :3]

def load_kitti_poses(path):
    poses = []
    with open(path, 'r') as f:
        for line in f:
            data = np.array([float(x) for x in line.strip().split()])
            poses.append(np.array([data[3], data[7], data[11]]))
    return np.array(poses)

def load_start_time(sequence_path):
    times_file = sequence_path / "times.txt"
    with open(times_file, "r") as f:
        return float(f.readline().strip())


def run_minkloc_benchmark(dataset_root: Path, scenario: str, weights: str, minkloc_dir: str):
    """
    Run MinkLoc3Dv2 benchmark and output recall stats to a csv.
    The benchmark uses the original Autowalk as the base map and queries it with all other sequences.
    """

    distance_threshold = 2.5
    csv_output_file = Path.cwd() / f"{scenario}_minkloc_benchmark_results.csv"
    
    base_sequence = dataset_root / scenario / "autowalk" / "pseudo_gt" / "sequences" / "00"
    query_sequences_dir = dataset_root / scenario / "kitti" / "sequences"

    # build base database
    base_bins = sorted(glob.glob(str(base_sequence / "velodyne" / "*.bin")))
    base_poses_path = base_sequence.parent.parent / "poses" / f"{base_sequence.name}.txt"

    base_poses = load_kitti_poses(base_poses_path)
    base_start_time = load_start_time(base_sequence)

    extractor = MinkLoc3DExtractor(weights, minkloc_dir)
    db_descriptors = []

    for path in base_bins:
        points = load_kitti_bin(path)
        desc = extractor.generate(points)
        db_descriptors.append(desc)

    tree = cKDTree(db_descriptors)

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
            q_descriptor = extractor.generate(q_points)

            # retrieve top 5 nearest neighbors
            _, best_match_indices = tree.query(q_descriptor, k=5)

            q_pose = query_poses[i]
            
            # recall@1 
            rank1_pose = base_poses[best_match_indices[0]]
            if np.linalg.norm(q_pose - rank1_pose) <= distance_threshold:
                r1_correct += 1

            # recall@5 
            for idx in best_match_indices:
                match_pose = base_poses[idx]
                if np.linalg.norm(q_pose - match_pose) <= distance_threshold:
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
                res["total"]
            ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MinkLoc3Dv2 benchmark on the SALT dataset.")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Path to the SALT dataset root directory.")
    parser.add_argument("--scenario", type=str, required=True, choices=["indoor", "indoor_lidar", "outdoor"], help="The scenario to evaluate.")
    parser.add_argument("--weights", type=str, default=os.path.expanduser("~/MinkLoc3Dv2/weights/minkloc3dv2_refined.pth"), help="Path to the trained MinkLoc3Dv2 weights.")
    parser.add_argument("--minkloc_dir", type=str, default=os.path.expanduser("~/MinkLoc3Dv2"), help="Path to the MinkLoc3Dv2 repository.")
    args = parser.parse_args()

    run_minkloc_benchmark(args.dataset_root, args.scenario, args.weights, args.minkloc_dir)