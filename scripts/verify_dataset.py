import json
import argparse
from pathlib import Path


def verify_dataset_counts(base):
    """Checks the dataset to ensure all the correct number of files are downloaded"""

    # configure these based on what data you want to verify in the dataset
    scenarios = ["indoor", "indoor_lidar", "outdoor"]
    data_types = [
        "autowalk",
        "expanded_data",
        "kitti",
        "localization_response",
        "mcap",
    ]

    missing_dirs = []
    count_mismatches = []
    runs_checked = 0

    print(f"Checking for SALT Dataset files in: {base.resolve()}")

    # check each scenario
    for scn in scenarios:
        scn_dir = base / scn
        metadata_file = scn_dir / f"{scn}_metadata.json"

        print(f"Checking Environment: {scn}")

        # open the metadata file
        if not metadata_file.exists():
            print(f"Missing metadata file: {metadata_file}")
            continue

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Failed to parse {metadata_file.name}: {e}")
            continue

        # check each run
        for run_id, run_data in metadata.items():
            runs_checked += 1

            # pull expected number of waypoints.
            try:
                expected_count = run_data["autowalk_metrics"]["waypoints_recorded"]
            except KeyError:
                print(f"{run_id} is missing 'waypoints_recorded' in metadata.")
                continue

            kitti_seq = f"{int(run_id.split('_')[-1]):02d}"
            count_targets = []

            # setup directories that the number of files should be checked
            if "expanded_data" in data_types:
                count_targets.extend(
                    [
                        scn_dir / "expanded_data" / run_id / "grids" / "no_step",
                        scn_dir
                        / "expanded_data"
                        / run_id
                        / "grids"
                        / "obstacle_distance",
                        scn_dir / "expanded_data" / run_id / "grids" / "terrain",
                        scn_dir / "expanded_data" / run_id / "grids" / "terrain_valid",
                        scn_dir / "expanded_data" / run_id / "point_clouds",
                    ]
                )
            if "localization_response" in data_types:
                count_targets.extend([scn_dir / "localization_response" / run_id])
            if "kitti" in data_types:
                count_targets.extend(
                    [scn_dir / "kitti" / "sequences" / kitti_seq / "velodyne"]
                )

            for target_dir in count_targets:
                # check if the directory exists
                if not target_dir.exists():
                    missing_dirs.append(target_dir)
                elif target_dir.is_dir():
                    # count files
                    actual_count = sum(
                        1 for item in target_dir.iterdir() if item.is_file()
                    )
                    # compare the number of files with expected number of waypoints in metadata
                    if actual_count != expected_count:
                        count_mismatches.append(
                            f"{target_dir}\n Expected: {expected_count}, Found: {actual_count}"
                        )

            # check MCAP files
            if "mcap" in data_types:
                mcap_file = scn_dir / "mcap" / f"{run_id}.mcap"
                if not mcap_file.exists():
                    missing_dirs.append(mcap_file)

            # check kitti poses
            if "kitti" in data_types:
                kitti_poses = scn_dir / "kitti" / "poses" / f"{kitti_seq}.txt"
                if not kitti_poses.exists():
                    missing_dirs.append(kitti_poses)
                kitti_vision_poses = (
                    scn_dir / "kitti" / "vision_poses" / f"{kitti_seq}.txt"
                )
                if not kitti_vision_poses.exists():
                    missing_dirs.append(kitti_vision_poses)
                kitti_odom_poses = scn_dir / "kitti" / "odom_poses" / f"{kitti_seq}.txt"
                if not kitti_odom_poses.exists():
                    missing_dirs.append(kitti_odom_poses)

        # check autowalk folder
        if "autowalk" in data_types:

            aw_dir = scn_dir / "autowalk"

            # check graph.json
            graph_file = aw_dir / "graph.json"
            if not graph_file.exists():
                missing_dirs.append(graph_file)

            # check gt_cloud
            gt_cloud = aw_dir / "pseudo_gt" / "cloud.ply"
            if not gt_cloud.exists():
                missing_dirs.append(gt_cloud)

            # check gt_poses
            gt_cloud = aw_dir / "pseudo_gt" / "poses" / "00.txt"
            if not gt_cloud.exists():
                missing_dirs.append(gt_cloud)

            wp_total_num = run_data["autowalk_metrics"]["waypoints_recorded"] + len(
                run_data["autowalk_metrics"]["missed_waypoints"]
            )

            # autowalk directories to check the number of files in
            count_targets = [
                aw_dir / "waypoint_snapshots",
                aw_dir / "pseudo_gt" / "sequences" / "00" / "velodyne",
                aw_dir / "edge_snapshots",
            ]

            for target_dir in count_targets:
                if not target_dir.exists():
                    missing_dirs.append(target_dir)
                elif target_dir.is_dir():
                    # count files
                    actual_count = sum(
                        1 for item in target_dir.iterdir() if item.is_file()
                    )
                    if target_dir == aw_dir / "edge_snapshots":
                        if scn == "indoor":
                            check_count = 114
                        elif scn == "indoor_lidar":
                            check_count = 104
                        elif scn == "outdoor":
                            check_count = 424
                    else:
                        check_count = wp_total_num
                    # compare the number of files with expected number of waypoints in metadata
                    if actual_count != check_count:
                        count_mismatches.append(
                            f"{target_dir}\n Expected: {check_count}, Found: {actual_count}"
                        )

    print(f"Verification complete. Checked {runs_checked} runs.")

    if not missing_dirs and not count_mismatches:
        print("All directories exist and file counts are exactly as expected.")
    else:
        if missing_dirs:
            print(f"\n[!] {len(missing_dirs)} Missing Directories/Files:")
            for md in missing_dirs[:10]:
                print(f"  - {md}")
            if len(missing_dirs) > 10:
                print(f"  ... and {len(missing_dirs) - 10} more.")

        if count_mismatches:
            print(f"\n{len(count_mismatches)} File Count Mismatches:")
            for cm in count_mismatches[:15]:
                print(f"{cm}")
            if len(count_mismatches) > 15:
                print(f"... and {len(count_mismatches) - 15} more issues.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify dataset file counts against the metadata files."
    )
    parser.add_argument(
        "--base_dir",
        nargs="?",
        type=Path,
        help="Path to the base SALT dataset root folder",
    )
    args = parser.parse_args()

    verify_dataset_counts(args.base_dir)
