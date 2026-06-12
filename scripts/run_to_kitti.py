import argparse
import open3d as o3d
import numpy as np
import json
import re
from pathlib import Path
from salt_helper import tfdict_to_tfmatrix, unpack_point_cloud, iso_to_nanoseconds

# spot to kitti tf
CALIB_TRANSFORM = np.array(
    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float
)
CALIB_INV = np.linalg.inv(CALIB_TRANSFORM)


def kitti_extended(run_dir: Path, out_dir: Path):
    json_files = sorted(run_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {run_dir}")
        return

    # open first json
    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    # get initial seed to body offsets
    seed_tf_body0 = tfdict_to_tfmatrix(data["localization"]["seedTformBody"])

    tree0 = data["liveData"]["pointCloud"]["source"]["transformsSnapshot"][
        "childToParentEdgeMap"
    ]

    # calculate initial seed to frame
    body_tf_odom0 = tfdict_to_tfmatrix(tree0["odom"])
    body_tf_vision0 = tfdict_to_tfmatrix(tree0["vision"])

    seed_tf_odom0 = seed_tf_body0 @ body_tf_odom0
    seed_tf_vision0 = seed_tf_body0 @ body_tf_vision0

    # setup output directories
    vision_poses_dir = out_dir / "vision_poses"
    odom_poses_dir = out_dir / "odom_poses"
    vision_poses_dir.mkdir(parents=True, exist_ok=True)
    odom_poses_dir.mkdir(parents=True, exist_ok=True)

    # get the run number
    match = re.search(r"\d+", run_dir.name)
    run_name = f"{int(match.group()):02d}" if match else "poses"

    vision_poses_path = vision_poses_dir / f"{run_name}.txt"
    odom_poses_path = odom_poses_dir / f"{run_name}.txt"

    # save the files as a KITTI format.
    with open(vision_poses_path, "w", encoding="utf-8") as vf, open(
        odom_poses_path, "w", encoding="utf-8"
    ) as of:

        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            edges = data["liveData"]["pointCloud"]["source"]["transformsSnapshot"][
                "childToParentEdgeMap"
            ]

            # get body to frame
            T_body_odom = tfdict_to_tfmatrix(edges["odom"])
            T_body_vision = tfdict_to_tfmatrix(edges["vision"])

            # invert tfs
            odom_tf_body = np.linalg.inv(T_body_odom)
            vision_tf_body = np.linalg.inv(T_body_vision)

            # seed to body
            seed_tf_vision = seed_tf_vision0 @ vision_tf_body
            seed_tf_odom = seed_tf_odom0 @ odom_tf_body

            # fotate from Spot's XY plane to KITTI's XZ plane
            kitti_vision = CALIB_TRANSFORM @ seed_tf_vision @ CALIB_INV
            kitti_odom = CALIB_TRANSFORM @ seed_tf_odom @ CALIB_INV

            # flatten top 3x4 matrices into KITTI row-major format
            v_kitti = " ".join([f"{x:.6e}" for x in kitti_vision[:3, :].flatten()])
            o_kitti = " ".join([f"{x:.6e}" for x in kitti_odom[:3, :].flatten()])

            vf.write(v_kitti + "\n")
            of.write(o_kitti + "\n")


def run_to_kitti(
    run_dir: Path,
    out_dir: Path,
    gt_map: o3d.geometry.PointCloud,
    global_poses: np.ndarray,
    debug: bool,
    p2plane: bool,
):
    try:
        seq_id_str = f"{int(run_dir.name.split('_')[-1]):02d}"
    except ValueError:
        seq_id_str = run_dir.name

    print(f"\n--- Processing {run_dir.name} (KITTI Seq {seq_id_str}) ---")

    json_files = sorted(run_dir.glob("*.json"), key=lambda x: int(x.stem))
    if not json_files:
        print(f"Error: Localization folder is empty.")
        return

    # setup directories
    seq_dir = out_dir / "sequences" / seq_id_str
    velo_dir = seq_dir / "velodyne"
    velo_dir.mkdir(parents=True, exist_ok=True)

    poses_dir = out_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=True)
    out_poses_file = poses_dir / f"{seq_id_str}.txt"
    times_file = seq_dir / "times.txt"

    # write the calibration file
    with open(seq_dir / "calib.txt", "w") as f:
        f.write(
            "P0: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0\n"
            "P1: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0\n"
            "P2: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0\n"
            "P3: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0\n"
            "Tr: 0.0 -1.0 0.0 0.0 0.0 0.0 -1.0 0.0 1.0 0.0 0.0 0.0\n"
        )

    f_poses = open(out_poses_file, "w")
    f_times = open(times_file, "w")

    prev_vis_pose = None
    curr_global_pose = None
    prev_target_idx = None

    vis = None
    merged_cloud = None

    # if debug then visualize the new point clouds vs the ground truth
    if debug:

        # pc vis setup
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Live Tracking: {seq_id_str}", width=1280, height=720
        )
        vis.add_geometry(gt_map)
        merged_cloud = o3d.geometry.PointCloud()
        palette = [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.5, 1.0],
            [1.0, 1.0, 0.2],
            [0.2, 1.0, 1.0],
            [1.0, 0.2, 1.0],
        ]

    # loop through waypoints
    for i, json_file in enumerate(json_files):
        target_idx = int(json_file.stem)

        # open localization response
        with open(json_file, "r", encoding="utf-8") as file:
            loc_dict = json.load(file)

        # open point cloud
        pc_data = unpack_point_cloud(loc_dict["liveData"]["pointCloud"])
        valid_mask = np.isfinite(pc_data[:, :3]).all(axis=1)
        clean_points = pc_data[valid_mask, :3]

        frame_pc = o3d.geometry.PointCloud()
        frame_pc.points = o3d.utility.Vector3dVector(clean_points)

        # recolour the cloud to see them being added better
        if debug:
            frame_pc.colors = o3d.utility.Vector3dVector()
            frame_pc.paint_uniform_color(palette[i % len(palette)])

        vis_dict = loc_dict["liveData"]["pointCloud"]["source"]["transformsSnapshot"][
            "childToParentEdgeMap"
        ]["vision"]
        vis_mat = tfdict_to_tfmatrix(vis_dict)
        vis_pose = np.linalg.inv(vis_mat)

        # determine initial pose guess
        # reset to GT if it's the first frame OR if the waypoint gap > 30
        if i == 0 or (
            prev_target_idx is not None and (target_idx - prev_target_idx) > 30
        ):
            kitti_base = np.eye(4)
            kitti_base[:3, :4] = global_poses[target_idx].reshape(3, 4)
            init_global_pose = CALIB_INV @ kitti_base @ CALIB_TRANSFORM
        else:
            delta_vis = np.linalg.inv(prev_vis_pose) @ vis_pose
            init_global_pose = curr_global_pose @ delta_vis

        frame_pc.transform(init_global_pose)

        # icp alignment
        center = init_global_pose[:3, 3]
        bbox = o3d.geometry.AxisAlignedBoundingBox(center - 30.0, center + 30.0)
        local_gt = gt_map.crop(bbox)

        # use point 2 plane icp
        if p2plane:

            frame_pc.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
            )

            loss_coarse = o3d.pipelines.registration.TukeyLoss(k=0.65)
            reg_coarse = o3d.pipelines.registration.registration_icp(
                frame_pc,
                local_gt,
                3,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    loss_coarse
                ),
            )
            frame_pc.transform(reg_coarse.transformation)

            loss_fine = o3d.pipelines.registration.TukeyLoss(k=0.1)
            reg_fine = o3d.pipelines.registration.registration_icp(
                frame_pc,
                local_gt,
                0.7,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    loss_fine
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
                ),
            )
            frame_pc.transform(reg_fine.transformation)
        else:

            # use point 2 point icp
            reg_coarse = o3d.pipelines.registration.registration_icp(
                frame_pc,
                local_gt,
                3,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            frame_pc.transform(reg_coarse.transformation)

            reg_fine = o3d.pipelines.registration.registration_icp(
                frame_pc,
                local_gt,
                0.7,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
                ),
            )
            frame_pc.transform(reg_fine.transformation)

        # update the poses
        total_correction = reg_fine.transformation @ reg_coarse.transformation
        curr_global_pose = total_correction @ init_global_pose
        prev_vis_pose = vis_pose
        prev_target_idx = target_idx

        # add to cloud to visualize
        if debug:
            merged_cloud += frame_pc
            if (
                i == 0 or (target_idx - prev_target_idx) > 30
            ):  # Reset bounding box on jump
                vis.add_geometry(frame_pc, reset_bounding_box=True)
            else:
                vis.add_geometry(frame_pc, reset_bounding_box=False)

        # write poses
        kitti_out_tf = CALIB_TRANSFORM @ curr_global_pose @ CALIB_INV
        pose_row = kitti_out_tf[:3, :4].flatten()
        f_poses.write(" ".join([f"{x:.6e}" for x in pose_row]) + "\n")

        # write times
        iso_str = loc_dict["liveData"]["pointCloud"]["source"]["acquisitionTime"]
        time_sec = iso_to_nanoseconds(iso_str) / 1e9

        f_times.write(f"{time_sec:.6f}\n")

        # write .bin files
        valid_points = pc_data[valid_mask]
        if valid_points.shape[1] >= 4:
            bin_data = valid_points[:, :4].astype(np.float32)
        else:
            bin_data = np.hstack(
                (valid_points[:, :3], np.zeros((len(valid_points), 1)))
            ).astype(np.float32)

        bin_data.tofile(str(velo_dir / f"{i:06d}.bin"))

        # visualize
        if debug:
            vis.poll_events()
            vis.update_renderer()
            if i % 10 == 0:
                merged_cloud = merged_cloud.voxel_down_sample(voxel_size=0.1)

    f_poses.close()
    f_times.close()

    print(f"Completed run {seq_id_str} with {len(json_files)} scans")

    if debug:
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Spot localization response run to KITTI odometry format."
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        help="Path to the localization response run directory",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Base KITTI output directory",
    )
    parser.add_argument(
        "--gt_map",
        type=Path,
        help="Path to ground truth point cloud .ply file",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        help="Path to ground truth poses.txt",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable live Open3D visualization"
    )

    parser.add_argument(
        "--p2plane",
        action="store_true",
        help="Point to plane ICP instead of point to point",
    )

    parser.add_argument(
        "--extended",
        action="store_true",
        help="Stores the vision and odom poses",
    )

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory '{args.run_dir}' does not exist.")
        exit(1)

    print("--- Initialization ---")
    print(f"Loading Ground Truth Map: {args.gt_map.name}")
    global_gt_map = o3d.io.read_point_cloud(str(args.gt_map))
    global_gt_map = global_gt_map.voxel_down_sample(voxel_size=0.1)
    global_gt_map.paint_uniform_color([0.5, 0.5, 0.5])

    print("Pre-computing GT Normals...")
    global_gt_map.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    )

    print(f"Loading Base Poses: {args.poses.name}")
    global_kitti_poses = np.loadtxt(args.poses)

    run_to_kitti(
        args.run_dir,
        args.out_dir,
        global_gt_map,
        global_kitti_poses,
        args.debug,
        args.p2plane,
    )

    if args.extended:
        kitti_extended(args.run_dir, args.out_dir)
