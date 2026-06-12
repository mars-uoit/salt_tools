import json
import argparse
from pathlib import Path
import open3d as o3d
import tifffile
from salt_helper import unpack_grid, unpack_point_cloud
from PIL import Image


def run_to_expanded_data(run_dir: Path, out_dir):
    """Extract pointclouds and grids from a localization run folder and save them in a human readable format"""

    # create directories
    run_name = run_dir.name
    output_folder = out_dir / run_name

    pc_folder = output_folder / "point_clouds"
    grids_folder = output_folder / "grids"

    pc_folder.mkdir(parents=True, exist_ok=True)
    grids_folder.mkdir(parents=True, exist_ok=True)

    for file_path in run_dir.glob("*.json"):
        waypoint_num = file_path.stem

        # open json
        with file_path.open("r") as f:
            data = json.load(f)

        live_data = data.get("liveData", {})

        # point clouds
        if "pointCloud" in live_data:
            pcd_array = unpack_point_cloud(live_data["pointCloud"])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_array)
            o3d.io.write_point_cloud(str(pc_folder / f"{waypoint_num}.ply"), pcd)

        # grids
        if "robotLocalGrids" in live_data:
            for grid in live_data["robotLocalGrids"]:
                grid_type = grid["localGridTypeName"]
                grid_type_folder = grids_folder / grid_type
                grid_type_folder.mkdir(parents=True, exist_ok=True)

                grid_array = unpack_grid(grid)
                ext = ".png" if grid["cellFormat"] == "CELL_FORMAT_UINT8" else ".tif"
                output_path = str(grid_type_folder / f"{waypoint_num}{ext}")

                if ext == ".png":
                    Image.fromarray(grid_array).save(output_path)
                else:
                    tifffile.imwrite(output_path, grid_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pointclouds and grids from a localization run folder."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        help="Path to the input localization response run folder.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Path to the output directory for all expanded data.",
    )
    args = parser.parse_args()

    run_to_expanded_data(args.run_dir, args.out_dir)
