# salt_tools

Basic tools and conversion scripts to interface with the Spot Autowalk Long-Term (SALT) Dataset including tools for data management and data conversions.

## Abstract

The SALT Dataset was collected performing repeated Autowalks with a Boston Dynamics Spot robot across three different scenarios including both indoor and outdoor environments. The main contribution is 217 traversals of an outdoor Autowalk around the Ontario Tech University campus over a 16 month period. In total the dataset contains 286 traversals and 194 km of walking over a period of nearly two years. The data is presented in multiple formats for ease of use including: native format, a human readable format, a KITTI Odometry format including pseudo ground truth poses, and an MCAP format for use with ROS2. This dataset aims to enable researchers to develop new algorithms for long-term mapping applications.

A compilation video of one year of outdoor runs is available [Watch on YouTube](https://www.youtube.com/watch?v=Nof6R7y4wDE).

## Setup

The dataset was developed and tested using Python 3.10. Using pip all necessary packages to run any of the examples can be installed using:

```
python -m pip install -r requirements.txt
```

## Scripts Overview

### Visualization

For quick visualization a Foxglove layout (salt_foxglove_layout.json) is included. You can import this layout into Foxglove to instantly visualize the generated .mcap files including point clouds, elevation and occupancy girds, TF trees, battery conditions, and joint tracking for the [Spot URDF](https://github.com/rai-opensource/spot_description).

### Dataset Management

* **unzip_dataset.py -** Extracts the multiple downloaded zip files that make up the dataset into a single unified dataset directory structure.

  ```
  python scripts/unzip_dataset.py --zip_dir /path/to/zips --extract_to /path/to/dataset
  ```
* **verify_dataset.py -** Validates the extracted dataset's file counts, missing directories, and file structure against the metadata.json files. The script can be customized to only check certain datatypes or scenarios.

  ```
  python scripts/verify_dataset.py --base_dir /path/to/spot_autowalk_long_term_dataset
  ```

## Data Conversion

The natively recorded data is the localization response files and the Autowalk files. Separate data types for the same data is provided in SALT for ease of use. The scripts used to convert the data are available here to allow users to customize the data if needed.

* **run_to_expanded_data.py -** Extracts the point clouds (.ply) and local grids (.tif/.png) from the localization response run folder into a more human readable format.

  ```
  python scripts/run_to_expanded_data.py --run_dir /path/to/localization_response/run_XXX --out_dir /path/to/output
  ```
* **run_to_kitti.py -**  Converts a localization response run to KITTI Odometry format. Uses ICP and odometry to align a run with a pre-existing ground truth map. Does not validate that the poses are correctly aligned, a user must do that manually. Provides a path to the completed ground truth point cloud and poses.txt from the original Autowalk. Set --debug if you want to visualize the poses and point cloud assembling. Set --extended to include vision and odom odometry topics as KITTI poses for comparison to Spot's native odometry.

  ```
  python scripts/run_to_kitti.py --run_dir /path/to/localization_response/run_XXX --out_dir /path/to/output --gt_map /path/to/autowalk/pseudo_gt/cloud.ply --poses /path/autowalk/pseudo_gt/poses/00.txt  --debug --extended
  ```
* **run_to_mcap.py -** Converts as much data from a localization response run to MCAP ROS2 Humble format as possible with standard message types. The script was written without the need for a ROS2 installation allowing easier use for people not using ROS2. A KITTI poses.txt file can be provided to add a ground truth pose. MCAP files can be visualized quickly in Foxglove using the salt_foxglove_layout.json.

  ```
  python scripts/run_to_mcap.py --run_dir /path/to/localization_response/run_XXX --out_dir /path/to/output --kitti_poses /path/to/kitti/poses/XXX.txt
  ```

## Utility

* **salt_helper.py -** Used to convert Spot data from the localization response files into more conventional formats. Data conversions include: transforms, timestamps, point clouds, and grids.

## Citation

If you find this work useful, please consider citing our paper:

UNDER REVIEW

## Disclaimer

This software is provided as a research prototype and is not production-quality software. Please note that the code may contain missing features, bugs, and errors. The Mechatronic and Robotic Systems Laboratory does not offer maintenance or support for this software.
