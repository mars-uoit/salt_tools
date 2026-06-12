import numpy as np
import base64
from scipy.spatial.transform import Rotation
import datetime


def unpack_grid(grid_dict):
    """Unpacks a single grid data from a localization response file"""

    # Decode the local grid.
    if grid_dict["cellFormat"] == "CELL_FORMAT_INT16":
        data_type = np.int16
    elif grid_dict["cellFormat"] == "CELL_FORMAT_UINT8":
        data_type = np.uint8

    full_grid = np.repeat(
        np.frombuffer(base64.b64decode(grid_dict["data"]), dtype=data_type),
        grid_dict["rleCounts"],
    )

    # reshape the full grid to the correct dimensions
    full_grid = np.reshape(
        full_grid, (grid_dict["extent"]["numCellsX"], grid_dict["extent"]["numCellsY"])
    )

    if grid_dict["cellFormat"] == "CELL_FORMAT_UINT8":
        full_grid_uint8 = 255 * full_grid.astype(np.uint8)
        return full_grid_uint8

    # Apply the offset and scaling to the local grid.
    elif grid_dict["cellFormat"] == "CELL_FORMAT_INT16":
        full_grid_float = full_grid.astype(np.float32)
        if "cellValueScale" in grid_dict:
            full_grid_float *= grid_dict["cellValueScale"]
        if "cellValueOffset" in grid_dict:
            full_grid_float += grid_dict["cellValueOffset"]
        return full_grid_float


def unpack_point_cloud(pc_dict):
    """Unpacks the point cloud data from a localization response"""

    raw_bytes = base64.b64decode(pc_dict["data"])
    pcd = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, 3)

    return pcd


def tfdict_to_tfmatrix(parent_tf_child):
    """Converts a localization response transform to a 4x4 matrix"""

    # unpack the tf data
    if "parentTformChild" in parent_tf_child:
        pos_dict = parent_tf_child["parentTformChild"]["position"]
        rot_dict = parent_tf_child["parentTformChild"]["rotation"]
    else:
        pos_dict = parent_tf_child["position"]
        rot_dict = parent_tf_child["rotation"]

    # ensure elements exist
    translation = [pos_dict.get("x", 0), pos_dict.get("y", 0), pos_dict.get("z", 0)]
    quaternion = [
        rot_dict.get("x", 0),
        rot_dict.get("y", 0),
        rot_dict.get("z", 0),
        rot_dict.get("w", 1),
    ]

    # convert to 4x4 matrix
    matrix = np.eye(4)
    matrix[0:3, 0:3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[0:3, 3] = translation

    return matrix


def iso_to_nanoseconds(iso_str):
    """Formats Spot's timestamps to a nanosecond format"""

    iso_str = iso_str.replace("Z", "")
    time_part, fractional_part = iso_str.split(".")
    dt = datetime.datetime.fromisoformat(time_part).replace(
        tzinfo=datetime.timezone.utc
    )
    seconds = int(dt.timestamp())
    nanoseconds_part = int((fractional_part + "000000000")[:9])
    total_ns = (seconds * 10**9) + nanoseconds_part

    return total_ns
