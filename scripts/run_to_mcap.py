from pathlib import Path
import argparse
import logging
import json
import numpy as np
import math
from scipy.spatial.transform import Rotation
from mcap.writer import Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from salt_helper import (
    tfdict_to_tfmatrix,
    unpack_grid,
    iso_to_nanoseconds,
    unpack_point_cloud,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(asctime)s]: %(message)s",
    datefmt="%H:%M:%S",
)

# Gridmap Msg Schema Text
GRID_MAP_SCHEMA_TEXT = """
grid_map_msgs/GridMapInfo info
string[] layers
string[] basic_layers
std_msgs/Float32MultiArray[] data
uint16 outer_start_index
uint16 inner_start_index
================================================================================
MSG: grid_map_msgs/GridMapInfo
std_msgs/Header header
float64 resolution
float64 length_x
float64 length_y
geometry_msgs/Pose pose"""


STATICTF_LATCHING_QOS = [
    {
        "history": 1,
        "depth": 1,
        "reliability": 1,
        "durability": 1,
        "deadline": {"sec": 0, "nsec": 0},
        "lifespan": {"sec": 0, "nsec": 0},
        "liveliness": 1,
        "liveliness_lease_duration": {"sec": 0, "nsec": 0},
        "avoid_ros_namespace_conventions": False,
    }
]


# setup the typestore and adding the gridmaps message
typestore = get_typestore(Stores.ROS2_HUMBLE)
grid_map_types = get_types_from_msg(GRID_MAP_SCHEMA_TEXT, "grid_map_msgs/msg/GridMap")
typestore.register({**grid_map_types})


# region All the types of messages used
TFMessage = typestore.types["tf2_msgs/msg/TFMessage"]
TransformStamped = typestore.types["geometry_msgs/msg/TransformStamped"]
Transform = typestore.types["geometry_msgs/msg/Transform"]
Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
Header = typestore.types["std_msgs/msg/Header"]
Time = typestore.types["builtin_interfaces/msg/Time"]
JointState = typestore.types["sensor_msgs/msg/JointState"]
PointCloud2 = typestore.types["sensor_msgs/msg/PointCloud2"]
PointField = typestore.types["sensor_msgs/msg/PointField"]
GridMap = typestore.types["grid_map_msgs/msg/GridMap"]
GridMapInfo = typestore.types["grid_map_msgs/msg/GridMapInfo"]
Float32MultiArray = typestore.types["std_msgs/msg/Float32MultiArray"]
MultiArrayLayout = typestore.types["std_msgs/msg/MultiArrayLayout"]
MultiArrayDimension = typestore.types["std_msgs/msg/MultiArrayDimension"]
Pose = typestore.types["geometry_msgs/msg/Pose"]
Point = typestore.types["geometry_msgs/msg/Point"]
Twist = typestore.types["geometry_msgs/msg/Twist"]
TwistStamped = typestore.types["geometry_msgs/msg/TwistStamped"]
ColorRGBA = typestore.types["std_msgs/msg/ColorRGBA"]
Duration = typestore.types["builtin_interfaces/msg/Duration"]
MarkerArray = typestore.types["visualization_msgs/msg/MarkerArray"]
Marker = typestore.types["visualization_msgs/msg/Marker"]
NavPath = typestore.types["nav_msgs/msg/Path"]
PoseStamped = typestore.types["geometry_msgs/msg/PoseStamped"]
BatteryState = typestore.types["sensor_msgs/msg/BatteryState"]
MeshFile = typestore.types["visualization_msgs/msg/MeshFile"]
UVCoordinate = typestore.types["visualization_msgs/msg/UVCoordinate"]
CompressedImage = typestore.types["sensor_msgs/msg/CompressedImage"]
Imu = typestore.types["sensor_msgs/msg/Imu"]
# endregion


# ros2 channel configs (topic name, message type)
CHANNEL_CONFIG = {
    "pc": ("/spot/point_cloud", "sensor_msgs/msg/PointCloud2"),
    "tf_static": ("/tf_static", "tf2_msgs/msg/TFMessage"),
    "tf": ("/tf", "tf2_msgs/msg/TFMessage"),
    "grid": ("/spot/grids", "grid_map_msgs/msg/GridMap"),
    "joint": ("/spot/joint_states", "sensor_msgs/msg/JointState"),
    "vel_odom": ("/spot/velocity/odom", "geometry_msgs/msg/TwistStamped"),
    "vel_vision": ("/spot/velocity/vision", "geometry_msgs/msg/TwistStamped"),
    "blobs": ("/spot/tracked_markers", "visualization_msgs/msg/MarkerArray"),
    "stairs": (
        "/spot/staircase_markers",
        "visualization_msgs/msg/MarkerArray",
    ),
    "tag": (
        "/spot/apriltags",
        "visualization_msgs/msg/MarkerArray",
    ),
    "path": ("/spot/path", "nav_msgs/msg/Path"),
    "battery": ("/spot/battery", "sensor_msgs/msg/BatteryState"),
    "gravity": ("/spot/gravity_imu", "sensor_msgs/msg/Imu"),
    "gt": ("/ground_truth", "geometry_msgs/msg/PoseStamped"),
}


def run_to_mcap(run_dir: Path, out_dir: Path, kitti_pose_path: Path):
    spot_path_list = []

    mcap_file_path = out_dir / (run_dir.name + ".mcap")

    with open(mcap_file_path, "wb") as f:
        writer = Writer(f)
        writer.start()

        channels = register_channels(writer)

        # open the first waypoint localization response json
        loc_files = sorted(run_dir.iterdir())

        if not loc_files:
            logging.error(f"No localization response files found in {run_dir}")
            return

        first_loc_resp_path = loc_files[0]

        with open(
            first_loc_resp_path,
            "r",
            encoding="utf-8",
        ) as file:
            first_loc_resp = json.load(file)

        # starting transforms from the seed (map) to vision
        loc_dict = first_loc_resp["localization"]["seedTformBody"]
        first_state_tf = first_loc_resp["liveData"]["robotState"]["kinematicState"][
            "transformsSnapshot"
        ]["childToParentEdgeMap"]
        starting_timestamp = iso_to_nanoseconds(
            first_loc_resp["localization"]["timestamp"]
        )
        tf_msg = create_map_tf_msg(loc_dict, first_state_tf, starting_timestamp)
        writer.add_message(
            channel_id=channels["tf_static"],
            log_time=starting_timestamp,
            data=typestore.serialize_cdr(tf_msg, tf_msg.__msgtype__),
            publish_time=starting_timestamp,
        )

        # set the body to base_link static transform
        start_time_msg = create_time_msg(starting_timestamp)

        body_stf_base_link = TFMessage(
            transforms=[
                TransformStamped(
                    header=Header(stamp=start_time_msg, frame_id="body"),
                    child_frame_id="base_link",
                    transform=Transform(
                        translation=Vector3(x=0.0, y=0.0, z=0.0),
                        rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                )
            ]
        )

        write_msg(writer, channels["tf_static"], starting_timestamp, body_stf_base_link)

        # publish the kitti ground truth poses if provided
        if kitti_pose_path:

            kitti_ground_truth(writer, channels, kitti_pose_path)

            map_stf_map_gt = TFMessage(
                transforms=[
                    TransformStamped(
                        header=Header(stamp=start_time_msg, frame_id="map"),
                        child_frame_id="map_gt",
                        transform=Transform(
                            translation=Vector3(x=0.0, y=0.0, z=0.0),
                            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                        ),
                    )
                ]
            )

            write_msg(writer, channels["tf_static"], starting_timestamp, map_stf_map_gt)

        # loop through all the run files
        for localization_response_path in loc_files:

            # open localization response
            with open(
                localization_response_path,
                "r",
                encoding="utf-8",
            ) as file:
                loc_resp = json.load(file)

            process_localization_file(loc_resp, writer, channels, spot_path_list)

        writer.finish()


def process_localization_file(loc_resp, writer, channels, spot_path_list):
    """Processes a localization file and writes all the required information to the mcap file"""

    live_data = loc_resp["liveData"]

    # point clouds
    pc_dict = live_data["pointCloud"]
    pc_timestamp = iso_to_nanoseconds(pc_dict["source"]["acquisitionTime"])
    pc_msg = create_pointcloud2_msg(pc_dict, pc_timestamp)
    write_msg(writer, channels["pc"], pc_timestamp, pc_msg)

    # tf for pointcloud
    pc_tfsnap_dict = pc_dict["source"]["transformsSnapshot"]["childToParentEdgeMap"]
    pctf_msg = create_pc_tf_msg(pc_tfsnap_dict, pc_timestamp)
    write_msg(writer, channels["tf"], pc_timestamp, pctf_msg)

    # grid data
    grids_dict = live_data["robotLocalGrids"]
    grid_timestamp = iso_to_nanoseconds(grids_dict[0]["acquisitionTime"])
    grid_msg = create_gridmap_msg(grids_dict, grid_timestamp)
    write_msg(writer, channels["grid"], grid_timestamp, grid_msg)

    # tf for grid data
    grid_tfsnap_dict = grids_dict[0]["transformsSnapshot"]["childToParentEdgeMap"]
    gridtf_msg = create_grid_tf_msg(grid_tfsnap_dict, grid_timestamp)
    write_msg(writer, channels["tf"], grid_timestamp, gridtf_msg)

    # tracked world objects
    if "objects" in live_data:
        for obj_dict in live_data["objects"]:
            obj_timestamp = iso_to_nanoseconds(obj_dict["acquisitionTime"])
            # fiducials
            if "apriltagProperties" in obj_dict:
                tag_msg = create_apriltag_marker_msg(obj_dict, obj_timestamp)
                write_msg(writer, channels["tag"], obj_timestamp, tag_msg)
            # staircases
            elif "staircaseProperties" in obj_dict:
                stair_msg = create_staircase_msg(obj_dict, obj_timestamp)
                write_msg(writer, channels["stairs"], obj_timestamp, stair_msg)
            # tracked moving objects
            elif "trackedEntityProperties" in obj_dict:
                # save the blobs as a box with a velocity vector
                marker_msg = create_marker_objects_msg(obj_dict, obj_timestamp)
                write_msg(writer, channels["blobs"], obj_timestamp, marker_msg)

    # kinematics
    kin_obj = live_data["robotState"]["kinematicState"]
    kin_timestamp = iso_to_nanoseconds(kin_obj["acquisitionTimestamp"])

    # joint_states
    joint_msg = create_joint_state_msg(kin_obj["jointStates"], kin_timestamp)
    write_msg(writer, channels["joint"], kin_timestamp, joint_msg)

    # kinematics tf
    kintf_msg = create_kin_tf_msg(
        kin_obj["transformsSnapshot"]["childToParentEdgeMap"], kin_timestamp
    )
    write_msg(writer, channels["tf"], kin_timestamp, kintf_msg)

    # path of spot in vision
    vision_tf_body_msg = kin_obj["transformsSnapshot"]["childToParentEdgeMap"]["vision"]
    vision_body_pose = create_posestamped_msg(vision_tf_body_msg, kin_timestamp)
    spot_path_list.append(vision_body_pose)
    time_msg = Time(sec=kin_timestamp // 10**9, nanosec=kin_timestamp % 10**9)
    header = Header(stamp=time_msg, frame_id="vision")
    spot_path_msg = NavPath(header=header, poses=spot_path_list)
    write_msg(writer, channels["path"], kin_timestamp, spot_path_msg)

    # kinematics velocity
    kin_vel_odom_msg = create_twist_msg(
        kin_obj["velocityOfBodyInOdom"], kin_timestamp, "odom"
    )
    write_msg(writer, channels["vel_odom"], kin_timestamp, kin_vel_odom_msg)
    kin_vel_vision_msg = create_twist_msg(
        kin_obj["velocityOfBodyInVision"], kin_timestamp, "vision"
    )
    write_msg(writer, channels["vel_vision"], kin_timestamp, kin_vel_vision_msg)

    # battery state
    battery_dict = loc_resp["liveData"]["robotState"]["batteryStates"][0]
    # borrow kin time becasue there is no battery time
    battery_msg = create_battery_state_msg(battery_dict, kin_timestamp)
    write_msg(writer, channels["battery"], kin_timestamp, battery_msg)

    # gravity imu
    kin_dict = loc_resp["liveData"]["robotState"]["kinematicState"]
    grav_time = iso_to_nanoseconds(kin_dict["acquisitionTimestamp"])
    grav_msg = create_gravity_imu_msg(
        kin_dict["transformsSnapshot"]["childToParentEdgeMap"]["flat_body"], grav_time
    )
    write_msg(writer, channels["gravity"], grav_time, grav_msg)


def register_channels(writer):
    """Registers schemas and channels based on CHANNEL_CONFIG. Returns a dict of channel IDs."""

    channel_ids = {}
    schema_cache = {}

    for key, config in CHANNEL_CONFIG.items():
        topic = config[0]
        msg_type = config[1]

        # Register schema only once per type
        if msg_type not in schema_cache:
            schema_cache[msg_type] = writer.register_schema(
                name=msg_type,
                encoding="ros2msg",
                data=typestore.generate_msgdef(msg_type)[0].encode("utf-8"),
            )

        if topic == "/tf_static":
            metadata = {"offered_qos_profiles": json.dumps(STATICTF_LATCHING_QOS)}
        else:
            metadata = {}

        channel_ids[key] = writer.register_channel(
            schema_id=schema_cache[msg_type],
            topic=topic,
            message_encoding="cdr",
            metadata=metadata,
        )
    return channel_ids


def write_msg(writer, channel_id, timestamp, ros_msg):
    """adds an message to the mcap file"""
    writer.add_message(
        channel_id=channel_id,
        log_time=timestamp,
        data=typestore.serialize_cdr(ros_msg, ros_msg.__msgtype__),
        publish_time=timestamp,
    )


def create_map_tf_msg(loc_tf_dict, state_tf_dict, timestamp):
    """Creates the tf message for the seed (map) to body static tf"""

    time_obj = Time(sec=timestamp // 10**9, nanosec=timestamp % 10**9)

    seed_tf_body = tfdict_to_tfmatrix(loc_tf_dict)
    body_tf_vision = tfdict_to_tfmatrix(state_tf_dict["vision"])
    seed_tf_vision = seed_tf_body @ body_tf_vision
    seed_tf_vision_msg = format_transform_stamped(
        seed_tf_vision, time_obj, "map", "vision"
    )

    return TFMessage([seed_tf_vision_msg])


def create_battery_state_msg(battery_dict, timestamp):
    """Creates the bettery state message"""
    time_msg = create_time_msg(timestamp)

    battery_msg = BatteryState(
        header=Header(stamp=time_msg, frame_id="body"),
        voltage=battery_dict["voltage"],
        temperature=np.mean(battery_dict["temperatures"]),
        current=battery_dict["current"],
        charge=math.nan,
        capacity=math.nan,
        design_capacity=math.nan,
        percentage=battery_dict["chargePercentage"] / 100.0,
        power_supply_status=1,
        power_supply_health=0,
        power_supply_technology=2,
        present=True,
        cell_voltage=np.array([], dtype=np.float32),
        cell_temperature=np.array(battery_dict["temperatures"], dtype=np.float32),
        location="",
        serial_number=battery_dict["identifier"],
    )

    return battery_msg


def create_posestamped_msg(pose_dict, timestamp):
    """Creates a posestamped message for the path"""

    time_msg = create_time_msg(timestamp)
    header = Header(stamp=time_msg, frame_id="vision")

    body_tf_vision_matrix = tfdict_to_tfmatrix(pose_dict)
    vision_tf_body_matrix = np.linalg.inv(body_tf_vision_matrix)

    tx, ty, tz = vision_tf_body_matrix[0:3, 3]

    rotation_matrix = vision_tf_body_matrix[0:3, 0:3]
    qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()

    position = Point(
        x=tx,
        y=ty,
        z=tz,
    )
    orientation = Quaternion(
        x=qx,
        y=qy,
        z=qz,
        w=qw,
    )
    pose = Pose(position=position, orientation=orientation)
    pose_stamped = PoseStamped(header=header, pose=pose)
    return pose_stamped


def kitti_ground_truth(writer, channels, kitti_pose_path):
    """Takes the kitti ground truth poses if available and adds them as a pose"""
    seq_id = kitti_pose_path.stem
    times_txt_path = kitti_pose_path.parent.parent / "sequences" / seq_id / "times.txt"

    if not times_txt_path.exists():
        raise FileNotFoundError(f"Missing timestamps: {times_txt_path}")

    poses = np.loadtxt(kitti_pose_path)
    times = np.loadtxt(times_txt_path)

    calib_transform = np.array(
        [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float
    )
    calib_inv = np.linalg.inv(calib_transform)

    for pose_row, t_sec in zip(poses, times):
        kitti_matrix = np.eye(4)
        kitti_matrix[:3, :4] = pose_row.reshape(3, 4)

        global_matrix = calib_inv @ kitti_matrix @ calib_transform

        trans = global_matrix[:3, 3]
        rot = Rotation.from_matrix(global_matrix[:3, :3]).as_quat()

        # Time conversions
        timestamp_ns = int(t_sec * 1e9)
        sec = int(t_sec)
        nanosec = int((t_sec - sec) * 1e9)

        # Construct PoseStamped
        gt_pose_msg = PoseStamped(
            header=Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id="map_gt"),
            pose=Pose(
                position=Point(x=trans[0], y=trans[1], z=trans[2]),
                orientation=Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3]),
            ),
        )

        write_msg(writer, channels["gt"], timestamp_ns, gt_pose_msg)


def create_apriltag_marker_msg(obj_dict, timestamp):
    """create a marker for apriltags"""

    tag_dict = obj_dict["apriltagProperties"]
    time_msg = create_time_msg(timestamp)

    life_time = Duration(sec=2, nanosec=0)

    tag_id = tag_dict["tagId"]

    bbox = Vector3(x=tag_dict["dimensions"]["x"], y=tag_dict["dimensions"]["y"], z=0.05)

    vision_tf_tag = tfdict_to_tfmatrix(
        obj_dict["transformsSnapshot"]["childToParentEdgeMap"][
            "filtered_fiducial_" + str(tag_id)
        ]
    )

    tx, ty, tz = vision_tf_tag[0:3, 3]
    rotation_matrix = vision_tf_tag[0:3, 0:3]
    qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()

    pose = Pose(
        position=Point(x=tx, y=ty, z=tz),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )

    mesh_file = MeshFile(filename="", data=np.array([], dtype=np.uint8))
    comp_img = CompressedImage(
        header=Header(stamp=time_msg, frame_id="vision"),
        format="",
        data=np.array([], dtype=np.uint8),
    )

    box_marker = Marker(
        header=Header(stamp=time_msg, frame_id="vision"),
        ns="apriltag",
        id=tag_id,
        type=1,
        action=0,
        pose=pose,
        scale=bbox,
        color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5),
        lifetime=life_time,
        frame_locked=False,
        points=[],
        colors=[],
        text="",
        mesh_resource="",
        mesh_use_embedded_materials=False,
        texture_resource="",
        texture=comp_img,
        uv_coordinates=[UVCoordinate(u=0.0, v=0.0)],
        mesh_file=mesh_file,
    )

    return MarkerArray(markers=[box_marker])


def create_staircase_msg(staircase_dict, timestamp):
    """Encodes the staircase dict as a set of boxes the size of the stairs
    The staircase dict gives the bottom center of the first step as the start
    and then the top center position of the step relative to the start"""

    life_time = Duration(sec=2, nanosec=0)
    time_msg = create_time_msg(timestamp)

    tag_id = 999
    stairs = staircase_dict["staircaseProperties"]["staircase"]["steps"]

    # get start position array
    vision_tf_start = tfdict_to_tfmatrix(
        staircase_dict["staircaseProperties"]["staircase"]["stairTform"][
            "frameTformStairs"
        ]
    )

    marker_array = []

    previous_z = 0.0

    # loop through stairs
    for i, current_stair in enumerate(stairs):
        current_x = current_stair["point"].get("x", 0.0)
        current_y = current_stair["point"].get("y", 0.0)
        current_z = current_stair["point"].get("z", 0.0)
        width = current_stair["width"].get("width", 1.0)

        # height of step
        scale_z = current_z - previous_z
        previous_z = current_z

        average_run = staircase_dict["staircaseProperties"]["staircase"]["averageRun"]

        # depth of the step
        if i + 1 < len(stairs):
            next_x = stairs[i + 1]["point"].get("x", 0.0)
            scale_x = next_x - current_x
        else:
            # last step can't get value so set to average
            scale_x = average_run

        # check becasue sometimes it reports a step double the size
        if scale_x > average_run * 1.5:
            scale_x = 0.5 * scale_x

        # find the center of the box marker
        center_x = current_x + (scale_x / 2.0)
        center_y = current_y
        center_z = current_z - (scale_z / 2.0)

        # create a tf matrix
        start_tf_stair = np.eye(4)
        start_tf_stair[0:3, 3] = [center_x, center_y, center_z]
        vision_tf_stair = vision_tf_start @ start_tf_stair
        tx, ty, tz = vision_tf_stair[0:3, 3]
        rotation_matrix = vision_tf_stair[0:3, 0:3]
        qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()

        pose = Pose(
            position=Point(x=tx, y=ty, z=tz),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )
        bbox = Vector3(x=scale_x, y=width, z=scale_z)

        mesh_file = MeshFile(filename="", data=np.array([], dtype=np.uint8))
        comp_img = CompressedImage(
            header=Header(stamp=time_msg, frame_id="vision"),
            format="",
            data=np.array([], dtype=np.uint8),
        )

        box_marker = Marker(
            header=Header(stamp=time_msg, frame_id="vision"),
            ns="stairs",
            id=tag_id,
            type=1,
            action=0,
            pose=pose,
            scale=bbox,
            color=ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.5),
            lifetime=life_time,
            frame_locked=False,
            points=[],
            colors=[],
            text="",
            mesh_resource="",
            mesh_use_embedded_materials=False,
            texture_resource="",
            texture=comp_img,
            uv_coordinates=[UVCoordinate(u=0.0, v=0.0)],
            mesh_file=mesh_file,
        )

        marker_array.append(box_marker)
        tag_id += 1

    marker_array_msg = MarkerArray(markers=marker_array)
    return marker_array_msg


def create_marker_objects_msg(object_dict, timestamp):
    """creates markers for blobs including a bounding box and arrow for velocity"""

    life_time = Duration(sec=2, nanosec=0)
    time_msg = create_time_msg(timestamp)

    marker_array = []

    obj_timestamp = iso_to_nanoseconds(object_dict["acquisitionTime"])
    time_msg = Time(sec=obj_timestamp // 10**9, nanosec=obj_timestamp % 10**9)

    tracked_props = object_dict["trackedEntityProperties"]

    marker_id = tracked_props["entityId"]

    tf_snap = object_dict["transformsSnapshot"]["childToParentEdgeMap"][
        "blob" + str(marker_id)
    ]["parentTformChild"]

    position = Point(
        x=tf_snap["position"]["x"],
        y=tf_snap["position"]["y"],
        z=tf_snap["position"]["z"],
    )

    rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)

    pose = Pose(position=position, orientation=rotation)

    bbox = Vector3(
        x=tracked_props["sizeInFrame"]["x"],
        y=tracked_props["sizeInFrame"]["y"],
        z=tracked_props["sizeInFrame"]["z"],
    )

    arrow_point = Point(
        x=tracked_props["velocity"].get("x", 0.0),
        y=tracked_props["velocity"].get("y", 0.0),
        z=tracked_props["velocity"].get("z", 0.0),
    )

    arrow_base = Point(
        x=(0.0),
        y=(0.0),
        z=(0.0),
    )

    mesh_file = MeshFile(filename="", data=np.array([], dtype=np.uint8))
    comp_img = CompressedImage(
        header=Header(stamp=time_msg, frame_id="vision"),
        format="",
        data=np.array([], dtype=np.uint8),
    )

    box_marker = Marker(
        header=Header(stamp=time_msg, frame_id="vision"),
        ns="boxes",
        id=marker_id,
        type=1,
        action=0,
        pose=pose,
        scale=bbox,
        color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5),
        lifetime=life_time,
        frame_locked=False,
        points=[],
        colors=[],
        text="",
        mesh_resource="",
        mesh_use_embedded_materials=False,
        texture_resource="",
        texture=comp_img,
        uv_coordinates=[UVCoordinate(u=0.0, v=0.0)],
        mesh_file=mesh_file,
    )

    arrow_marker = Marker(
        header=Header(stamp=time_msg, frame_id="vision"),
        ns="velocities",
        id=marker_id,
        type=0,
        action=0,
        pose=pose,
        scale=Vector3(x=0.1, y=0.2, z=0.0),
        color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # red
        lifetime=life_time,
        frame_locked=False,
        points=[arrow_base, arrow_point],
        colors=[],
        text="",
        mesh_resource="",
        mesh_use_embedded_materials=False,
        texture_resource="",
        texture=comp_img,
        uv_coordinates=[UVCoordinate(u=0.0, v=0.0)],
        mesh_file=mesh_file,
    )

    marker_array.append(box_marker)
    marker_array.append(arrow_marker)

    marker_array_msg = MarkerArray(markers=marker_array)
    return marker_array_msg


def create_joint_state_msg(joint_dict, timestamp):
    """creates joint messages for all of Spot's joints"""

    time_msg = create_time_msg(timestamp)
    names = []
    positions = []
    velocities = []
    loads = []

    for joint in joint_dict:
        input_name = joint["name"]
        output_name = ""
        if input_name[0] == "a":
            output_name = output_name + input_name.replace("0.", "_")

        else:
            if input_name[0] == "f":
                output_name = output_name + "front"
            elif input_name[0] == "h":
                output_name = output_name + "rear"

            if input_name[1] == "l":
                output_name = output_name + "_left"
            elif input_name[1] == "r":
                output_name = output_name + "_right"

            if input_name[4] == "x":
                output_name = output_name + "_hip_x"
            elif input_name[4] == "y":
                output_name = output_name + "_hip_y"
            elif input_name[4] == "n":
                output_name = output_name + "_knee"

        names.append(output_name)
        positions.append(joint["position"])  # - (np.pi / 2.0))
        velocities.append(joint["velocity"])
        loads.append(joint["load"])

    joint_msg = JointState(
        header=Header(stamp=time_msg, frame_id="base_link"),
        name=names,
        position=np.array(positions, dtype=np.float64),
        velocity=np.array(velocities, dtype=np.float64),
        effort=np.array(loads, dtype=np.float64),
    )
    return joint_msg


def create_pointcloud2_msg(pc_dict, timestamp):
    """Creates a pointcloud2 msg"""

    header = Header(
        stamp=Time(sec=timestamp // 10**9, nanosec=timestamp % 10**9),
        frame_id=pc_dict["source"]["frameNameSensor"],
    )

    fields = [
        PointField(name="x", offset=int(0), datatype=int(7), count=int(1)),
        PointField(name="y", offset=int(4), datatype=int(7), count=int(1)),
        PointField(name="z", offset=int(8), datatype=int(7), count=int(1)),
    ]

    points = unpack_point_cloud(pc_dict)

    point_step = 12
    row_step = point_step * len(points)

    data = points.view(np.uint8).flatten()

    msg = PointCloud2(
        header=header,
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=row_step,
        data=data,
        is_dense=True,
    )

    return msg


def create_twist_msg(vel_dict, timestamp, frame_id):
    """Creates a twist stamped msg"""

    time_msg = create_time_msg(timestamp)
    header = Header(stamp=time_msg, frame_id=frame_id)

    linear = Vector3(
        x=vel_dict["linear"]["x"], y=vel_dict["linear"]["y"], z=vel_dict["linear"]["z"]
    )
    angular = Vector3(
        x=vel_dict["angular"]["x"],
        y=vel_dict["angular"]["y"],
        z=vel_dict["angular"]["z"],
    )

    msg = TwistStamped(header=header, twist=Twist(linear=linear, angular=angular))
    return msg


def create_pc_tf_msg(tf_dict, timestamp):
    """Create the tf msg for the pointcloud"""
    transforms = create_default_tf_msg(tf_dict, timestamp)
    time_msg = create_time_msg(timestamp)

    odom_tf_sensor = tfdict_to_tfmatrix(tf_dict["sensor_origin_generated"])

    body_tf_odom = tfdict_to_tfmatrix(tf_dict["odom"])

    body_tf_sensor = body_tf_odom @ odom_tf_sensor
    body_tf_sensor_msg = format_transform_stamped(
        body_tf_sensor, time_msg, "base_link", "sensor_origin_generated"
    )

    transforms.append(body_tf_sensor_msg)

    return TFMessage(transforms=transforms)


def create_kin_tf_msg(tf_dict, timestamp):
    """create the tf message for the kinematic dict"""
    transforms = create_default_tf_msg(tf_dict, timestamp)
    time_msg = create_time_msg(timestamp)

    odom_tf_gpe = tfdict_to_tfmatrix(tf_dict["gpe"])
    body_tf_odom = tfdict_to_tfmatrix(tf_dict["odom"])

    body_tf_gpe = body_tf_odom @ odom_tf_gpe

    body_tf_gpe_msg = format_transform_stamped(
        body_tf_gpe, time_msg, "base_link", "gpe"
    )

    transforms.append(body_tf_gpe_msg)

    return TFMessage(transforms=transforms)


def create_grid_tf_msg(tf_dict, timestamp):
    """Create the tf msg for the grids"""

    transforms = create_default_tf_msg(tf_dict, timestamp)
    time_msg = create_time_msg(timestamp)

    vision_tf_grid = tfdict_to_tfmatrix(
        next((v for k, v in tf_dict.items() if "local_grid" in k), None)
    )

    body_tf_vision = tfdict_to_tfmatrix(tf_dict["vision"])

    body_tf_grid = body_tf_vision @ vision_tf_grid

    body_tf_grid_msg = format_transform_stamped(
        body_tf_grid, time_msg, "base_link", "local_grids_corner"
    )

    transforms.append(body_tf_grid_msg)

    return TFMessage(transforms=transforms)


def create_default_tf_msg(tf_dict, timestamp):
    """Creates the tf msgs for the common frames"""

    transforms = []
    time_msg = create_time_msg(timestamp)

    # vision_tf_odom
    body_tf_vision = tfdict_to_tfmatrix(tf_dict["vision"])
    body_tf_odom = tfdict_to_tfmatrix(tf_dict["odom"])
    vision_tf_odom = np.linalg.inv(body_tf_vision) @ body_tf_odom

    vision_tf_odom_msg = format_transform_stamped(
        vision_tf_odom, time_msg, "vision", "odom"
    )

    transforms.append(vision_tf_odom_msg)

    # odom_tf_body
    odom_tf_body = np.linalg.inv(body_tf_odom)
    odom_tf_body_msg = format_transform_stamped(odom_tf_body, time_msg, "odom", "body")

    transforms.append(odom_tf_body_msg)

    return transforms


def format_transform_stamped(tf, time_msg, parent_name, child_name):
    """Takes a tf matrix and formats to a transform stamped msg."""

    tx, ty, tz = tf[0:3, 3]

    rotation_matrix = tf[0:3, 0:3]
    qx, qy, qz, qw = Rotation.from_matrix(rotation_matrix).as_quat()

    tf_wp = TransformStamped(
        header=Header(stamp=time_msg, frame_id=parent_name),
        child_frame_id=child_name,
        transform=Transform(
            translation=Vector3(x=tx, y=ty, z=tz),
            rotation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        ),
    )
    return tf_wp


def create_gridmap_msg(grids_dict, timestamp):
    """Creates a gridmap msg."""

    time_msg = create_time_msg(timestamp)
    header = Header(stamp=time_msg, frame_id="local_grids_corner")

    layers = []
    data_arrays = []

    resolution = grids_dict[0]["extent"]["cellSize"]
    width = grids_dict[0]["extent"]["numCellsX"]
    height = grids_dict[0]["extent"]["numCellsY"]

    # physical grid size
    length_x = width * resolution
    length_y = height * resolution

    # loop through grids
    for grid_dict in grids_dict:

        # terrain_valid is just a mask for terrain
        if not grid_dict["localGridTypeName"] == "terrain_valid":

            # if the grid is terrain unpack it and apply the mask
            if grid_dict["localGridTypeName"] == "terrain":
                terrain_data = unpack_grid(grid_dict)
                terrain_valid = unpack_grid(
                    next(
                        grid
                        for grid in grids_dict
                        if grid["localGridTypeName"] == "terrain_valid"
                    )
                )

                terrain_data[terrain_valid == 0] = float("NaN")
                grid_array = terrain_data
            else:
                grid_array = unpack_grid(grid_dict)

            layers.append(grid_dict["localGridTypeName"])

            # Flatten array and create MultiArrayLayout
            flat_data = grid_array.astype(np.float32).ravel()

            dim_height = MultiArrayDimension(
                label="column_index", size=height, stride=height * width
            )
            dim_width = MultiArrayDimension(label="row_index", size=width, stride=width)

            layout = MultiArrayLayout(dim=[dim_height, dim_width], data_offset=0)
            data_arrays.append(Float32MultiArray(layout=layout, data=flat_data))

    # setup grid info
    info = GridMapInfo(
        header=header,
        resolution=float(resolution),
        length_x=float(length_x),
        length_y=float(length_y),
        pose=Pose(
            position=Point(x=length_x / 2, y=length_y / 2, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=1.0, w=0.0),
        ),
    )

    # 3. Build Final Message
    return GridMap(
        info=info,
        layers=layers,
        basic_layers=layers,
        data=data_arrays,
        outer_start_index=0,
        inner_start_index=0,
    )


def create_gravity_imu_msg(tf_dict, timestamp):
    """Create a fake imu message with gravity based on the spot flat body frame"""

    time_msg = create_time_msg(timestamp)
    header = Header(stamp=time_msg, frame_id="body")

    orientation_covariance = np.array(
        [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, -1.0], dtype=np.float64
    )

    angular_velocity_covariance = np.array(
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )

    linear_acceleration_covariance = np.array(
        [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4], dtype=np.float64
    )

    rot_dict = tf_dict["parentTformChild"]["rotation"]

    quat = [
        rot_dict.get("x", 0.0),
        rot_dict.get("y", 0.0),
        rot_dict.get("z", 0.0),
        rot_dict.get("w", 0.0),
    ]

    rot = Rotation.from_quat(quat)
    grav_flat = np.array([0.0, 0.0, 9.81])
    grav_body = rot.apply(grav_flat)

    orientation = Quaternion(
        x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
    )

    imu_msg = Imu(
        header=header,
        orientation=orientation,
        orientation_covariance=orientation_covariance,
        angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
        angular_velocity_covariance=angular_velocity_covariance,
        linear_acceleration=Vector3(
            x=float(grav_body[0]), y=float(grav_body[1]), z=float(grav_body[2])
        ),
        linear_acceleration_covariance=linear_acceleration_covariance,
    )
    return imu_msg


def create_time_msg(timestamp):
    return Time(sec=timestamp // 10**9, nanosec=timestamp % 10**9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a run folder to MCAP format."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the localization response run folder",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Path to the output folder for mcap files",
    )
    parser.add_argument(
        "--kitti_pose",
        type=Path,
        required=False,
        help="The route to the kitti pose.txt file for ground truth poses",
    )

    args = parser.parse_args()

    run_to_mcap(args.run_dir, args.out_dir, args.kitti_pose)
