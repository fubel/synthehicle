import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import json
import os
import cv2
import carla
import bbox_new_process
import math


######################################################################
# Use these functions to calculate bounding boxes and apply the filter
######################################################################

# Use this function to get 2D bounding boxes of visible vehicles to camera using semantic LIDAR
def get_all_data(
    vehicles,
    camera,
    lidar_data,
    max_dist=100,
    min_detect=1,
    show_img=None,
    json_path=None,
    use_depth=False,
    path=None,
    depth_data=None,
    framenumber=None,
    save_lidar=False,
):
    filtered_data = filter_lidar(lidar_data, camera, max_dist)
    if show_img != None and save_lidar:
        show_lidar(filtered_data, camera, show_img, path, framenumber)

    filtered_data = np.array([p for p in filtered_data if p.object_idx != 0])
    filtered_data, filtered_v_s_dist2 = get_points_id(
        filtered_data, vehicles, camera, max_dist
    )

    visible_id, idx_counts = np.unique(
        [p.object_idx for p in filtered_data], return_counts=True
    )
    visible_vehicles = [v for v in vehicles if v.id in visible_id]
    # visible_vehicles_sensor_dist2 = [dist for (id, dist) in filtered_v_s_dist2 if id in visible_id]
    #
    # visible_vehicles_sensor_dist2 = []
    # for v in visible_vehicles:
    #     id = v.id
    #     visible_vehicles_sensor_dist2.append([dist for (ID, dist) in filtered_v_s_dist2 if ID == id])

    if not use_depth:
        visible_vehicles = [
            v
            for v in vehicles
            if idx_counts[(visible_id == v.id).nonzero()[0]] >= min_detect
        ]

    # Combine info of depth camera
    if use_depth:
        visible_vehicles = [
            v for v in vehicles if idx_counts[(visible_id == v.id).nonzero()[0]] >= 12
        ]
        bounding_boxes = bbox_new_process.get_bb_data(visible_vehicles, camera)
        depth_array = process_depth_data(depth_data)
        depth_based_bb, depth_based_vehicle = bbox_new_process.apply_filters_to_3d_bb(
            bounding_boxes,
            depth_array,
            show_img.height,
            show_img.width,
            vehicle=visible_vehicles,
        )
        depth_based_vehicles_id = {
            v.id: v.type_id for v in depth_based_vehicle}
    else:
        # Based only on Lidar
        bounding_boxes_3d = [
            get_bounding_box(vehicle, camera) for vehicle in visible_vehicles
        ]
        bounding_boxes_2d = [get_2d_bb(vehicle, camera)
                             for vehicle in visible_vehicles]
        world_coords = [
            vehicle_to_world(create_bb_points(vehicle), vehicle)
            for vehicle in visible_vehicles
        ]
        id_bounding_boxes_2d = {}
        for i, v in enumerate(visible_vehicles):
            id = v.id
            id_bounding_boxes_2d[id] = bounding_boxes_2d[i]

        filtered_bb_2d_lidar = []
        filtered_bb_3d_lidar = []
        filtered_world_coords = []
        filtered_vehicle_lidar = []
        locations = []
        rotations = []
        for w_coord, db3, bb, vehicle in zip(
            world_coords, bounding_boxes_3d, bounding_boxes_2d, visible_vehicles
        ):
            bb_xmin = bb[0][0]
            bb_ymin = bb[0][1]
            bb_xmax = bb[1][0]
            bb_ymax = bb[1][1]
            if math.ceil(bb_xmax - bb_xmin) <= 32 or math.ceil(bb_ymax - bb_ymin) <= 32:
                continue
            filtered_bb_3d_lidar.append(db3)
            filtered_bb_2d_lidar.append(bb)
            filtered_world_coords.append(w_coord)
            filtered_vehicle_lidar.append(vehicle)
            t = vehicle.get_transform()
            locations.append([t.location.x, t.location.y, t.location.z])
            rotations.append(
                [t.rotation.pitch, t.rotation.yaw, t.rotation.roll])

        filtered_vehicles_id_lidar = {
            v.id: v.type_id for v in filtered_vehicle_lidar}
        bounding_boxes_2d_final = filtered_bb_2d_lidar
        bounding_boxes_3d_final = filtered_bb_3d_lidar

    filtered_out = {}
    if use_depth:
        filtered_out["vehicles_id"] = depth_based_vehicles_id
        filtered_out["vehicles"] = depth_based_vehicle
        filtered_out["bbox"] = depth_based_bb
    else:
        filtered_out["vehicles_id"] = filtered_vehicles_id_lidar
        filtered_out["vehicles"] = filtered_vehicle_lidar
        filtered_out["bbox"] = bounding_boxes_2d_final
        filtered_out["bbox_3d"] = bounding_boxes_3d_final
        filtered_out["world_coords"] = filtered_world_coords
        filtered_out["locations"] = locations
        filtered_out["rotations"] = rotations
        # filtered_out['dist2'] = filtered_vehicle_sensor_dist2

    if json_path is not None:
        filtered_out["class"] = get_vehicle_class(
            filtered_out["vehicles"], json_path)
    return filtered_out, filtered_data


##########################################
# Use this function to get camera k matrix
##########################################

# Get camera matrix
def get_camera_intrinsic(sensor):
    VIEW_WIDTH = int(sensor.attributes["image_size_x"])
    VIEW_HEIGHT = int(sensor.attributes["image_size_y"])
    VIEW_FOV = int(float(sensor.attributes["fov"]))
    intrinsic = np.identity(3)
    intrinsic[0, 2] = VIEW_WIDTH / 2.0
    intrinsic[1, 2] = VIEW_HEIGHT / 2.0
    intrinsic[0, 0] = intrinsic[1, 1] = VIEW_WIDTH / (
        2.0 * np.tan(VIEW_FOV * np.pi / 360.0)
    )
    return intrinsic


def get_camera_homography(sensor):
    change = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]])
    intrinsic_matrix = get_camera_intrinsic(sensor)
    extrinsic_matrix = np.array(sensor.get_transform().get_inverse_matrix())
    standard_change = np.dot(change, np.delete(extrinsic_matrix, 2, 1)[:-1, :])
    homography_matrix = np.dot(intrinsic_matrix, standard_change)
    result = homography_matrix / homography_matrix[-1, -1].reshape((-1, 1))
    return result


################################################
# Use these functions to find 2D BB in the image
################################################

# Extract bounding box vertices of vehicle
def create_bb_points(vehicle):
    cords = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return cords


# Get transformation matrix from carla.Transform object
def get_matrix(transform):
    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def mat2transform(matrix):
    s_p = matrix[2, 0]
    pitch = np.rad2deg(np.arcsin(s_p))
    c_p = np.cos(np.radians(pitch))
    c_r = matrix[2, 2] / c_p
    roll = np.rad2deg(np.arccos(c_r))
    s_y = matrix[1, 0] / c_p
    yaw = np.rad2deg(np.arcsin(s_y))
    T = carla.Transform(
        carla.Location(x=matrix[0, 3], y=matrix[1, 3], z=matrix[2, 3]),
        carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
    )
    return T


def get_camera_position(json_path=None):
    f = open(json_path)
    camera_data = json.load(f)
    camera_info = camera_data
    x = camera_info["x"]
    y = camera_info["y"]
    z = camera_info["z"]
    pitch = math.degrees(camera_info["pitch"])
    yaw = math.degrees(camera_info["yaw"])
    roll = math.degrees(camera_info["roll"])
    T = carla.Transform(
        carla.Location(x=x, y=y, z=z), carla.Rotation(
            pitch=pitch, yaw=yaw, roll=roll)
    )
    return T


def get_camera_position_matrix(json_path=None):
    f = open(json_path)
    camera_data = json.load(f)
    camera_position_matrix = camera_data["spectator_matrix"]

    return np.mat(camera_position_matrix)


# Transform coordinate from vehicle reference to world reference
def vehicle_to_world(cords, vehicle):
    bb_transform = carla.Transform(vehicle.bounding_box.location)
    bb_vehicle_matrix = get_matrix(bb_transform)
    vehicle_world_matrix = get_matrix(vehicle.get_transform())
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords


# Transform coordinate from world reference to sensor reference
def world_to_sensor(cords, sensor):
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)
    return sensor_cords


# Transform coordinate from vehicle reference to sensor reference
def vehicle_to_sensor(cords, vehicle, sensor):
    world_cord = vehicle_to_world(cords, vehicle)
    sensor_cord = world_to_sensor(world_cord, sensor)
    return sensor_cord


# Summarize bounding box creation and project the poins in sensor image
def get_bounding_box(vehicle, sensor):
    camera_k_matrix = get_camera_intrinsic(sensor)
    bb_cords = create_bb_points(vehicle)
    cords_x_y_z = vehicle_to_sensor(bb_cords, vehicle, sensor)[:3, :]
    cords_y_minus_z_x = np.concatenate(
        [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]]
    )
    bbox = np.transpose(np.dot(camera_k_matrix, cords_y_minus_z_x))
    camera_bbox = np.concatenate(
        [bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1
    )
    return camera_bbox


##############################################################################
# Draw 2D bounding box (4 vertices) from 3D bounding box (8 vertices) in image
##############################################################################

# 2D bounding box is represented by two corner points
def p3d_to_p2d_bb(p3d_bb, sensor):
    width = int(sensor.attributes["image_size_x"])
    height = int(sensor.attributes["image_size_y"])

    min_x = np.amin(p3d_bb[:, 0])
    min_y = np.amin(p3d_bb[:, 1])
    max_x = np.amax(p3d_bb[:, 0])
    max_y = np.amax(p3d_bb[:, 1])
    if min_x < 0:
        min_x = 0
    if min_y < 0:
        min_y = 0
    if max_x > width:
        max_x = width - 1
    if max_y > height:
        max_y = height - 1
    p2d_bb = np.array([[min_x, min_y], [max_x, max_y]])
    return p2d_bb


# Summarize 2D bounding box creation
def get_2d_bb(vehicle, sensor):
    p3d_bb = get_bounding_box(vehicle, sensor)
    p2d_bb = p3d_to_p2d_bb(p3d_bb, sensor)
    return p2d_bb


# Calculating the IoU od two bounding box
def iou(box1, box2):
    # determine the coordinates of the intersection rectangle
    x_left = max(box1[0][0], box2[0][0])
    y_top = max(box1[0][1], box2[0][1])
    x_right = min(box1[1][0], box2[1][0])
    y_bottom = min(box1[1][1], box2[1][1])
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # compute the area of both AABBs
    bb1_area = (box1[1][0] - box1[0][0]) * (box1[1][1] - box1[0][1])
    bb2_area = (box2[1][0] - box2[0][0]) * (box2[1][1] - box2[0][1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


##################################################
# Use these functions to remove invisible vehicles
##################################################

# Get numpy 2D array of vehicles' location and rotation from world reference, also locations from sensor reference
def get_list_transform(vehicles_list, sensor):
    t_list = []
    for vehicle in vehicles_list:
        v = vehicle.get_transform()
        transform = [
            v.location.x,
            v.location.y,
            v.location.z,
            v.rotation.roll,
            v.rotation.pitch,
            v.rotation.yaw,
        ]
        t_list.append(transform)
    t_list = np.array(t_list).reshape((len(t_list), 6))

    transform_h = np.concatenate(
        (t_list[:, :3], np.ones((len(t_list), 1))), axis=1)
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    transform_s = np.dot(world_sensor_matrix, transform_h.T).T
    return t_list, transform_s


# Remove vehicles that are not in the FOV of the sensor
def filter_angle(vehicles_list, v_transform, v_transform_s, sensor):
    attr_dict = sensor.attributes
    VIEW_FOV = float(attr_dict["fov"])
    v_angle = np.arctan2(
        v_transform_s[:, 1], v_transform_s[:, 0]) * 180 / np.pi

    selector = np.array(np.absolute(v_angle) < (int(VIEW_FOV) / 2))
    vehicles_list_f = [v for v, s in zip(vehicles_list, selector) if s]
    v_transform_f = v_transform[selector[:, 0], :]
    v_transform_s_f = v_transform_s[selector[:, 0], :]
    return vehicles_list_f, v_transform_f, v_transform_s_f


# Remove vehicles that have distance > max_dist from the sensor
def filter_distance(vehicles_list, v_transform, v_transform_s, sensor, max_dist=100):
    s = sensor.get_transform()
    s_transform = np.array([s.location.x, s.location.y, s.location.z])
    dist2 = np.sum(np.square(v_transform[:, :3] - s_transform), axis=1)

    selector = dist2 < (max_dist**2)
    vehicles_list_f = [v for v, s in zip(vehicles_list, selector) if s]
    v_transform_f = v_transform[selector, :]
    v_transform_s_f = v_transform_s[selector, :]
    return vehicles_list_f, v_transform_f, v_transform_s_f


# Apply angle and distance filters in one function
def filter_angle_distance(vehicles_list, sensor, max_dist=100):
    vehicles_transform, vehicles_transform_s = get_list_transform(
        vehicles_list, sensor)
    vehicles_list, vehicles_transform, vehicles_transform_s = filter_distance(
        vehicles_list, vehicles_transform, vehicles_transform_s, sensor, max_dist
    )
    vehicles_list, vehicles_transform, vehicles_transform_s = filter_angle(
        vehicles_list, vehicles_transform, vehicles_transform_s, sensor
    )
    s = sensor.get_transform()
    s_transform = np.array([s.location.x, s.location.y, s.location.z])
    vehicles_list_f_dist2 = np.sum(
        np.square(vehicles_transform[:, :3] - s_transform), axis=1
    )
    return vehicles_list, vehicles_list_f_dist2


# Filter out lidar points that are outside camera FOV
def filter_lidar(lidar_data, camera, max_dist):
    CAM_W = int(camera.attributes["image_size_x"])
    CAM_H = int(camera.attributes["image_size_y"])
    CAM_HFOV = float(camera.attributes["fov"])
    CAM_VFOV = np.rad2deg(
        2 * np.arctan(np.tan(np.deg2rad(CAM_HFOV / 2)) * CAM_H / CAM_W)
    )
    lidar_points = np.array(
        [[p.point.y, -p.point.z, p.point.x] for p in lidar_data])

    dist2 = np.sum(np.square(lidar_points), axis=1).reshape((-1))
    p_angle_h = np.absolute(
        np.arctan2(lidar_points[:, 0], lidar_points[:, 2]) * 180 / np.pi
    ).reshape((-1))
    p_angle_v = np.absolute(
        np.arctan2(lidar_points[:, 1], lidar_points[:, 2]) * 180 / np.pi
    ).reshape((-1))

    selector = np.array(
        np.logical_and(
            np.logical_and(dist2 > 0, dist2 < (max_dist**2)),
            np.logical_and(p_angle_h < (CAM_HFOV / 2),
                           p_angle_v < (CAM_VFOV / 2)),
        )
    )
    filtered_lidar = [pt for pt, s in zip(lidar_data, selector) if s]
    return filtered_lidar


# Save camera image with projected lidar points for debugging purpose
def show_lidar(lidar_data, camera, carla_img, path, framenumber):
    color_map = {
        "0": (128, 128, 128),
        "1": (153, 153, 0),
        "2": (153, 153, 0),
        "3": (255, 255, 255),
        "4": (0, 0, 255),
        "5": (153, 153, 0),
        "6": (255, 255, 255),
        "7": (255, 255, 255),
        "8": (255, 255, 255),
        "9": (0, 204, 0),
        "10": (255, 128, 0),
        "11": (153, 153, 0),
        "12": (255, 51, 51),
        "13": (255, 255, 255),
        "14": (255, 255, 255),
        "15": (96, 96, 96),
        "16": (64, 64, 64),
        "17": (64, 64, 64),
        "18": (255, 51, 51),
        "19": (255, 255, 255),
        "20": (255, 255, 255),
        "21": (0, 128, 255),
        "22": (255, 255, 255),
    }
    lidar_np = np.array([[p.point.y, -p.point.z, p.point.x]
                        for p in lidar_data])
    cam_k = get_camera_intrinsic(camera)

    # Project LIDAR 3D to Camera 2D
    lidar_2d = np.transpose(np.dot(cam_k, np.transpose(lidar_np)))
    lidar_2d = (lidar_2d / lidar_2d[:, 2].reshape((-1, 1))).astype(int)

    # Visualize the result
    c_scale = []
    for pts in lidar_data:
        # if pts.object_idx == 0:
        #     c_scale.append(255)
        # else:
        #     c_scale.append(0)
        semantic_tag = str(pts.object_tag)
        color = color_map[semantic_tag]
        c_scale.append(color)

    carla_img.convert(carla.ColorConverter.Raw)
    img_bgra = np.array(carla_img.raw_data).reshape(
        (carla_img.height, carla_img.width, 4)
    )
    img_rgb = np.zeros((carla_img.height, carla_img.width, 3))
    img_rgb[:, :, 0] = img_bgra[:, :, 2]
    img_rgb[:, :, 1] = img_bgra[:, :, 1]
    img_rgb[:, :, 2] = img_bgra[:, :, 0]
    img_rgb = np.uint8(img_rgb)

    for p, c in zip(lidar_2d, c_scale):
        # c = int(c)
        cv2.circle(img_rgb, tuple(p[:2]), 1, c, -1)
    filename = f"{path}/out_lidar_img/%06d.jpg" % framenumber
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, img_rgb)


# Add actor ID of the vehcile hit by the lidar points
def get_points_id(lidar_points, vehicles, camera, max_dist):
    vehicles_f, vehicles_f_dist2 = filter_angle_distance(
        vehicles, camera, max_dist)

    vehicles_f_dist2_new = []
    for i, v in enumerate(vehicles_f):
        id = v.id
        vehicles_f_dist2_new.append((id, vehicles_f_dist2[i]))

    fixed_lidar_points = []
    for p in lidar_points:
        sensor_world_matrix = get_matrix(camera.get_transform())

        pw = np.dot(sensor_world_matrix, [
                    [p.point.x], [p.point.y], [p.point.z], [1]])
        pw = carla.Location(pw[0, 0], pw[1, 0], pw[2, 0])
        for v in vehicles_f:
            if v.bounding_box.contains(pw, v.get_transform()):
                p.object_idx = v.id
                break
        fixed_lidar_points.append(p)
    return fixed_lidar_points, vehicles_f_dist2_new


####################################
# Function to return vehicle's class
####################################


def get_vehicle_class(vehicles, json_path=None):
    f = open(json_path)
    json_data = json.load(f)
    vehicles_data = json_data["classification"]
    other_class = json_data["reference"].get("others")
    class_list = []
    for v in vehicles:
        v_class = int(vehicles_data.get(v.type_id, other_class))
        class_list.append(v_class)
    return class_list


###########################
# Function to save output
###########################

# Use this function to save the rgb image (with and without bounding box) and bounding boxes data
def save_output(
    frame_number,
    carla_img,
    id,
    bboxes,
    bboxes_3d,
    world_coords,
    vehicle_class=None,
    cc_rgb=carla.ColorConverter.Raw,
    path=None,
    save_patched=False,
    add_data=None,
    out_format="pickle",
):
    file_name = path + "/out_bbox"
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    carla_img.save_to_disk(path + "/out_rgb/%06d.jpg" % frame_number, cc_rgb)

    out_dict = {}
    bboxes_list = [bbox.tolist() for bbox in bboxes]
    bboxes_3d_list = [bbox.tolist() for bbox in bboxes_3d]
    world_coords_list = [wc.tolist() for wc in world_coords]
    out_dict["bboxes"] = bboxes_list
    out_dict["bboxes_3d"] = bboxes_3d_list
    out_dict["world_coords"] = world_coords_list

    vehicle_id_list = [v_id for v_id in id]
    out_dict["vehicle_id"] = vehicle_id_list

    if vehicle_class is not None:
        out_dict["vehicle_class"] = vehicle_class
    if add_data is not None:
        out_dict["others"] = add_data
    if out_format == "json":
        filename = path + "/out_bbox/%06d.txt" % frame_number
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)
    else:
        filename = path + "/out_bbox/%06d.pkl" % frame_number
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, "w") as outfile:
            json.dump(out_dict, outfile, indent=4)

    if save_patched:
        carla_img.convert(cc_rgb)
        img_bgra = np.array(carla_img.raw_data).reshape(
            (carla_img.height, carla_img.width, 4)
        )
        img_rgb = np.zeros((carla_img.height, carla_img.width, 3))

        img_rgb[:, :, 0] = img_bgra[:, :, 2]
        img_rgb[:, :, 1] = img_bgra[:, :, 1]
        img_rgb[:, :, 2] = img_bgra[:, :, 0]
        img_rgb = np.uint8(img_rgb)
        image = Image.fromarray(img_rgb, "RGB")
        img_draw = ImageDraw.Draw(image)
        myFont = ImageFont.truetype("FreeMono.ttf", 15)
        for index, crop in enumerate(bboxes):
            u1 = int(crop[0, 0])
            v1 = int(crop[0, 1])
            u2 = int(crop[1, 0])
            v2 = int(crop[1, 1])
            crop_bbox = [(u1, v1), (u2, v2)]
            vehicle_id = vehicle_id_list[index]
            if "walker" in id[vehicle_id]:
                img_draw.text(
                    (u1 + 2, v1 + 2), f"ID:{vehicle_id}", font=myFont, fill=(255, 0, 0)
                )
                img_draw.rectangle(crop_bbox, outline="blue")
            else:
                img_draw.text(
                    (u1 + 2, v1 + 2), f"ID: {vehicle_id}", font=myFont, fill=(255, 0, 0)
                )
                img_draw.rectangle(crop_bbox, outline="red")

        filename = path + "/out_rgb_bbox/%06d.png" % frame_number
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        image.save(filename)


# Use this function to get vehciles' snapshots that can be processed by auto_annotate() function.
def snap_processing(vehiclesActor, walkers, worldSnap):
    objects = []
    for v in vehiclesActor:
        vid = v.id
        vsnap = worldSnap.find(vid)
        if vsnap is None:
            continue
        vsnap.bounding_box = v.bounding_box
        vsnap.type_id = v.type_id
        objects.append(vsnap)

    for w in walkers:
        wid = w.id
        wsnap = worldSnap.find(wid)
        if wsnap is None:
            continue
        wsnap.bounding_box = w.bounding_box
        wsnap.type_id = w.type_id
        objects.append(wsnap)

    return objects


##########################################
# Use this function to deal with depth img
##########################################


def process_depth_data(data):
    """
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized
    """
    data = np.array(data.raw_data)
    data = data.reshape((data.height, data.width, 4))
    data = data.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth_meters = normalized_depth * 1000
    return depth_meters
