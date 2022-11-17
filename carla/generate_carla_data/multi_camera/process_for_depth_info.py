import os
import sys
import carla
import numbers
import numpy as np


def get_bb_data(vehicles, cam):
    bounding_boxes_vehicles = get_bounding_boxes(vehicles, cam)
    return bounding_boxes_vehicles

def get_bounding_boxes(vehicles, camera):
    """
    Creates 3D bounding boxes based on carla vehicle list and camera.
    """
    bounding_boxes = [get_bounding_box(vehicle, camera) for vehicle in vehicles]
    return bounding_boxes

def get_camera_intrinsic(sensor):
    VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    VIEW_FOV = int(float(sensor.attributes['fov']))
    calibration = np.identity(3)
    calibration[0, 2] = VIEW_WIDTH / 2.0
    calibration[1, 2] = VIEW_HEIGHT / 2.0
    calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    return calibration

def get_bounding_box(vehicle, camera):
    """
    Returns 3D bounding box for a vehicle based on camera view.
    """
    bb_cords = create_bb_points(vehicle)
    cords_x_y_z = vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    bbox = np.transpose(np.dot(get_camera_intrinsic(camera), cords_y_minus_z_x))
    camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
    return camera_bbox


def create_bb_points(vehicle):
    """
    Returns 3D bounding box for a vehicle.
    """
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

def vehicle_to_sensor(cords, vehicle, sensor):
    """
    Transforms coordinates of a vehicle bounding box to sensor.
    """
    world_cord = vehicle_to_world(cords, vehicle)
    sensor_cord = world_to_sensor(world_cord, sensor)
    return sensor_cord

def vehicle_to_world(cords, vehicle):
    """
    Transforms coordinates of a vehicle bounding box to world.
    """
    bb_transform = carla.Transform(vehicle.bounding_box.location)
    bb_vehicle_matrix = get_matrix(bb_transform)
    vehicle_world_matrix = get_matrix(vehicle.get_transform())
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords

def world_to_sensor(cords, sensor):
    """
    Transforms world coordinates to sensor.
    """
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)
    return sensor_cords

def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """
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

## new bounding box preocess

def process_rgb_img(vehicles, cam):
    bb = get_bb_data(vehicles, cam)
    return bb

######################################################
def apply_filters_to_3d_bb(bb_3d_data, depth_array, sensor_width, sensor_height, vehicle=None):
    # Bounding Box processing
    bb_3d_vehicles = bb_3d_data

    # Depth + bb coordinate check
    valid_bb_vehicles, valid_bb_vehicles_id = filter_bounding_boxes(bb_3d_vehicles, depth_array, sensor_width, sensor_height, actor='vehicle', vehicle=vehicle)

    return valid_bb_vehicles, valid_bb_vehicles_id

def filter_bounding_boxes(bb_data, depth_data, frame_width, frame_height, actor, vehicle=None):

    good_bounding_boxes = []
    good_bounding_boxes_id = []
    bb_data = np.array(bb_data)
    depth_data = np.transpose(depth_data)
    assert actor == "vehicle" or actor == "walker"
    if actor == "vehicle":
        bounds = [-0.40 * frame_width, 1.40 * frame_width, -0.40 * frame_height, 1.40 * frame_height]
    elif actor == "walker":
        bounds = [0, frame_width, 0, frame_height]

    for i, actor_bb_3d in enumerate(bb_data):
        # Apply some medium constraining on the data to not exclude every impossible point
        possible_bb_3d_points = np.array([x for x in actor_bb_3d if
                                          bounds[0] <= x[0] <= bounds[1] and bounds[2] <= x[1] <= bounds[3]])
        if len(possible_bb_3d_points) < 2:  # You can't have a box with only one point!
            continue
        # Transform out of boundaries points into possible points
        possible_bb_3d_points = adjust_points_to_img_size(frame_width, frame_height, possible_bb_3d_points)
        possible_bb_3d_points, bbox_exists, max_2d_area = get_4_points_max_2d_area(possible_bb_3d_points)

        if bbox_exists:
            xmin, ymin, xmax, ymax, visible_points = tighten_bbox_points(possible_bb_3d_points, depth_data)
            if all([isinstance(x, numbers.Number) for x in [xmin, ymin, xmax, ymax]]):
                tightened_bb_area = (xmax - xmin) * (ymax - ymin)
                tightened_bb_size_to_img = tightened_bb_area / (frame_height * frame_width)
                if tightened_bb_size_to_img > 2.5E-4:  # Experimental value
                    good_bounding_boxes.append(np.array([[xmin, ymin], [xmax, ymax]]))
                    good_bounding_boxes_id.append(vehicle[i])

    return good_bounding_boxes, good_bounding_boxes_id


def adjust_points_to_img_size(width, height, bb_3d_points):
    for xyz_point in bb_3d_points:
        if xyz_point[0] < 0:
            xyz_point[0] = 0
        if xyz_point[0] > width:
            xyz_point[0] = width - 1
        if xyz_point[1] < 0:
            xyz_point[1] = 0
        if xyz_point[1] > height:
            xyz_point[1] = height - 1
        xyz_point[0] = xyz_point[0]
        xyz_point[1] = xyz_point[1]
    return bb_3d_points


def tighten_bbox_points(possible_bb_3d_points, depth_data):
    visible_points_status, possible_bb_3d_points = check_visible_points(possible_bb_3d_points, depth_data)
    # No points with occlusion
    if visible_points_status.count(True) == 4 or visible_points_status.count(True) == 3:
        xmin, ymin, xmax, ymax = check_if_bbox_has_too_much_occlusion(possible_bb_3d_points, depth_data)
        color_idx = '3or4'

    # A pair of points occluded
    elif visible_points_status.count(True) == 2:
        xmin, ymin, xmax, ymax = get_bbox_for_2_visible_points(possible_bb_3d_points, depth_data, visible_points_status)
        color_idx = '2'

    elif visible_points_status.count(True) == 1:
        xmin, ymin, xmax, ymax = get_bbox_for_1_visible_point(possible_bb_3d_points, depth_data, visible_points_status)
        color_idx = '1'

    elif visible_points_status.count(True) == 0:
        xmin, ymin, xmax, ymax = None, None, None, None
        color_idx = '0'

    return xmin, ymin, xmax, ymax, color_idx


def check_visible_points(possible_bb_3d_points, depth_data):
    """
    visible_points=[True, True, False, False]
    """
    visible_points = []
    # Check which points are occluded
    for xyz_point in range(len(possible_bb_3d_points)):
        x = int(possible_bb_3d_points[xyz_point][0])
        y = int(possible_bb_3d_points[xyz_point][1])
        z = possible_bb_3d_points[xyz_point][2]
        depth_on_sensor = depth_data[x][y]
        point_visible = False
        if 0.0 <= z <= depth_on_sensor:
            point_visible = True
        visible_points.append(point_visible)
    return visible_points, possible_bb_3d_points


def get_4_points_max_2d_area(bb_3d_points):
    xmin = int(min(bb_3d_points[:, 0]))
    ymin = int(min(bb_3d_points[:, 1]))
    xmax = int(max(bb_3d_points[:, 0]))
    ymax = int(max(bb_3d_points[:, 1]))
    max_2d_area = (xmax - xmin) * (ymax - ymin)
    # If there is no area (e.g. its a line), then there is no bounding box!
    bbox_exists = True
    if xmin == xmax or ymin == ymax:
        bbox_exists = False
        return None, bbox_exists, 0

    # Getting the Z point that is closer to the camera (since we are extrapolating the min/max points anyways)
    # This way, more bbox points can be salvaged after the depth filtering (since they will be considered in front of
    # the object that the depth array is seeing)
    z = np.min(bb_3d_points[:, 2])
    bb_3d_points = np.array([[xmin, ymin, z], [xmin, ymax, z], [xmax, ymin, z], [xmax, ymax, z]])
    return bb_3d_points, bbox_exists, max_2d_area


def get_bbox_for_2_visible_points(possible_bb_3d_points, depth_data, points_occlusion_status):
    xmin, ymin, xmax, ymax = check_if_bbox_has_too_much_occlusion(possible_bb_3d_points, depth_data)
    return xmin, ymin, xmax, ymax


def get_bbox_for_1_visible_point(possible_bb_3d_points, depth_data, points_occlusion_status):
    xmin, ymin, xmax, ymax = check_if_bbox_has_too_much_occlusion(possible_bb_3d_points, depth_data)
    return xmin, ymin, xmax, ymax

def check_if_bbox_has_too_much_occlusion(possible_bb_3d_points, depth_data):
    # Either get the box as it is or don't get it at all, based on how much occlusion it has
    # Compute area to see how much of it is occluded
    # possible_bb_3d_points = [x, y, z], [x,y,z], [x,y,z], [x,y,z]
    xmin, ymin, xmax, ymax = compute_bb_coords(possible_bb_3d_points)
    common_depth = possible_bb_3d_points[0][2]
    depth_data_patch = depth_data[xmin:xmax, ymin:ymax]
    visible_points_count = (common_depth < depth_data_patch).sum()
    if visible_points_count / depth_data_patch.size > 0.1:
        return xmin, ymin, xmax, ymax
    else:
        return None, None, None, None


def compute_bb_coords(possible_bb_3d_points):
    # bbcoords = nparray([point1, point2, point3, point4])
    # point1,2,3, or 4 = nparray([x, y, z]])
    xmin = int(min(possible_bb_3d_points[:, 0]))
    ymin = int(min(possible_bb_3d_points[:, 1]))
    xmax = int(max(possible_bb_3d_points[:, 0]))
    ymax = int(max(possible_bb_3d_points[:, 1]))
    return xmin, ymin, xmax, ymax


def remove_bbs_too_much_IOU(input_bounding_boxes, input_bounding_boxes_id):
    bounding_boxes = []
    id = []
    for x in input_bounding_boxes:
        box = np.array(x[:-1])
        bounding_boxes.append(box)

    # bounding_boxes = np.array([x[:-1] for x in input_bounding_boxes])  # Removing the color index
    # If two bbs are overlapping too much, then we make a new bbox which takes the max size of the
    # union of both boxes
    if len(bounding_boxes) > 2:
        there_are_overlapping_boxes = True
        while there_are_overlapping_boxes:
            there_are_overlapping_boxes = False
            bb_idx = 0
            while bb_idx < len(bounding_boxes):
                bb_ref = bounding_boxes[bb_idx]
                bb_compared_idx = bb_idx + 1
                while bb_compared_idx < len(bounding_boxes):
                    bb_compared = bounding_boxes[bb_compared_idx]
                    # Compute intersection - Min of the maxes; max of the mins
                    xmax = min(bb_ref[1][0], bb_compared[1][0])
                    xmin = max(bb_ref[0][0], bb_compared[0][0])
                    ymin = max(bb_ref[0][1], bb_compared[0][1])
                    ymax = min(bb_ref[1][1], bb_compared[1][1])
                    # Check if there is intersection between the bbs
                    if (xmax-xmin) > 0 and (ymax-ymin) > 0:
                        intersection_area = (xmax - xmin + 1) * (ymax - ymin + 1)
                        bb_ref_area = (bb_ref[1][0] - bb_ref[0][0] + 1) * (bb_ref[1][1] - bb_ref[0][1] + 1)
                        bb_compared_area = (bb_compared[1][0] - bb_compared[0][0] + 1) * (bb_compared[1][1] - bb_compared[0][1] + 1)
                        IoU = intersection_area / (bb_compared_area + bb_ref_area - intersection_area)
                        if IoU > 0.90:  # Remove both bbs and get a new, bigger one
                            there_are_overlapping_boxes = True
                            xmin = min(bb_compared[0][0], bb_ref[0][0])
                            ymin = min(bb_compared[0][1], bb_ref[0][1])
                            xmax = max(bb_compared[1][0], bb_ref[1][0])
                            ymax = max(bb_compared[1][1], bb_ref[1][1])
                            bounding_boxes[bb_idx] = np.array([xmin, ymin], [xmax, ymax])
                            bounding_boxes = np.delete(bounding_boxes, (bb_compared_idx), axis=0)

                    bb_compared_idx += 1
                bb_idx += 1


    return bounding_boxes
