import carla
from carla import ColorConverter as cc
import numpy as np
import os
import json
import time
import math
import queue

def process_img(image):
    image.convert(cc.Raw)
    image.save_to_disk('_out/camera_7.png')
    time.sleep(2)

def transform2M(transform):

    location, rotation = transform.location, transform.rotation
    x, y, z = location.x, location.y, location.z
    pitch, yaw, roll = math.radians(rotation.pitch), math.radians(rotation.yaw), math.radians(rotation.roll)

    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(pitch), 0, -math.sin(pitch)], [0, 1, 0], [math.sin(pitch), 0, math.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), math.sin(roll)], [0, -math.sin(roll), math.cos(roll)]])
    R = Rz.dot(Ry).dot(Rx)
    t = np.array([[x], [y], [z]])

    M = np.hstack((R, t))
    M = np.vstack((M, np.array([[0,0,0,1]])))
    return M

def get_camera_intrinsic(sensor):
    VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    VIEW_FOV = int(float(sensor.attributes['fov']))
    calibration = np.identity(3)
    calibration[0, 2] = VIEW_WIDTH / 2.0
    calibration[1, 2] = VIEW_HEIGHT / 2.0
    calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    return calibration

def get_camera_homography(sensor):

    change = np.array([[0,  1,  0],
                       [0,  0, -1],
                       [1,  0,  0]])
    intrinsic_matrix = get_camera_intrinsic(sensor)
    extrinsic_matrix = np.array(sensor.get_transform().get_inverse_matrix())

    standard_change = np.dot(change, np.delete(extrinsic_matrix, 2, 1)[:-1, :])

    homography_matrix = np.dot(intrinsic_matrix, standard_change)

    result = (homography_matrix / homography_matrix[-1, -1].reshape((-1, 1)))

    return result

def get_extrinsic_matrix(sensor):

    extrinsic_matrix = np.array(sensor.get_transform().get_inverse_matrix())

    return extrinsic_matrix


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

def get_camera_info(transform):
    location, rotation = transform.location, transform.rotation
    x, y, z = location.x, location.y, location.z
    pitch, yaw, roll = math.radians(rotation.pitch), math.radians(rotation.yaw), math.radians(rotation.roll)
    return x , y, z, pitch, yaw, roll

def retrieve_data(sensor_queue, frame, timeout=1):
    while True:
        try:
            data = sensor_queue.get(True, timeout)
        except queue.Empty:
            return None
        if data.frame == frame:
            return data

def main():

    path = './_out'
    nonvehicles_list = []
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)

    # client.load_world('Town10HD')
    world = client.get_world()

    try:
        spectator = world.get_spectator().get_transform()
        settings = world.get_settings()
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.fixed_delta_seconds = 0.1
            settings.synchronous_mode = True
            world.apply_settings(settings)
        else:
            synchronous_master = False

        q_list = []
        tick_queue = queue.Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)

        bp_library = world.get_blueprint_library()

        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('sensor_tick', str(0.1))
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('enable_postprocess_effects', 'True')
        camera = world.spawn_actor(camera_bp, spectator)
        nonvehicles_list.append(camera)
        camera.listen(process_img)

        cam_queue = queue.Queue()
        q_list.append(cam_queue)

        frame_number = 0
        time_sim = 0
        while True:
            if frame_number == 1:
                break

            nowFrame = world.tick()
            if time_sim >= 0.1:
                frame_number += 1
                x, y, z, pitch, yaw, roll = get_camera_info(camera.get_transform())
                intrinsic_matrix = get_camera_intrinsic(camera)
                extrinsic_matrix = get_extrinsic_matrix(camera)
                out_dict = {'intrinsic_matrix': intrinsic_matrix.tolist(),
                                'extrinsic_matrix': extrinsic_matrix.tolist(),
                                'x': x,
                                'y': y,
                                'z': z,
                                'pitch': pitch,
                                'yaw': yaw,
                                'roll': roll
                                }

                filename = '_out/camera_7.txt'
                if not os.path.exists(os.path.dirname(filename)):
                    os.makedirs(os.path.dirname(filename))

                with open(filename, 'w') as outfile:
                    json.dump(out_dict, outfile, indent=4)
                time_sim = 0
            time_sim += 0.1

    finally:
        try:
            time.sleep(2)
            nonvehicles_list[0].stop()
        except:
            print('Sensors has not been initiated')

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        client.apply_batch([carla.command.DestroyActor(x) for x in nonvehicles_list])
        print('Exit')


if __name__ == '__main__':
    main()
