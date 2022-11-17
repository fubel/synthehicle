import os
import time
import carla

from loguru import logger
from datetime import datetime

import argparse
import random
import queue
import numpy as np
import carla_vehicle_process as cva
from WeatherSelector import WeatherSelector
import integrate_txt_file
import img_2_video


def retrieve_data(sensor_queue, frame, timeout=1):
    while True:
        try:
            data = sensor_queue.get(True, timeout)
        except queue.Empty:
            return None
        if data.frame == frame:
            return data


def set_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--host",
        default="127.0.0.1",
        type=str,
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "--port", default=2000, type=int, help="TCP port to listen to (default: 2000)"
    )
    argparser.add_argument(
        "--tm_port",
        default=8000,
        type=int,
        help="port to communicate with TM (default: 8000)",
    )
    argparser.add_argument(
        "--town_info_path",
        # default='/media/CityFlow/data/scenes',
        default="../scenes",
        type=str,
        help="path to town information",
    )
    argparser.add_argument(
        "--map_name", default="Town10HD", type=str, help="name of map: Town01-07"
    )
    argparser.add_argument(
        "--fps", default=0.1, type=float, help="fps of generated data"
    )
    argparser.add_argument(
        "--overlap",
        help="set whether the cemera filed of view overlaps",
        action="store_true",
    )
    argparser.add_argument(
        "--save_video", default=False, type=bool, help="generate video file"
    )
    argparser.add_argument(
        "--save_lidar", default=False, type=bool, help="save lidar images"
    )
    argparser.add_argument(
        "--number-of-vehicles",
        default=250,
        type=int,
        help="number of vehicles (default: 150)",
    )
    argparser.add_argument(
        "--number_walker", default=30, type=int, help="number of walker (default: 20)"
    )
    argparser.add_argument(
        "--weather_option",
        default=3,
        type=int,
        help="0: Day, 1: Dawn, 2: Rain, 4: Night",
    )
    argparser.add_argument(
        "--distance_between_v",
        default=2.0,
        type=float,
        help="distance between vehicles",
    )
    argparser.add_argument("--max_dist", default=120,
                           type=int, help="lidar range")
    argparser.add_argument(
        "--resolution", default="1080p", type=str, help="resolution of generated images"
    )
    argparser.add_argument(
        "--number_of_frame", default=1800, type=int, help="number of frames generated"
    )
    argparser.add_argument(
        "--number_of_dangerous_vehicles",
        default=10,
        type=int,
        help="number of dangerous_vehicles",
    )
    argparser.add_argument("--neptune", action="store_true")
    argparser.add_argument(
        "--output_path", default="..", type=str, help="path for output data"
    )
    args = argparser.parse_args()
    return args


def main():
    args = set_args()
    logger.info(args)

    weather_dict = {
        0: "day",
        1: "dawn",
        2: "rain",
        3: "night",
    }

    vehicles_list = []
    nonvehicles_list = []
    num_cam = 0
    image_resolution = {"1080p": [1920, 1080],
                        "4k": [3840, 2160], "8k": [7680, 4320]}
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    client.load_world(args.map_name)
    logger.info("***** Loading map *****")
    world = client.get_world()

    if args.overlap:
        town_info_path = os.path.join(args.output_path, "scenes")
    elif not args.overlap:
        town_info_path = os.path.join(args.output_path, "scenes_non_overlap")

    town_path = f"{town_info_path}/{args.map_name}"
    weather_condition = weather_dict[args.weather_option]
    scence_path = f'{town_path}/{weather_condition}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'

    # Generate folders to save data
    logger.info("***** Generating folders *****")
    if not os.path.exists(os.path.dirname(scence_path)):
        os.makedirs(os.path.dirname(scence_path))

    files = os.listdir(f"{town_path}/camera_info")
    for file_name in files:
        if "camera" in file_name:
            num_cam += 1

    config_path = scence_path + "camera_name.txt"
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, "w") as config_file:
        for i in range(int(num_cam / 2)):
            config_file.write(f"C0{i+1}\n")
        config_file.close()

    camera_data_path = []
    save_file = []
    for i in range(int(num_cam / 2)):
        camera_data_path.append(f"{town_path}/camera_info/camera_{i + 1}.txt")
    for i in range(int(num_cam / 2)):
        save_file.append(scence_path + "C%02d" % (i + 1))
        file_name = save_file[i] + "/"
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

    try:
        # ------------------------
        # Generate traffic manager
        # ------------------------
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(
            args.distance_between_v)
        # --------------
        # Set weather
        # --------------
        # weather_options = WeatherSelector().get_weather_options()
        # WeatherSelector.set_weather(world, weather_options[args.weather_option])

        # ClearNoon - works fine without reflection issues
        weather_options = WeatherSelector().get_weather_options()
        WeatherSelector.set_weather(
            world, weather_options[args.weather_option])
        # ---------------------
        # Set synchronous mode
        # ---------------------
        logger.info("***** RUNNING in synchronous mode *****")
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = args.fps
            world.apply_settings(settings)
        else:
            synchronous_master = False
        # --------------
        # Get blueprints
        # --------------
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = f"Requested {args.number_of_vehicles} vehicles, but could only find {number_of_spawn_points} spawn points"
            logger.warning(msg, args.number_of_vehicles,
                           number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points[: args.number_of_vehicles]):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            # Taking out bicycles and motorcycles
            if int(blueprint.get_attribute("number_of_wheels")) > 2:
                if blueprint.has_attribute("color"):
                    color = random.choice(
                        blueprint.get_attribute("color").recommended_values
                    )
                    blueprint.set_attribute("color", color)
                if blueprint.has_attribute("driver_id"):
                    driver_id = random.choice(
                        blueprint.get_attribute("driver_id").recommended_values
                    )
                    blueprint.set_attribute("driver_id", driver_id)
                blueprint.set_attribute("role_name", "autopilot")
                batch.append(
                    SpawnActor(blueprint, transform).then(
                        SetAutopilot(FutureActor, True)
                    )
                )
                spawn_points.pop(0)

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logger.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        all_vehicles = world.get_actors().filter("vehicle.*")
        # set several of the cars as dangerous car
        for i in range(args.number_of_dangerous_vehicles):
            danger_car = all_vehicles[i]
            # crazy car ignore traffic light, do not keep safe distance, and very fast
            traffic_manager.ignore_lights_percentage(danger_car, 100)
            traffic_manager.distance_to_leading_vehicle(danger_car, 0)
            traffic_manager.vehicle_percentage_speed_difference(
                danger_car, -80)

        logger.info("Created %d vehicles" % len(vehicles_list))

        # -------------
        # Spawn Walkers
        # -------------
        # 1. take all the random locations to spawn
        walkers_list = []
        walker_spawn_points = []
        for i in range(args.number_walker):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc != None:
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        batch_walker = []
        for spawn_point in walker_spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            batch_walker.append(SpawnActor(walker_bp, spawn_point))

        results = client.apply_batch_sync(batch_walker, True)
        for i in range(len(results)):
            if results[i].error:
                logger.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})

        # 3. we spawn the walker controller
        batch_controller = []
        walker_controller_bp = world.get_blueprint_library().find(
            "controller.ai.walker"
        )
        for i in range(len(walkers_list)):
            batch_controller.append(
                SpawnActor(
                    walker_controller_bp, carla.Transform(
                    ), walkers_list[i]["id"]
                )
            )
        results = client.apply_batch_sync(batch_controller, True)
        for i in range(len(results)):
            if results[i].error:
                logger.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id

        # 4. we put altogether the walkers and controllers id to get the objects from their id
        all_id = []
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])

        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        # world.tick()
        # world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(
                world.get_random_location_from_navigation())
            # random max speed
            all_actors[i].set_max_speed(
                1 + random.random() / 2
            )  # max speed between 1 and 2 (default is 1.4 m/s)

        logger.info("Created %d walkers \n" % len(walkers_list))
        # -----------------------------
        # Spawn sensors
        # -----------------------------
        q_list = []
        idx = 0
        tick_queue = queue.Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)
        tick_idx = idx
        idx = idx + 1
        # -----------------------------------------
        # Get camera position and spawn rgb camera
        # -----------------------------------------
        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("sensor_tick", str(args.fps))
        # cam_bp.set_attribute('enable_postprocess_effects', str(False))
        cam_bp.set_attribute("image_size_x", str(
            image_resolution[args.resolution][0]))
        cam_bp.set_attribute("image_size_y", str(
            image_resolution[args.resolution][1]))

        num_camera = len(camera_data_path)
        for i in range(num_camera):
            cam_transform = cva.get_camera_position(camera_data_path[i])
            cam = world.spawn_actor(cam_bp, cam_transform)
            nonvehicles_list.append(cam)

            cam_queue = queue.Queue()
            cam.listen(cam_queue.put)
            q_list.append(cam_queue)
            cam_idx = idx
            idx = idx + 1

        logger.info("**** RGB camera ready ****")

        # --------------------
        # Spawn LIDAR sensor
        # --------------------

        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        lidar_bp.set_attribute("sensor_tick", str(args.fps))
        lidar_bp.set_attribute("channels", "356")
        lidar_bp.set_attribute("points_per_second", "22400000")
        lidar_bp.set_attribute("upper_fov", "50")
        lidar_bp.set_attribute("lower_fov", "-50")
        lidar_bp.set_attribute("range", "180")
        lidar_bp.set_attribute("rotation_frequency", "40")

        for i in range(num_camera):
            cam_transform = cva.get_camera_position(camera_data_path[i])

            lidar = world.spawn_actor(lidar_bp, cam_transform)
            nonvehicles_list.append(lidar)
            lidar_queue = queue.Queue()
            lidar.listen(lidar_queue.put)
            q_list.append(lidar_queue)
            lidar_idx = idx
            idx = idx + 1

        logger.info("**** LIDAR ready ****")

        # -------------------
        # Spawn depth sensor
        # -------------------

        depth_bp = world.get_blueprint_library().find("sensor.camera.depth")
        depth_bp.set_attribute("sensor_tick", str(args.fps))
        depth_bp.set_attribute(
            "image_size_x", str(image_resolution[args.resolution][0])
        )
        depth_bp.set_attribute(
            "image_size_y", str(image_resolution[args.resolution][1])
        )
        depth_bp.set_attribute("fov", "90")

        for i in range(num_camera):
            cam_transform = cva.get_camera_position(camera_data_path[i])
            depth_camera = world.spawn_actor(depth_bp, cam_transform)
            nonvehicles_list.append(depth_camera)

            depth_queue = queue.Queue()
            depth_camera.listen(depth_queue.put)
            q_list.append(depth_queue)
            depth_idx = idx
            idx = idx + 1
            logger.info("**** Depth camera ready ****")

        # -------------------
        # Spawn segmentation sensor
        # -------------------
        segm_bp = world.get_blueprint_library().find(
            "sensor.camera.semantic_segmentation"
        )
        segm_bp.set_attribute("sensor_tick", str(args.fps))
        segm_bp.set_attribute("image_size_x", str(
            image_resolution[args.resolution][0]))
        segm_bp.set_attribute("image_size_y", str(
            image_resolution[args.resolution][1]))
        segm_bp.set_attribute("fov", "90")

        iseg_bp = world.get_blueprint_library().find(
            "sensor.camera.instance_segmentation"
        )
        iseg_bp.set_attribute("sensor_tick", str(args.fps))
        iseg_bp.set_attribute("image_size_x", str(
            image_resolution[args.resolution][0]))
        iseg_bp.set_attribute("image_size_y", str(
            image_resolution[args.resolution][1]))
        iseg_bp.set_attribute("fov", "90")

        for i in range(num_camera):
            cam_transform = cva.get_camera_position(camera_data_path[i])
            segm_camera = world.spawn_actor(segm_bp, cam_transform)
            nonvehicles_list.append(segm_camera)

            seg_queue = queue.Queue()
            segm_camera.listen(seg_queue.put)
            q_list.append(seg_queue)
            segm_idx = idx
            idx = idx + 1
            logger.info("**** Semantic Segmentation camera ready ****")

        for i in range(num_camera):
            cam_transform = cva.get_camera_position(camera_data_path[i])
            iseg_camera = world.spawn_actor(iseg_bp, cam_transform)
            nonvehicles_list.append(iseg_camera)

            iseg_queue = queue.Queue()
            iseg_camera.listen(iseg_queue.put)
            q_list.append(iseg_queue)
            iseg_idx = idx
            idx = idx + 1
            logger.info("**** Instance Segmentation camera ready ****")

        # ---------------
        # Begin the loop
        # ---------------
        time_sim = 0
        frame_number = 0
        save_depth = True
        save_segm = True
        logs_path = os.path.join(scence_path, "logs.txt")
        logger.info("**** Begin the loop ****")
        while True:
            if frame_number == args.number_of_frame:
                break
            # Extract the available data
            t1 = time.time()
            nowFrame = world.tick()

            # Check whether it's time for sensor to capture data
            if time_sim >= 0.1:
                data = [retrieve_data(q, nowFrame) for q in q_list]
                assert all(x.frame == nowFrame for x in data if x is not None)

                # Skip if any sensor data is not available
                if None in data:
                    continue

                frame_number += 1

                vehicles_raw = world.get_actors().filter("vehicle.*")
                walker_raw = world.get_actors().filter("walker.*")
                snap = data[tick_idx]

                rgb_img = data[cam_idx - num_camera +
                               1: lidar_idx - num_camera + 1]
                lidar_img = data[
                    lidar_idx - num_camera + 1: depth_idx - num_camera + 1
                ]
                depth_img = data[depth_idx - num_camera +
                                 1: segm_idx - num_camera + 1]
                segm_img = data[segm_idx - num_camera +
                                1: iseg_idx - num_camera + 1]
                iseg_img = data[iseg_idx - num_camera + 1:]

                # Attach additional information to the snapshot
                vehicles = cva.snap_processing(vehicles_raw, walker_raw, snap)

                # Calculating visible bounding boxesw
                for i in range(num_camera):
                    cam = nonvehicles_list[i]
                    Lidar_img = lidar_img[i]
                    Rgb_img = rgb_img[i]
                    Depth_img = depth_img[i]
                    Segm_img = segm_img[i]
                    Iseg_img = iseg_img[i]

                    if save_depth:
                        save_path = save_file[i] + "/out_depth"
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        Depth_img.save_to_disk(
                            save_path + "/%06d.jpg" % frame_number,
                            carla.ColorConverter.LogarithmicDepth,
                        )

                    if save_segm:
                        save_path = save_file[i] + "/out_segm"
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        Segm_img.save_to_disk(
                            save_path + "/%06d.png" % frame_number,
                            carla.ColorConverter.CityScapesPalette,
                        )
                        save_path = save_file[i] + "/out_iseg"
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        Iseg_img.save_to_disk(
                            save_path + "/%06d.png" % frame_number,
                        )

                    filtered_out, _ = cva.get_all_data(
                        vehicles,
                        cam,
                        Lidar_img,
                        show_img=Rgb_img,
                        json_path="vehicle_class_json_file.txt",
                        path=save_file[i],
                        framenumber=frame_number,
                        max_dist=args.max_dist,
                        save_lidar=args.save_lidar,
                    )
                    # Save the results
                    cva.save_output(
                        frame_number,
                        Rgb_img,
                        filtered_out["vehicles_id"],
                        filtered_out["bbox"],
                        filtered_out["bbox_3d"],
                        filtered_out["world_coords"],
                        filtered_out["class"],
                        save_patched=False,
                        out_format="json",
                        path=save_file[i],
                    )
                    spend_time = time.time() - t1

                logger.info(
                    "Generate frame: %d  Spend time: %d s" % (
                        frame_number, spend_time)
                )

                with open(logs_path, "a+") as f:
                    f.write(f"Frame:{frame_number}, Spend time:{spend_time}\n")

                time_sim = 0
            time_sim = time_sim + settings.fixed_delta_seconds

    finally:
        try:
            for nonvehicle in nonvehicles_list:
                nonvehicle.stop()
        except:
            logger.info("Sensors has not been initiated")

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        logger.info("Destroying %d vehicles" % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x)
                           for x in vehicles_list])

        logger.info("Destroying %d NPC walkers" % len(walkers_list))
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        logger.info("Destroying %d nonvehicles" % len(nonvehicles_list))
        client.apply_batch([carla.command.DestroyActor(x)
                           for x in nonvehicles_list])

        # ---------------------------------------
        # Begin the Processing the generate data
        # ---------------------------------------

        if args.save_video:
            logger.info("Generating the Video...")
            img_2_video.img2video(save_file)
        time.sleep(2.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Finish!")
