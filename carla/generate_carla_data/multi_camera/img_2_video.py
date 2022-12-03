import cv2
import os

def img2video(cam_list):
    cam = cam_list
    for i in range(len(cam)):

        image_folder = f'{cam[i]}/out_rgb_bbox'
        video_name = f'{cam[i]}/video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort()
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        video = cv2.VideoWriter(video_name, fourcc, 10, (width, height), True)

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

# def main():


# if __name__ == '__main__':
#     main()