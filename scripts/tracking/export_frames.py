import cv2
from pathlib import Path
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--video_path", help="Path to video", type=str, required=True
    )
    parser.add_argument("-s", '--skip_frame', type=int, default=0)
    parser.add_argument("-r", '--frame_range', type=float, default=(0, -1), nargs=2)
    parser.add_argument("-fn", '--file_number', type=int, default=1)
    parser.add_argument("-o", "--out_dir", help="Output directory", type=str, default="frames")
    args = parser.parse_args()

    video_path = args.video_path
    skip_frame = args.skip_frame
    frame_start, frame_end = args.frame_range
    file_number = args.file_number
    frame_number = frame_start
    frame_end = frame_end if frame_end > 0 else float('inf')
    out_dir = args.out_dir

    cap = cv2.VideoCapture(video_path)

    out_folder = Path(out_dir)
    out_folder.mkdir(parents=True, exist_ok=True)

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret and frame_number <= frame_end:
            # Print frame number
            print("Frame #", cap.get(cv2.CAP_PROP_POS_FRAMES))

            cv2.imwrite(f'{out_folder}/frame_{file_number:04d}.jpg', frame)
            file_number += 1
            frame_number += (skip_frame+1)  # i.e. at 30 fps, this advances one second
        else:
            cap.release()
            break
