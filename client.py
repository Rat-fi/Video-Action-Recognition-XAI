import argparse
import logging
import cv2
import numpy as np
import zmq
import time
from threading import Thread
import av


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoClient:
    def __init__(self, server_address, client_address, video_path, resize, crop, adversarial):
        self.context = zmq.Context()

        # Socket to send frames
        self.send_socket = self.context.socket(zmq.PUSH)
        self.send_socket.connect(server_address)

        # Socket to receive processed frames
        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.connect(client_address)

        self.running = True
        self.video_path = video_path

        # Preprocessing configuration
        self.resize = resize.lower() == "on"
        self.crop = crop.lower() == "on"
        self.adversarial = adversarial.lower()

    def send_video(self):
        preprocessing_config = {
            "resize": self.resize,
            "crop": self.crop,
            "adversarial": self.adversarial,
        }

        container = av.open(self.video_path)
        for frame in container.decode(video=0):
            if not self.running:
                break
            frame_rgb = frame.to_rgb().to_ndarray()

            # Send frame and configuration as a dictionary
            self.send_socket.send_pyobj({"frame": frame_rgb, "config": preprocessing_config})

        self.send_socket.send_pyobj("LAST FRAME")
        container.close()

    def display_frames(self):
        while self.running:
            try:
                logging.info("Waiting to receive frames from server...")
                message = self.recv_socket.recv_pyobj()

                # Handle frame-related messages
                if isinstance(message, tuple) and len(message) == 3:
                    original_frame, classified_original_frame, classified_processed_frame = message
                    logging.info("Received frame data.")

                    max_height = max(original_frame.shape[0], classified_original_frame.shape[0], classified_processed_frame.shape[0])

                    def pad_frame(frame, target_height):
                        if frame.shape[0] < target_height:
                            padding = target_height - frame.shape[0]
                            top_padding = padding // 2
                            bottom_padding = padding - top_padding
                            return cv2.copyMakeBorder(frame, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        return frame

                    # Pad frames to the same height
                    original_frame_padded = pad_frame(original_frame, max_height)
                    classified_original_padded = pad_frame(classified_original_frame, max_height)
                    classified_processed_padded = pad_frame(classified_processed_frame, max_height)

                    # Combine frames horizontally
                    combined_frame = np.hstack((
                        cv2.cvtColor(original_frame_padded, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(classified_original_padded, cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(classified_processed_padded, cv2.COLOR_RGB2BGR),
                    ))

                    # Display the final combined frame
                    label_bar_height = 40
                    label_bar = np.zeros((label_bar_height, combined_frame.shape[1], 3), dtype=np.uint8)

                    font_scale = 0.5
                    font_thickness = 1
                    label_color = (255, 255, 255)
                    text_y = label_bar_height - 15

                    width_original = original_frame_padded.shape[1]
                    width_classified_original = classified_original_padded.shape[1]
                    width_classified_processed = classified_processed_padded.shape[1]

                    text_x_original = width_original // 2 - 50
                    text_x_classified_original = width_original + (width_classified_original // 2) - 100
                    text_x_classified_processed = width_original + width_classified_original + (width_classified_processed // 2) - 125

                    cv2.putText(label_bar, "Original Frame", (text_x_original, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness)
                    cv2.putText(label_bar, "Explained Original Frame", (text_x_classified_original, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness)
                    cv2.putText(label_bar, "Explained Attacked Frame", (text_x_classified_processed, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness)

                    final_display = np.vstack((label_bar, combined_frame))

                    cv2.imshow("Combined Frames", final_display)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

                # Break out if non-frame message (e.g., attention_data or LAST FRAME)
                elif isinstance(message, dict) and message.get("type") == "attention_data":
                    logging.info("Final attention data received during display_frames; deferring to receive_results.")
                    break
                elif message == "LAST FRAME":
                    logging.info("Received 'LAST FRAME' signal during display_frames.")
                    break

            except Exception as e:
                logging.error(f"Error receiving frames: {e}")
                self.running = False


    def receive_results(self):
        while True:
            try:
                logging.info("Waiting for results from the server...")
                message = self.recv_socket.recv_pyobj()

                # Check for attention data type
                if isinstance(message, dict) and message.get("type") == "attention_data":
                    # Full attention data received
                    attention_data = message["content"]
                    
                    # Save the attention data to a JSON file
                    with open("received_attention_data.json", "w") as f:
                        import json
                        json.dump(attention_data, f, indent=4)
                    
                    logging.info("Received and saved full attention data as 'received_attention_data.json'.")

                    # Break after processing the full attention data
                    break

                else:
                    # Log unexpected message formats
                    logging.warning("Received unexpected message format in receive_results.")
            except Exception as e:
                logging.error(f"Error receiving results: {e}")
                break

    def run(self):
        # Start sending video frames
        sender_thread = Thread(target=self.send_video)
        sender_thread.start()

        # Display frames in real-time
        self.display_frames()

        # After all frames are displayed, process final results (attention data)
        self.receive_results()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Client")
    parser.add_argument("-video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("-server", type=str, required=True, help="Server's ip address")
    parser.add_argument("-resize", type=str, default="off", help="Enable resizing: on/off")
    parser.add_argument("-crop", type=str, default="off", help="Enable cropping: on/off")
    parser.add_argument("-adversarial", type=str, default="none", help="Adversarial attack: none/noise/blur/crbr")
    args = parser.parse_args()

    server_address = f"tcp://{args.server}:6000"
    client_address = f"tcp://{args.server}:6001"
    video_path = args.video

    client = VideoClient(server_address, client_address, video_path, args.resize, args.crop, args.adversarial)
    client.run()
