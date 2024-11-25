import os
import torch
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from PIL import Image as PILImage
import torchvision.transforms as trn
import cv2
import logging
from scipy.signal import savgol_filter
import zmq
import av
from threading import Thread
import time
import json
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttentionExtractor:
    def __init__(self, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        logging.info(f"Initializing AttentionExtractor with model: {model_name}")
        self.model = TimesformerForVideoClassification.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.id2label = self.model.config.id2label
        logging.info("AttentionExtractor initialized successfully.")

    def extract_attention(self, frames):
        logging.info(f"Extracting attention and logits for {len(frames)} frames.")
        inputs = self.image_processor(frames, return_tensors="pt").to(self.device)
        logging.info(f"Input tensor shape: {inputs['pixel_values'].shape}")  # Debug input shape
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        last_layer_attention = outputs.attentions[-1]
        logging.info(f"Last layer attention shape: {last_layer_attention.shape}")  # Debug attention shape
        spatial_attention = last_layer_attention.mean(1)
        logging.info(f"Logits shape: {outputs.logits.shape}")  # Debug logits shape
        return spatial_attention.cpu().numpy(), outputs.logits.cpu().numpy()

    def apply_attention_heatmap(self, frame, attention, predicted_label):
        logging.info(f"Applying attention heatmap for predicted label: {self.id2label[predicted_label]}")
        att_map = attention[1:].reshape(int(np.sqrt(attention.shape[0] - 1)), -1)
        att_resized = cv2.resize(att_map, (frame.shape[1], frame.shape[0]))
        att_norm = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * att_norm), cv2.COLORMAP_JET)
        blend = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        
        # Adjust font size dynamically based on frame height
        font_scale = frame.shape[0] / 500.0
        font_thickness = max(1, int(font_scale))
        label_text = f"Predicted: {self.id2label[predicted_label]}"
        
        # Add text to the bottom-left corner with padding
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = 10
        text_y = frame.shape[0] - 10
        cv2.putText(blend, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        return blend

class VideoServer:
    def __init__(self, model_name, output_dir, port=6000, client_port=6001):
        logging.info(f"Starting VideoServer on port {port} and client port {client_port}.")
        self.extractor = AttentionExtractor(model_name)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.context = zmq.Context()
        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.bind(f"tcp://*:{port}")

        self.send_socket = self.context.socket(zmq.PUSH)
        self.send_socket.bind(f"tcp://*:{client_port}")

        self.aggregate_attention_data = []
        self.aggregate_logits = [] 
        self.running = True
        self.frame_buffer = []
        self.attention_buffer = []
        self.all_logits = []
        self.frame_count = 0
        logging.info("VideoServer initialized successfully.")

    def preprocess_frame(self, frame, config):

        logging.info(f"Preprocessing frame with config: {config}")

        pil_img = PILImage.fromarray(frame)  # Convert frame to PIL Image

        # Apply resizing
        if config.get("resize", False):
            resized_img = trn.Resize(256)(pil_img)
            frame = np.array(resized_img)
            logging.info("Resized frame to 256.")

        # Apply cropping
        if config.get("crop", False):
            cropped_img = trn.CenterCrop(224)(PILImage.fromarray(frame))
            frame = np.array(cropped_img)
            logging.info("Cropped frame to 224x224.")

        # Apply adversarial noise
        if config.get("adversarial") == "noise":
            logging.info("Adding shot noise to the frame.")
            z = np.array(frame, copy=True) / 255.0
            noisy_img = np.clip(np.random.poisson(z * 300) / 300.0, 0, 1)
            frame = np.uint8(255 * noisy_img)

        return frame

    def receive_frames(self):
        logging.info("Waiting to receive frames from client...")
        while self.running:
            message = self.recv_socket.recv_pyobj()
            if message == "LAST FRAME":
                logging.info("Received 'LAST FRAME' signal. Processing completed for this video.")
                
                # Convert attention_buffer to JSON-serializable format
                serializable_attention_buffer = []
                for item in self.attention_buffer:
                    serializable_attention_buffer.append({
                        "sequence_index": item["sequence_index"],
                        "original": {
                            "max_attention": float(item["original"]["max_attention"]),
                            "min_attention": float(item["original"]["min_attention"]),
                            "mean_attention": float(item["original"]["mean_attention"]),
                            "predicted_label": item["original"]["predicted_label"],
                        },
                        "processed": {
                            "max_attention": float(item["processed"]["max_attention"]),
                            "min_attention": float(item["processed"]["min_attention"]),
                            "mean_attention": float(item["processed"]["mean_attention"]),
                            "predicted_label": item["processed"]["predicted_label"],
                        },
                    })

                # Save attention data to a JSON file
                output_file = os.path.join(self.output_dir, "attention_data_full_video.json")
                with open(output_file, "w") as f:
                    import json
                    json.dump(serializable_attention_buffer, f, indent=4)
                logging.info(f"Full video attention data saved to {output_file}")

                # Send the attention data JSON to the client
                try:
                    self.send_socket.send_pyobj("LAST FRAME")
                    self.send_socket.send_pyobj({"type": "attention_data", "content": serializable_attention_buffer})
                    logging.info("Attention results sent to the client successfully.")
                except Exception as e:
                    logging.error(f"Failed to send attention results to the client. Error: {e}")

                continue

            logging.info("Frames received from client.")
            self.process_frames(message)

    def process_frames(self, message):

        # Append the received frames to the frame buffer
        frame, config = message["frame"], message["config"]
        self.frame_buffer.append((frame, config))
        batch_size = 8  # Timesformer batch size

        # Process when enough frames are accumulated
        while len(self.frame_buffer) >= batch_size:
            batch = self.frame_buffer[:batch_size]
            self.frame_buffer = self.frame_buffer[batch_size:]

            # Separate frames and configurations
            frames = [item[0] for item in batch]
            configs = [item[1] for item in batch]

            logging.info("Preprocessing batch of frames.")

            # Preprocess all frames in the batch based on their respective configurations
            preprocessed_frames = [self.preprocess_frame(frame, config) for frame, config in zip(frames, configs)]

            logging.info("Classifying original frames.")
            # Classify original frames (entire sequence)
            spatial_attention_original, logits_original = self.extractor.extract_attention(frames)
            predicted_label_original = int(np.argmax(logits_original))  # Sequence-level prediction
            attention_original = spatial_attention_original.mean(axis=0)  # Average attention across heads

            logging.info("Classifying preprocessed frames.")
            # Classify preprocessed frames (entire sequence)
            spatial_attention_processed, logits_processed = self.extractor.extract_attention(preprocessed_frames)
            predicted_label_processed = int(np.argmax(logits_processed))  # Sequence-level prediction
            attention_processed = spatial_attention_processed.mean(axis=0)  # Average attention across heads

            # Collect attention data for the sequence
            attention_data = {
                "sequence_index": self.frame_count // batch_size,
                "original": {
                    "max_attention": attention_original.max(),
                    "min_attention": attention_original.min(),
                    "mean_attention": attention_original.mean(),
                    "predicted_label": predicted_label_original,
                },
                "processed": {
                    "max_attention": attention_processed.max(),
                    "min_attention": attention_processed.min(),
                    "mean_attention": attention_processed.mean(),
                    "predicted_label": predicted_label_processed,
                },
            }
            self.attention_buffer.append(attention_data)

            # Send individual frames with heatmaps as before
            for i in range(batch_size):
                # Generate heatmaps for original and preprocessed frames
                classified_original_frame = self.extractor.apply_attention_heatmap(
                    frames[i], attention_original[-1], predicted_label_original
                )
                classified_processed_frame = self.extractor.apply_attention_heatmap(
                    preprocessed_frames[i], attention_processed[-1], predicted_label_processed
                )

                logging.info(f"Sending processed frame {self.frame_count + i} to client.")
                self.send_socket.send_pyobj((
                    frames[i],
                    classified_original_frame,
                    classified_processed_frame,
                ))

            self.frame_count += batch_size

        logging.info("Batch processing completed.")

    def run(self):
        logging.info("VideoServer is now running.")
        Thread(target=self.receive_frames).start()
        while self.running:
            time.sleep(1)
        logging.info("VideoServer has shut down.")


if __name__ == "__main__":
    server = VideoServer('facebook/timesformer-base-finetuned-k400', 'Realtime')
    logging.info("Server started on port 6000.")
    server.run()