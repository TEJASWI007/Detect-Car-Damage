import cv2
import numpy as np
import av
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
import io
import tempfile
import os
import time
# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1

# Colors.
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)

class ObjectDetectionProcessor(VideoProcessorBase):
    
    def __init__(self, confidence_threshold, nms_threshold):
        self.model_weights = "car_damage_model(17 classes).onnx"
        self.coco_names = "./model/coco.names"
        self.net = cv2.dnn.readNet(self.model_weights)
        self.classes = self.load_classes(self.coco_names)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        # print("__init__", self.confidence_threshold, 1111111111111111111111111111111111111)

    def load_classes(self, file_path):
        with open(file_path, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        return classes

    def draw_label(self, im, label, x, y):
        # if label:
            text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
            dim, baseline = text_size[0], text_size[1]
            cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), BLACK, cv2.FILLED)
            cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)
        
    def pre_process(self, input_image):
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # print(111111111111111111111111111111111111111111)
        return outputs

    def post_process(self, input_image, outputs):
        class_ids = []
        confidences = []
        boxes = []

        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        x_factor, y_factor = image_width / INPUT_WIDTH, image_height / INPUT_HEIGHT

        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]

            if confidence >= self.confidence_threshold:
                # print("Post_process", self.confidence_threshold, 2222222222222222222222222222)
                classes_scores = row[5:]
                class_id = np.argmax(classes_scores)

                if classes_scores[class_id] > self.confidence_threshold:
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        # print(2222222222222222222222222222222222222222222222222222222222222222222222222222222222)
        for i in indices:
            box = boxes[i]
            left, top, width, height = box
            # print(f"Detected: {self.classes[class_ids[i]]} - Confidence: {confidences[i]}")

            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)

            label = "{}: {:.2f}".format(self.classes[class_ids[i]], confidences[i])
            self.draw_label(input_image, label, left, top)

        return input_image

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            frame_np = np.array(frame.to_image())
            # print(3333333333333333333333333333333333333333333333333333333333333333333333333)
            detections = self.pre_process(frame_np)
            result_frame = self.post_process(frame_np.copy(), detections)
            return av.VideoFrame.from_ndarray(result_frame, format='rgb24')
        except Exception as e:
            print("Error in recv:", e)
            return frame


def process_uploaded_image(file, processor):
    image_bytes = file.read()
    image_np = np.array(Image.open(io.BytesIO(image_bytes)))
    # st.image(image_np, caption="Original Image", use_column_width=True)
    detections = processor.pre_process(image_np)
    # print(112121212312321322132121321211221212121212121212121)
    result_image = processor.post_process(image_np.copy(), detections)
    # print(565656565656556556566556565656565656565656565656565656)
    with st.spinner('Wait for it...'):
        time.sleep(2)
        st.image(result_image, caption="Processed Image", use_column_width=True)
        st.success('Done!')
    if st.button('Three cheers'):
        st.toast('Hip!')
        time.sleep(.5)
        st.toast('Hip!')
        time.sleep(.5)
        st.toast('Hooray!', icon='🎉')


def process_uploaded_video(file, processor):
    st.sidebar.write("Processing uploaded video...")
    video_bytes = file.read()

    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_file_path = temp_video_file.name
    temp_video_file.write(video_bytes)

    video_uploaded = cv2.VideoCapture(temp_video_file_path)
    
    # Get the original video's dimensions
    frame_width = int(video_uploaded.get(3))
    frame_height = int(video_uploaded.get(4))

    # Define the codec and create VideoWriter object
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (frame_width, frame_height))

    progress_text = "Operation in progress. Please wait."
    with st.progress(0, text=progress_text):
        try:
            while True:
                ret, frame = video_uploaded.read()
                if not ret:
                    break

                detections = processor.pre_process(frame)
                result_frame = processor.post_process(frame.copy(), detections)

                video_writer.write(result_frame)

                # Display the processed frame using Streamlit
                st.image(result_frame, channels="BGR")  # Assuming frame is in BGR format

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # video_uploaded.release()
    # video_writer.release()

    # # Display the processed video
    # st.video(output_video_path)

    # # Close and remove the temporary files
    # temp_video_file.close()
    # os.remove(temp_video_file_path)
    # os.remove(output_video_path)


def main():
    st.title("**DETECT YOUR CAR DAMAGES 🚗💥**")

    st.sidebar.title("Object Detection Options")
    st.write("Damage Detection Options: ")
    option = st.radio(
            "input"
            , ["📷 Webcam", "🖼️ Upload Image", "🎥 Upload Video"],
            horizontal=True,label_visibility='hidden')

    # webrtc_ctx = None
    # print(webrtc_ctx,111111111111111111111111111111111111111111111)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.05)


    rtc_configuration = {"iceServers": [{"urls": "turn:relay1.expressturn.com:3478", "username": "efPU52K4SLOQ34W2QY", "credential": "1TJPNFxHKXrZfelz"}]}

    processor = ObjectDetectionProcessor(confidence_threshold, nms_threshold)

    if option == "📷 Webcam":
        st.sidebar.write("Using Webcam")
        try:
            # print(000000000000000000000000000000000000000000000000000000000000000)
            webrtc_ctx = webrtc_streamer(
                key="example",
                video_processor_factory=lambda: processor,
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration=rtc_configuration,
                async_processing=True,
                # video_frame_callback=processor.recv
            )
            print(webrtc_ctx,)
        except Exception as e:
            st.error(f"An error occurred: {e}")
        if webrtc_ctx and webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
            webrtc_ctx.video_processor.nms_threshold = nms_threshold
            # print(webrtc_ctx.video_processor,4444444444444444444444444444444444444444444444444444444444)
            # if webrtc_ctx.video_processor:
            #     # print(8888888888888888888888888888888888888888888888888888888888)
            #     webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
            #     webrtc_ctx.video_processor.nms_threshold = nms_threshold
    elif option == "🖼️ Upload Image":
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            process_uploaded_image(uploaded_image, processor)
    elif option == "🎥 Upload Video":
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv"])
        if uploaded_video:
            process_uploaded_video(uploaded_video, processor)

if __name__ == "__main__":
    main()
