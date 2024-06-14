# Car Damage Detection 🚗💥

This is a project for detecting car damages using object detection techniques. The application allows users to detect damages in images and videos.

## Introduction
Car Damage Detection is an application that utilizes object detection algorithms to identify damages on cars. It provides options for users to either use their webcam, upload an image, or upload a video for analysis.

## Features
- Detect car damages in real-time using webcam.
- Upload images or videos for damage detection.
- Adjust confidence and NMS thresholds for detection accuracy.

## Libraries
The project utilizes the following Python libraries:

- OpenCV
- NumPy
- av
- Streamlit
- Pillow

## Files Used
- **test2.py**: Python script containing the main application logic and Streamlit interface.
- **car_damage_model(17 classes).onnx**: Pre-trained object detection model for detecting car damages.
- **coco.names**: File containing the names of COCO dataset classes used by the model.
- **requirements.txt**: Text file listing all the Python dependencies required to run the application.
- **README.md**: Markdown file providing an overview of the project, setup instructions, usage guidelines, and other relevant information.

## Setup
1. Clone the repository:
https://github.com/AIOnGraph/Detect-Car-Damage.git
2. Install dependencies:
pip install -r requirements.txt

## Usage
1. Run the application:
streamlit run main.py
2. Select one of the available options:
- **Webcam**: Utilize your webcam for real-time damage detection.
- **Upload Image**: Upload an image for damage detection.
- **Upload Video**: Upload a video for damage detection.

## Contributing
Contributions are welcome! If you want to contribute to this project, feel free to open an issue or create a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
