# Axón Drift - Space Debris Detection


![AXONLOGO](https://github.com/user-attachments/assets/fad022aa-2984-4008-ae50-f8fc82dc522e)



## Description
Axón Drift is an innovative project that leverages artificial intelligence and space technologies to detect space debris, predict trajectories, assess collision risks, and plan evasion maneuvers. This project integrates machine learning models like YOLOv5, genetic algorithms, fuzzy logic, and Power BI visualization, all deployed using Azure tools.

### Key Features
- **Debris Detection**: Utilization of YOLOv5 to process images and real-time streaming.
- **Trajectory Prediction**: Orbit calculations using TLE (Two-Line Elements) data in Azure Machine Learning.
- **Risk Assessment**: Implementation of fuzzy logic in Azure Functions to evaluate collision risks.
- **Maneuver Planning**: Optimization with genetic algorithms in Azure Machine Learning.
- **Visualization**: Interactive dashboard in Power BI with automation via Azure Logic Apps.

## Requirements
- Python 3.8 or higher
- Libraries:
  - `tkinter`
  - `PIL`
  - `opencv-python` (cv2)
  - `torch` (for YOLOv5)
  - `tkinterweb`
  - `CairoSVG`
  - `pycairo`
- Git (to clone the repository)
- Access to a pre-trained YOLOv5 model (`best.pt`)
- IP Camera (optional, e.g., `http://192.168.1.68:8000/video`)
- Virtual environment (recommended)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Axon-Drift/Axon-Drift.git
   cd Axon-Drift
   cd src
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


4. **Download the YOLOv5 Model**:
   - Place the `best.pt` file in `models/yolo/debris_detector2/weights/`.

5. **Configure the Camera (Optional)**:
   - Adjust the URL in `gui.py` if using a different camera (line where `cv2.VideoCapture` is defined).

## Using `gui.py`

Follow these steps to run and use the Axón Drift graphical interface:

1. **Activate the Virtual Environment**:
   ```bash
   .venv\Scripts\activate
   ```

2. **Run the Script**:
   ```bash
   python gui.py
   ```
   - This will open a Tkinter window with the buttons "Process Image", "Process Video", and "Use Camera".

3. **Functionalities**:
   - **Process Image**:
     - Click "Process Image".
     - Select a `.jpg` or `.png` file using the file explorer.
     - The interface will process the image with YOLOv5, display the result in a pop-up window, and save predictions to `output/predictions/`.
     - A Power BI dashboard will open automatically during processing and close upon completion.
   - **Process Video**:
     - Click "Process Video".
     - Select an `.mp4` or `.avi` file.
     - The video will be processed, displayed in a pop-up window, and predictions will be saved locally.
     - The Power BI dashboard will open during processing.
   - **Use Camera**:
     - Click "Use Camera" to start streaming from the IP camera.
     - Real-time predictions will be displayed in a pop-up window.
     - Click "Stop Camera" to end the stream.
     - The Power BI dashboard will open when starting and close when stopping.


## Project Structure
- `gui.py`: Main graphical interface.
- `inference.py`: Detection logic with YOLOv5.
- `models/`: Folder for the `best.pt` model.
- `output/`: Folder for saving processed predictions and videos.
- `.gitignore`: Files ignored by Git (e.g., virtual environments).

## Contributions
- Feel free to open issues or pull requests on GitHub.
- Please follow style guidelines and document your changes.

## License
[Specify a license, e.g., MIT or GPL, if applicable. By default, no license.]

## Last Updated
June 19, 2025, 06:48 PM CST
