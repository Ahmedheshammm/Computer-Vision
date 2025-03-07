# Model Deployment Using Flask

This repository contains a Flask-based model deployment for water segmentation. Follow the steps below to set up and run the project.

## Prerequisites
- Python 3.10
- pip (Python package installer)
- Internet connection to download model files

## Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Install Dependencies
Install the required Python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Download the Model File
Download the model file from the Google Drive link provided in the `models` folder description.

- Place the downloaded model file inside the `models` folder in the root directory of the project.

### 4. Run the Application
Start the Flask application by running:
```bash
python3.10 app.py
```

The application will start on `http://127.0.0.1:5000/` by default.

## API Endpoints
| Endpoint       | Method | Description          |
|---------------|-------|--------------------|
| `/predict`    | POST  | Returns prediction results |

### Example Request
```bash
curl -X POST -F "file=@image.jpg" http://127.0.0.1:5000/predict
```

### Example Response
```json
{
  "prediction": "Class_Name",
  "confidence": 0.95
}
```

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
- Flask
- TensorFlow/PyTorch (if applicable)
- Streamlit (if applicable)

For any questions or support, please contact [Your Email].


