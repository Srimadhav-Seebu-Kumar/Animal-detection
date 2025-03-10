# Wild Animal Detection System

## Overview
This project detects humans and animals in images using YOLO models. If a wild animal is detected, an email notification is sent to the specified recipient.

## Features
- Detects whether an entity in an image is a human or an animal.
- Classifies detected animals into specific categories.
- Identifies wild animals from a predefined list.
- Sends an email alert if a wild animal is detected.

## Dependencies
Ensure you have the following installed before running the script:

```bash
pip install ultralytics opencv-python smtplib numpy
```

## Models Used
- `Hu-An.pt`: YOLO model for detecting humans and animals.
- `ani-class.pt`: YOLO model for classifying animals.

## Setup
1. Clone the repository or download the script.
2. Place the YOLO model files (`Hu-An.pt` and `ani-class.pt`) in the project directory.
3. Update the `sender_email` and `sender_password` in `send_email_notification()` with valid credentials.

## Usage
To run the detection script, execute:

```python
python detect.py
```

Alternatively, integrate the function into your application:

```python
image_path = 'path/to/image.jpg'
owner_email = 'recipient@example.com'
detect_human_or_animal(image_path, owner_email)
```

## How It Works
1. Loads the YOLO model for human/animal detection.
2. If an animal is detected, it classifies the specific animal.
3. Checks if the detected animal is a wild animal.
4. Sends an email notification if a wild animal is detected.

## Example Output
```
Detected: animal with confidence 0.92
Detected specific animal: Bear with confidence 0.87
Wild animal detected: Bear
Email sent to recipient@example.com notifying about the wild animal.
```

## Email Notification
If a wild animal is detected, an email is sent with the following details:

**Subject:** Wild Animal Detected!  
**Body:** A wild animal (e.g., Bear) has been detected. Please take immediate action.

## Security Note
Avoid storing email credentials in plain text. Use environment variables or a secure configuration file.

## License
This project is licensed under the MIT License.

