from ultralytics import YOLO
import cv2

human_animal_model = YOLO('runs/detect/train5/weights/Hu-An.pt')

def detect_and_mark(image_path, confidence_threshold=0.5):

    img = cv2.imread(image_path)

    results = human_animal_model(image_path)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            confidence = box.conf[0] 
            label = human_animal_model.names[cls] 

            if confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 255, 0) if label == 'human' else (0, 0, 255)  
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label_text = f'{label} {confidence:.2f}'
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                print(f"Detected: {label} with confidence {confidence:.2f}")

    cv2.imshow('Result', img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

image_path = r"C:\Users\ssk22\OneDrive\Documents\Chekrin project\download (2).jpeg"
detect_and_mark(image_path, confidence_threshold=0.5)
