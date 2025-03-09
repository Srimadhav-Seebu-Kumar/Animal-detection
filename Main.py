import cv2
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO

human_animal_model = YOLO('Hu-An.pt')

animal_classification_model = YOLO('ani-class.pt')

wild_animals = ['Bear', 'Deer', 'Elephant', 'Fox', 'Monkey']

def detect_human_or_animal(image_path, owner_email):
    
    img = cv2.imread(image_path)

    results = human_animal_model(image_path)

    # Process the result
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  
            confidence = box.conf[0]  
            
            #Confidence score threshold
            if confidence >= 0.50:
                label = human_animal_model.names[cls]  

                print(f"Detected: {label} with confidence {confidence:.2f}")

                if label == 'animal':
                    animal = classify_animal(image_path)

                    if animal in wild_animals:
                        print(f"Wild animal detected: {animal}")
                        send_email_notification(owner_email, animal)
                    else:
                        print(f"Detected non-wild animal: {animal}")

def classify_animal(image_path):
    animal_results = animal_classification_model(image_path)

    for animal_result in animal_results:
        for box in animal_result.boxes:
            animal_cls = int(box.cls[0])  
            animal_conf = box.conf[0]  
            
            #Confidence score threshold
            if animal_conf >= 0.50:
                animal_label = animal_classification_model.names[animal_cls] 

                print(f"Detected specific animal: {animal_label} with confidence {animal_conf:.2f}")
                return animal_label

def send_email_notification(owner_email, animal_name):
    # Configure your email 
    sender_email = "e2262986@gmail.com"
    sender_password = "Abcdef@23"
    subject = "Wild Animal Detected!"
    body = f"A wild animal ({animal_name}) has been detected. Please take immediate action."
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = owner_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, owner_email, msg.as_string())
        server.quit()
        print(f"Email sent to {owner_email} notifying about the wild animal.")
    except Exception as e:
        print(f"Failed to send email: {e}")

#TEST
image_path = 'download (2).jpeg'
owner_email = 'ssk222325@gmail.com'
detect_human_or_animal(image_path, owner_email)
