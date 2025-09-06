
# ==================================================
# SAFETY VIOLATION DETECTION + DJANGO ALERTS
# WITH 30-SECOND CLIP RECORDING
# ==================================================

from ultralytics import YOLO
import json
import cv2
import requests
import time
import os
import csv
import datetime as dt
import threading
from collections import deque
import face_recognition
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytesseract

# ----------------------
# CONFIGURATION
# ----------------------
MODEL_PATH = "C:/Users/yadav/OneDrive/Desktop/AIML project/runs/best.pt"
DJANGO_API_URL = "http://127.0.0.1:8000/api/alerts/create/"
CAMERA_ID = "CAM-01"

VIOLATION_MAP = {2: "NO_HELMET", 3: "NO_MASK", 4: "NO_SAFETY_VEST"}

ALERT_COOLDOWN_SECONDS = 30
SNAPSHOT_INTERVAL = 30
CLIP_SECONDS = 30  # each clip is 30 seconds
FPS = 20

# Directories
os.makedirs("snapshots", exist_ok=True)
os.makedirs("clips", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# CSV log
CSV_FILE = "logs/events.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "violation", "alert_sent", "snapshot", "clip",
            "people_detected", "image_caption", "detected_text"
        ])

# Buffers and timers
frame_buffer = deque(maxlen=FPS * CLIP_SECONDS)
last_alert_time = 0
last_snapshot_time = 0
clip_index = 1
clip_threads = []

# ----------------------
# FACE RECOGNITION SETUP
# ----------------------
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")

known_encodings = []
known_names = []

if os.path.exists(KNOWN_FACES_DIR):
    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
            print(f"✅ Loaded known face: {filename}")
        else:
            print(f"⚠️ No face found in {filename}, skipping.")
else:
    print(f"⚠️ Folder '{KNOWN_FACES_DIR}' not found. Face recognition disabled.")

def recognize_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
        face_names.append(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return face_names

# ----------------------
# SCENE CAPTIONING (BLIP)
# ----------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def get_image_caption(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ----------------------
# OCR SETUP
# ----------------------
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    raise FileNotFoundError(f"Tesseract not found at {TESSERACT_PATH}. Please install it.")

def extract_text(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        print(f"⚠️ OCR error: {e}")
        return ""

# ----------------------
# UTILITY FUNCTIONS
# ----------------------
def ts_str():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def log_event(violation, alert_sent, snapshot_path, clip_path, people_detected, caption, detected_text):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dt.datetime.now().isoformat(), violation, int(alert_sent),
            snapshot_path or "", clip_path or "", json.dumps(people_detected),
            caption, detected_text
        ])

def save_clip(frames, clip_path):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in frames:
        results = model.predict(source=f, device="cpu", verbose=False)
        annotated_frame = results[0].plot()
        recognize_faces(annotated_frame)
        out.write(annotated_frame)
    out.release()
    print(f"💾 Saved clip: {clip_path}")

def send_alert_to_django(frame, violation_type, clip_path=None):
    global last_alert_time
    current_time = time.time()
    if (current_time - last_alert_time) < ALERT_COOLDOWN_SECONDS:
        return False, None, None

    results = model.predict(source=frame, device="cpu", verbose=False)
    annotated_frame = results[0].plot()
    recognize_faces(annotated_frame)

    ok, buffer = cv2.imencode(".jpg", annotated_frame)
    if not ok:
        print("❌ Could not encode frame.")
        return False, None, None

    snapshot_path = f"snapshots/{violation_type}_{ts_str()}.jpg"
    cv2.imwrite(snapshot_path, annotated_frame)

    people_detected = recognize_faces(annotated_frame) or ["Unknown"]
    caption = get_image_caption(annotated_frame)
    detected_text = extract_text(annotated_frame)

    payload = {
        "violation_type": violation_type,
        "camera_id": CAMERA_ID,
        "clip_path": clip_path,
        "people_detected": json.dumps(people_detected),
        "image_caption": caption,
        "detected_text": detected_text
    }
    files = {"snapshot": ("snapshot.jpg", buffer.tobytes(), "image/jpeg")}

    try:
        resp = requests.post(DJANGO_API_URL, data=payload, files=files, timeout=10)
        if resp.status_code == 201:
            print(f"🚀 Alert '{violation_type}' sent!")
            last_alert_time = current_time
            return True, snapshot_path, people_detected
        else:
            print(f"❌ Error {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Django server error: {e}")

    return False, snapshot_path, people_detected

# ----------------------
# MAIN LOOP
# ----------------------
print("📦 Loading YOLO model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)  # Webcam

if not cap.isOpened():
    print("❌ Could not open video source.")
    exit()

print("✅ Running. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)

        # ----------------- Save 30-sec clip -----------------
        if len(frame_buffer) == FPS * CLIP_SECONDS:
            clip_path = f"clips/clip_{ts_str()}.mp4"
            t = threading.Thread(target=save_clip, args=(list(frame_buffer), clip_path))
            t.start()
            clip_threads.append(t)
            frame_buffer.clear()  # start next clip

        # Periodic snapshot
        if time.time() - last_snapshot_time >= SNAPSHOT_INTERVAL:
            results = model.predict(source=frame, device="cpu", verbose=False)
            annotated_snapshot = results[0].plot()
            recognize_faces(annotated_snapshot)
            path = f"snapshots/periodic_{ts_str()}.jpg"
            cv2.imwrite(path, annotated_snapshot)
            last_snapshot_time = time.time()

        # YOLO prediction
        results = model.predict(source=frame, device="cpu", verbose=False)
        violation = None
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id in VIOLATION_MAP:
                violation = VIOLATION_MAP[class_id]
                print(f"🚨 {violation} detected!")
                break

        if violation:
            # Save clip for violation (optional)
            clip_path = f"clips/{violation}_{ts_str()}.mp4"
            t = threading.Thread(target=save_clip, args=(list(frame_buffer), clip_path))
            t.start()
            clip_threads.append(t)

            # Send alert
            alert_sent, snapshot_path, people_detected = send_alert_to_django(frame, violation, clip_path)
            caption = get_image_caption(frame)
            detected_text = extract_text(frame)
            log_event(violation, alert_sent, snapshot_path, clip_path, people_detected, caption, detected_text)

        # Show live annotated frame
        annotated = results[0].plot()
        recognize_faces(annotated)
        cv2.imshow("YOLOv8 Monitor", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("🛑 Interrupted.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    for t in clip_threads:
        t.join()
    print("👋 Exiting. All clips saved successfully.")
