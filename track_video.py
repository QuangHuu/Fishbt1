import cv2
from ultralytics import YOLO
import numpy as np
import time
import yaml
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from pymongo import MongoClient
from datetime import datetime

# --- Final Optimized Configuration for the tracker ---
tracker_config = {
    'tracker_type': 'bytetrack',
    'reid_model': 'osnet_x0_25_msmt17.pt',   
    'track_high_thresh': 0.5,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.4,
    'track_buffer': 50,
    'match_thresh': 0.8,
    'min_box_area': 10,
    'mot20': False,
    'conf': 0.3,
    'fuse_score': True  
}

# --- MongoDB Atlas Connection ---
# IMPORTANT: Replace this with the full connection string from your Atlas dashboard.
# Make sure to replace <username> and <password> with your actual credentials.
CONNECTION_STRING = "mongodb+srv://fish-tracker-admin:admin@cluster0.1oleyi6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
try:
    client = MongoClient(CONNECTION_STRING, tls=True, tlsAllowInvalidCertificates=True)
    db = client.fish_tracking_db
    fish_data_collection = db.fish_data
    print("Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")
    client = None
    fish_data_collection = None

def get_last_known_distance(fish_id):
    """
    Fetches the last known total distance for a fish from the database.
    """
    if fish_data_collection is None:
        return 0.0
    
    try:
        document = fish_data_collection.find_one({"fish_id": fish_id})
        if document and "distance_traveled_pixels" in document:
            return document["distance_traveled_pixels"]
        return 0.0
    except Exception as e:
        print(f"Error fetching distance for fish {fish_id}: {e}")
        return 0.0

def save_data_to_db(fish_id, size, distance_traveled, is_active, location, status):
    """
    Saves or updates a fish's data in the MongoDB Atlas collection.
    
    Args:
        fish_id (int): The ID of the fish.
        size (float): The size of the fish.
        distance_traveled (float): Total distance traveled by the fish.
        is_active (bool): Whether the fish is currently in frame.
        location (dict): The current location of the fish.
        status (str): The behavioral status of the fish ('active', 'motionless', 'dead', or 'inactive').
    """
    if fish_data_collection is None:
        print("Database connection not available. Skipping save.")
        return

    try:
        fish_document = {
            "fish_id": fish_id,
            "size": size,
            "distance_traveled_pixels": distance_traveled,
            "last_updated": datetime.now(),
            "is_active": is_active,
            "current_location": location,
            "status": status
        }
        
        fish_data_collection.update_one(
            {"fish_id": fish_id},
            {"$set": fish_document},
            upsert=True
        )
    except Exception as e:
        print(f"An error occurred while saving data: {e}")

def load_tracker_config(config_dict):
    """
    Saves a dictionary to a temporary YAML file and returns the file path.
    """
    temp_path = Path("temp_tracker_config.yaml")
    with open(temp_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    return str(temp_path)

# --- Main Script ---
# IMPORTANT: Update this path to the location of your video file.
video_path = "C:\\Users\\occho\\fishbt\\ca loc.mp4"

# --- Data for Plotting and Logging ---
frame_numbers = []
live_counts = []
total_counts = []
last_known_positions = {}
distance_traveled = {}
last_motion_frame = {}
dead_conditions_met_since = {}

# --- New Constants for Dead Fish Detection ---
FPS = 30  # Assumes a video FPS of 30. Adjust if different.
FLAT_BODY_THRESHOLD = 15  # Pixels. Max difference between head and tail Y-coords.
MINIMAL_MOTION_THRESHOLD = 5  # Pixels. Max movement per frame to be considered "drifting".
TOP_ZONE_PERCENT = 0.15 # Top 15% of the frame is considered the "surface zone".
DEAD_TIMEOUT_MINUTES = 25 # The duration to wait before marking as dead.
DEAD_TIMEOUT_FRAMES = DEAD_TIMEOUT_MINUTES * 60 * FPS

# Matplotlib setup for the live count plot
plt.style.use('ggplot')
plt.ion()
fig_line, ax_line = plt.subplots()
line1, = ax_line.plot(frame_numbers, live_counts, label='Live Fish Count')
line2, = ax_line.plot(frame_numbers, total_counts, label='Total Unique Fish Count')
ax_line.set_title('Live Fish Tracking Data')
ax_line.set_xlabel('Frame Number')
ax_line.set_ylabel('Count')
ax_line.legend()

# Matplotlib setup for the live distance bar chart
fig_bar, ax_bar = plt.subplots()
ax_bar.set_title('Total Distance Traveled by Fish')
ax_bar.set_xlabel('Fish ID')
ax_bar.set_ylabel('Distance Traveled (pixels)')

# Load your custom-trained YOLOv8 model
model = YOLO('best (3).pt')

# Create a video capture object
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOP_ZONE_Y = int(FRAME_HEIGHT * TOP_ZONE_PERCENT)

last_seen_frame = {}
FRAME_TIMEOUT = 70

print("Processing video. Press 'q' to close the window.")
frame_count = 0
total_unique_fish = 0
seen_track_ids = set()

# Load the tracker config
tracker_config_path = load_tracker_config(tracker_config)

# Use `stream=True` to get a generator for frame-by-frame results.
results_generator = model.track(source=video_path, show=False, persist=True, tracker=tracker_config_path, stream=True)

# Loop through each frame in the video
for results in results_generator:
    frame_count += 1
    
    frame = results.orig_img
    annotated_frame = frame.copy()
    
    live_fish_count = 0
    current_frame_track_ids = set()

    if results.boxes.id is not None:
        live_fish_count = len(results.boxes.id)
        current_frame_track_ids = set(results.boxes.id.int().cpu().tolist())

        for track_id in current_frame_track_ids:
            last_seen_frame[track_id] = frame_count
            
            if track_id not in seen_track_ids:
                seen_track_ids.add(track_id)
                total_unique_fish = len(seen_track_ids)
                last_motion_frame[track_id] = frame_count

        for i in range(len(results.boxes.id)):
            track_id = int(results.boxes.id[i].cpu().item())
            box = results.boxes.xyxy[i].cpu().numpy().astype(int)
            
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            current_position = (center_x, center_y)

            current_total_distance = get_last_known_distance(track_id)
            dist = 0
            if track_id in last_known_positions:
                last_position = last_known_positions[track_id]
                dist = np.sqrt((current_position[0] - last_position[0])**2 + (current_position[1] - last_position[1])**2)
                current_total_distance += dist
            
            last_known_positions[track_id] = current_position
            distance_traveled[track_id] = current_total_distance
            
            # --- Determine fish status based on our new logic ---
            status = 'active'
            
            # Check for keypoints to perform dead fish analysis
            if results.keypoints is not None and results.keypoints.xy is not None and len(results.keypoints.xy[i]) >= 2:
                # We assume the first keypoint is the head and the last is the tail.
                head_kp = results.keypoints.xy[i].cpu().numpy()[0]
                tail_kp = results.keypoints.xy[i].cpu().numpy()[-1]

                # Condition 1: Flat and level body
                is_flat = abs(head_kp[1] - tail_kp[1]) < FLAT_BODY_THRESHOLD

                # Condition 2: Minimal motion
                is_drifting = dist < MINIMAL_MOTION_THRESHOLD

                # Condition 3: High vertical position (top of the frame)
                is_at_top = center_y < TOP_ZONE_Y

                # Final check: all three conditions must be met to start the timer
                if is_flat and is_drifting and is_at_top:
                    if track_id not in dead_conditions_met_since:
                        dead_conditions_met_since[track_id] = frame_count
                    
                    if frame_count - dead_conditions_met_since[track_id] >= DEAD_TIMEOUT_FRAMES:
                        status = 'dead'
                else:
                    # If any condition is not met, reset the timer for this fish
                    if track_id in dead_conditions_met_since:
                        del dead_conditions_met_since[track_id]
            else:
                # If keypoints are not available, fall back to motionless logic
                if dist < MINIMAL_MOTION_THRESHOLD:
                    if frame_count - last_motion_frame.get(track_id, frame_count) > FRAME_TIMEOUT:
                        status = 'motionless'
                else:
                    last_motion_frame[track_id] = frame_count

            # --- Call the database saving function with the new status ---
            fish_size_placeholder = 0.0
            save_data_to_db(
                fish_id=track_id,
                size=fish_size_placeholder,
                distance_traveled=current_total_distance,
                is_active=True,
                location={"x": center_x, "y": center_y},
                status=status
            )
            
            x1, y1, x2, y2 = box
            box_color = (0, 255, 0)
            text_color = (0, 255, 0)
            if status == 'dead':
                box_color = (0, 0, 255)
                text_color = (0, 0, 255)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            label = f'ID: {track_id} | Status: {status} | Dist: {current_total_distance:.2f}'
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            if results.keypoints is not None and results.keypoints.xy is not None and len(results.keypoints.xy) > i:
                keypoints = results.keypoints.xy[i].cpu().numpy().astype(int)
                for kp in keypoints:
                    cv2.circle(annotated_frame, (kp[0], kp[1]), 5, (0, 0, 255), -1)

    ids_to_remove = [track_id for track_id, last_frame in last_seen_frame.items() if frame_count - last_frame > FRAME_TIMEOUT]
    for track_id in ids_to_remove:
        if track_id in last_seen_frame:
            del last_seen_frame[track_id]
        if track_id in last_known_positions:
            del last_known_positions[track_id]
        if track_id in distance_traveled:
            del distance_traveled[track_id]
        
        # --- Mark fish as inactive when it leaves the frame ---
        save_data_to_db(
            fish_id=track_id,
            size=0.0,
            distance_traveled=get_last_known_distance(track_id),
            is_active=False,
            location=None,
            status='inactive'
        )

    cv2.putText(annotated_frame, f'Live Count: {live_fish_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    cv2.putText(annotated_frame, f'Total Count: {total_unique_fish}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

    # --- Update the line graph ---
    frame_numbers.append(frame_count)
    live_counts.append(live_fish_count)
    total_counts.append(total_unique_fish)

    line1.set_xdata(frame_numbers)
    line1.set_ydata(live_counts)
    line2.set_xdata(frame_numbers)
    line2.set_ydata(total_counts)

    ax_line.relim()
    ax_line.autoscale_view()
    
    # --- Update the bar chart ---
    ax_bar.clear()
    ax_bar.set_title('Total Distance Traveled by Fish')
    ax_bar.set_xlabel('Fish ID')
    ax_bar.set_ylabel('Distance Traveled (pixels)')

    sorted_fish = sorted(distance_traveled.items(), key=lambda item: item[1], reverse=True)
    sorted_ids = [item[0] for item in sorted_fish]
    sorted_distances = [item[1] for item in sorted_fish]
    
    ax_bar.bar([str(id) for id in sorted_ids], sorted_distances, color='skyblue')
    
    for i, dist in enumerate(sorted_distances):
        ax_bar.text(i, dist + 50, f'{dist:.2f}', ha='center')

    plt.pause(0.01)

    cv2.imshow("Fish Tracking", annotated_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Finalizing the script
cv2.destroyAllWindows()
cap.release()
plt.ioff()
plt.close(fig_line)
plt.close(fig_bar)

# Clean up the temporary YAML file
temp_file = Path("temp_tracker_config.yaml")
if temp_file.exists():
    temp_file.unlink()
