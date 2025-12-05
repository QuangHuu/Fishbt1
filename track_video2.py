import cv2  # OpenCV for video processing and drawing
from ultralytics import YOLO  # YOLOv8 for object detection and tracking
import numpy as np  # For numerical operations
import time  # For timing operations
import yaml  # For saving/loading YAML config files
from pathlib import Path  # For file path operations
import csv  # For CSV file operations (not used in main loop)
import matplotlib.pyplot as plt  # For plotting live data
from pymongo import MongoClient  # For MongoDB database operations
from datetime import datetime  # For timestamps
import math  # For mathematical calculations
import os  # For OS-level operations

# ---------------------------------------------------------------------
# --- CONFIGURATION & CONSTANTS ---
# ---------------------------------------------------------------------

tracker_config = {  # Configuration dictionary for ByteTrack tracker
    'tracker_type': 'bytetrack',  # Use ByteTrack for tracking
    'reid_model': 'osnet_x0_25_msmt17.pt',  # ReID model for ByteTrack
    'track_high_thresh': 0.5,  # High threshold for tracking
    'track_low_thresh': 0.1,  # Low threshold for tracking
    'new_track_thresh': 0.4,  # Threshold for new tracks
    'track_buffer': 50,  # Buffer size for tracking
    'match_thresh': 0.8,  # Matching threshold
    'min_box_area': 10,  # Minimum bounding box area
    'mot20': False,  # Use MOT20 dataset settings (False here)
    'conf': 0.3,  # Confidence threshold for detections
    'fuse_score': True  # Fuse scores for tracking
}

CONNECTION_STRING = "mongodb+srv://admin:helloadmin123@fish.caqok5c.mongodb.net/?appName=Fish"  # MongoDB Atlas connection string
try:
    client = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000, tls=True, tlsAllowInvalidCertificates=True)  # Connect to MongoDB Atlas
    client.admin.command('ping')  # Test the connection
    db = client.fish_tracking_db  # Select the database
    fish_data_collection = db.fish_data  # Select the collection
    print("Successfully connected to MongoDB Atlas!")  # Print success message
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}")  # Print error message
    client = None  # Set client to None if connection fails
    fish_data_collection = None  # Set collection to None if connection fails

FPS = 30  # Frames per second of the video
FRAME_TIMEOUT = 70  # Number of frames before a fish is considered inactive
MINIMAL_MOTION_THRESHOLD = 0.2  # Minimum movement (meters) to be considered active
LETHARGY_TIMEOUT_MINUTES = 5  # Minutes before a fish is considered lethargic
LETHARGY_TIMEOUT_FRAMES = LETHARGY_TIMEOUT_MINUTES * 60 * FPS  # Convert minutes to frames
DEAD_TIMEOUT_MINUTES = 360  # Minutes before a fish is considered dead
DEAD_TIMEOUT_FRAMES = DEAD_TIMEOUT_MINUTES * 60 * FPS  # Convert minutes to frames

video_path = "C:\\Users\\occho\\fishbt\\ca loc.mp4"  # Path to the input video file
model_path = 'best (3).pt'  # Path to the yolov8 model weights
temp_config_file = "temp_tracker_config.yaml"  # Temporary YAML file for tracker config

# ---------------------------------------------------------------------
# --- FUNCTION DEFINITIONS ---
# ---------------------------------------------------------------------

def get_3d_keypoint_coordinates(head_2d, tail_2d, frame_width, frame_height):
    """
    Converts 2D keypoints (pixels) to mock 3D coordinates (meters).
    Replace with real depth camera logic for actual 3D measurement.
    """
    MOCK_TANK_SIZE = 1.0  # Assume tank is 1 meter wide/high for scaling
    head_norm_x = head_2d[0] / frame_width  # Normalize head x coordinate
    head_norm_y = head_2d[1] / frame_height  # Normalize head y coordinate
    head_x_m = (head_norm_x * MOCK_TANK_SIZE) - (MOCK_TANK_SIZE / 2)  # Convert to meters (centered)
    head_y_m = head_norm_y * MOCK_TANK_SIZE  # Convert to meters
    head_z_m = 1.5 + (head_norm_y * 0.5)  # Mock z coordinate
    head_3d = (head_x_m, head_y_m, head_z_m)  # Head 3D position
    tail_x_m = ((tail_2d[0] / frame_width) * MOCK_TANK_SIZE) - (MOCK_TANK_SIZE / 2)  # Normalize and convert tail x
    tail_y_m = (tail_2d[1] / frame_height) * MOCK_TANK_SIZE  # Normalize and convert tail y
    tail_z_m = 1.5 + ((tail_2d[1] / frame_height) * 0.5) + 0.05  # Mock tail z coordinate
    tail_3d = (tail_x_m, tail_y_m, tail_z_m)  # Tail 3D position
    return head_3d, tail_3d  # Return both positions

def calculate_3d_centroid(head_3d, tail_3d):
    """
    Calculates the midpoint between head and tail in 3D space.
    """
    cx = (head_3d[0] + tail_3d[0]) / 2  # Average x
    cy = (head_3d[1] + tail_3d[1]) / 2  # Average y
    cz = (head_3d[2] + tail_3d[2]) / 2  # Average z
    return (cx, cy, cz)  # Return centroid

def get_last_known_distance(fish_id):
    """
    Fetches the last known total distance traveled for a fish from MongoDB.
    """
    if fish_data_collection is None:  # If DB is not connected
        return 0.0  # Return zero
    try:
        document = fish_data_collection.find_one({"fish_id": fish_id})  # Query DB for fish_id
        if document and "distance_traveled_meters" in document:  # If found and has distance
            return document["distance_traveled_meters"]  # Return stored distance
        return 0.0  # If not found, return zero
    except Exception as e:
        print(f"Error fetching distance for fish {fish_id}: {e}")  # Print error
        return 0.0  # Return zero on error

def save_data_to_db(fish_id, size, distance_traveled_m, is_active, location_m, status):
    """
    Saves or updates a fish's record in MongoDB.
    """
    if fish_data_collection is None:  # If DB is not connected
        print("Database connection not available. Skipping save.")  # Print warning
        return  # Exit function
    try:
        fish_document = {  # Create document to save
            "fish_id": fish_id,
            "size": size,
            "distance_traveled_meters": distance_traveled_m,
            "last_updated": datetime.now(),
            "is_active": is_active,
            "current_location_meters": location_m,
            "status": status
        }
        fish_data_collection.update_one(  # Update or insert document
            {"fish_id": fish_id},
            {"$set": fish_document},
            upsert=True
        )
    except Exception as e:
        print(f"An error occurred while saving data: {e}")  # Print error

def load_tracker_config(config_dict):
    """
    Saves the tracker config dictionary to a temporary YAML file.
    """
    temp_path = Path(temp_config_file)  # Path for temp YAML
    with open(temp_path, 'w') as f:  # Open file for writing
        yaml.dump(config_dict, f, default_flow_style=False)  # Dump config to YAML
    return str(temp_path)  # Return path as string

# ---------------------------------------------------------------------
# --- DATA STRUCTURES FOR TRACKING & PLOTTING ---
# ---------------------------------------------------------------------

frame_numbers = []  # List to store frame numbers for plotting
live_counts = []  # List to store live fish counts for plotting
total_counts = []  # List to store total unique fish counts for plotting
last_known_positions_3d = {}  # Dict to store last known 3D positions for each fish
distance_traveled = {}  # Dict to store total distance traveled for each fish
inactivity_start_frame = {}  # Dict to store frame when inactivity started for each fish

# ---------------------------------------------------------------------
# --- PLOTTING SETUP ---
# ---------------------------------------------------------------------

plt.style.use('ggplot')  # Use ggplot style for plots
plt.ion()  # Enable interactive mode for live updating

fig_line, ax_line = plt.subplots()  # Create figure and axis for line plot
line1, = ax_line.plot(frame_numbers, live_counts, label='Live Fish Count')  # Line for live count
line2, = ax_line.plot(frame_numbers, total_counts, label='Total Unique Fish Count')  # Line for total count
ax_line.set_title('Live Fish Tracking Data')  # Set plot title
ax_line.set_xlabel('Frame Number')  # Set x label
ax_line.set_ylabel('Count')  # Set y label
ax_line.legend()  # Show legend

fig_bar, ax_bar = plt.subplots()  # Create figure and axis for bar chart
ax_bar.set_title('Total Distance Traveled by Fish (Meters)')  # Set bar chart title
ax_bar.set_xlabel('Fish ID')  # Set x label
ax_bar.set_ylabel('Distance Traveled (Meters)')  # Set y label

# ---------------------------------------------------------------------
# --- MODEL & VIDEO SETUP ---
# ---------------------------------------------------------------------

model = YOLO(r"C:\Users\occho\fishbt\best (3).pt")  # Load YOLOv8 model
cap = cv2.VideoCapture(video_path)  # Open video file
if not cap.isOpened():  # If video cannot be opened
    print(f"Error: Could not open video file at {video_path}")  # Print error
    exit()  # Exit script

FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get frame width
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get frame height

last_seen_frame = {}  # Dict to store last seen frame for each fish

print("Processing video. Press 'q' to close the window.")  # Print start message
frame_count = 0  # Initialize frame counter
total_unique_fish = 0  # Initialize total unique fish counter
seen_track_ids = set()  # Set to store all seen fish IDs

tracker_config_path = load_tracker_config(tracker_config)  # Save tracker config to YAML
results_generator = model.track(  # Start YOLOv8 tracking
    source=video_path,
    show=False,
    persist=True,
    tracker=tracker_config_path,
    stream=True
)

# ---------------------------------------------------------------------
# --- MAIN PROCESSING LOOP ---
# ---------------------------------------------------------------------

try:
    for results in results_generator:  # Loop through each frame's results
        frame_count += 1  # Increment frame counter
        frame = results.orig_img  # Get original frame
        annotated_frame = frame.copy()  # Copy frame for annotation
        live_fish_count = 0  # Initialize live fish count
        current_frame_track_ids = set()  # Set for current frame's fish IDs

        # --- Fish Detection & Tracking ---
        if results.boxes.id is not None:  # If there are tracked fish in this frame
            live_fish_count = len(results.boxes.id)  # Count live fish
            current_frame_track_ids = set(results.boxes.id.int().cpu().tolist())  # Get unique fish IDs

            # Update tracking info for each fish
            for track_id in current_frame_track_ids:  # Loop through each fish ID
                last_seen_frame[track_id] = frame_count  # Update last seen frame
                if track_id not in seen_track_ids:  # If fish is new
                    seen_track_ids.add(track_id)  # Add to seen IDs
                    total_unique_fish = len(seen_track_ids)  # Update total unique fish count
                    inactivity_start_frame[track_id] = frame_count  # Start inactivity timer

            # --- Per-Fish Processing ---
            for i in range(len(results.boxes.id)):  # Loop through each detected fish
                track_id = int(results.boxes.id[i].cpu().item())  # Get fish ID
                box_pixels = results.boxes.xyxy[i].cpu().numpy().astype(int)  # Get bounding box
                x1, y1, x2, y2 = box_pixels  # Unpack box coordinates

                status = 'active'  # Default status
                current_position_3d = None  # Initialize 3D position
                current_total_distance_m = get_last_known_distance(track_id)  # Get last known distance
                dist_m = 0  # Initialize movement distance

                # If keypoints are available, calculate 3D position and movement
                if results.keypoints is not None and results.keypoints.xy is not None and len(results.keypoints.xy) > i:
                    keypoints_2d_px = results.keypoints.xy[i].cpu().numpy().astype(int)  # Get keypoints
                    if len(keypoints_2d_px) >= 2:  # If at least head and tail
                        head_kp = keypoints_2d_px[0]  # Head keypoint
                        tail_kp = keypoints_2d_px[-1]  # Tail keypoint
                        head_3d, tail_3d = get_3d_keypoint_coordinates(  # Convert to 3D
                            head_kp, tail_kp, FRAME_WIDTH, FRAME_HEIGHT
                        )
                        current_position_3d = calculate_3d_centroid(head_3d, tail_3d)  # Get centroid
                        if track_id in last_known_positions_3d:  # If fish has previous position
                            last_position_3d = last_known_positions_3d[track_id]  # Get last position
                            dist_m = math.sqrt(  # Calculate 3D movement
                                (current_position_3d[0] - last_position_3d[0])**2 +
                                (current_position_3d[1] - last_position_3d[1])**2 +
                                (current_position_3d[2] - last_position_3d[2])**2
                            )
                            if dist_m >= MINIMAL_MOTION_THRESHOLD:  # If moved enough
                                current_total_distance_m += dist_m  # Add to total distance

                        last_known_positions_3d[track_id] = current_position_3d  # Update last position
                        distance_traveled[track_id] = current_total_distance_m  # Update total distance

                        # --- Status Detection Logic ---
                        is_motionless = dist_m < MINIMAL_MOTION_THRESHOLD  # Check if fish is motionless

                        if is_motionless:  # If fish hasn't moved
                            if track_id not in inactivity_start_frame:  # If no inactivity timer
                                inactivity_start_frame[track_id] = frame_count  # Start timer
                            inactivity_duration = frame_count - inactivity_start_frame[track_id]  # Calculate inactivity duration
                            if inactivity_duration >= DEAD_TIMEOUT_FRAMES:  # If inactive too long
                                status = 'dead'  # Mark as dead
                            elif inactivity_duration >= LETHARGY_TIMEOUT_FRAMES:  # If inactive moderately long
                                status = 'lethargic'  # Mark as lethargic
                            else:
                                status = 'active'  # Otherwise, still active
                        else:  # If fish moved
                            inactivity_start_frame[track_id] = frame_count  # Reset inactivity timer
                            status = 'active'  # Mark as active

                # Prepare location data for database
                location_data_m = None  # Default location
                if current_position_3d:  # If position available
                    location_data_m = {
                        "x": current_position_3d[0],
                        "y": current_position_3d[1],
                        "z": current_position_3d[2]
                    }

                fish_size_placeholder = 0.0  # Placeholder for fish size
                save_data_to_db(  # Save/update fish data in DB
                    fish_id=track_id,
                    size=fish_size_placeholder,
                    distance_traveled_m=current_total_distance_m,
                    is_active=True,
                    location_m=location_data_m,
                    status=status
                )

                # --- Frame Annotation ---
                box_color = (0, 255, 0)  # Default box color (green)
                text_color = (0, 255, 0)  # Default text color (green)
                if status == 'dead':  # If dead
                    box_color = (0, 0, 255)  # Red box
                    text_color = (0, 0, 255)  # Red text
                elif status == 'lethargic':  # If lethargic
                    box_color = (0, 255, 255)  # Yellow box
                    text_color = (0, 255, 255)  # Yellow text

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)  # Draw bounding box
                label = f'ID: {track_id} | Status: {status} | Dist: {current_total_distance_m:.2f} m'  # Label text
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)  # Draw label

                # Draw keypoints if available
                if results.keypoints is not None and results.keypoints.xy is not None and len(results.keypoints.xy) > i:
                    keypoints = results.keypoints.xy[i].cpu().numpy().astype(int)  # Get keypoints
                    for kp in keypoints:  # Loop through each keypoint
                        cv2.circle(annotated_frame, (kp[0], kp[1]), 5, (0, 0, 255), -1)  # Draw keypoint

        # --- Handle Inactive Fish ---
        ids_to_remove = [  # List of fish IDs to remove
            track_id for track_id, last_frame in last_seen_frame.items()
            if frame_count - last_frame > FRAME_TIMEOUT  # If not seen for too long
        ]
        for track_id in ids_to_remove:  # Loop through each inactive fish
            save_data_to_db(  # Mark fish as inactive in DB
                fish_id=track_id,
                size=0.0,
                distance_traveled_m=get_last_known_distance(track_id),
                is_active=False,
                location_m=None,
                status='inactive'
            )
            last_seen_frame.pop(track_id, None)  # Remove from last seen dict
            last_known_positions_3d.pop(track_id, None)  # Remove from positions dict
            distance_traveled.pop(track_id, None)  # Remove from distance dict
            inactivity_start_frame.pop(track_id, None)  # Remove from inactivity dict

        # --- Update Plots & Display ---
        cv2.putText(annotated_frame, f'Live Count: {live_fish_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)  # Show live count
        cv2.putText(annotated_frame, f'Total Count: {total_unique_fish}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)  # Show total count

        frame_numbers.append(frame_count)  # Add frame number to list
        live_counts.append(live_fish_count)  # Add live count to list
        total_counts.append(total_unique_fish)  # Add total count to list

        line1.set_xdata(frame_numbers)  # Update line plot x data
        line1.set_ydata(live_counts)  # Update line plot y data
        line2.set_xdata(frame_numbers)  # Update line plot x data
        line2.set_ydata(total_counts)  # Update line plot y data

        ax_line.relim()  # Recalculate limits
        ax_line.autoscale_view()  # Autoscale view

        ax_bar.clear()  # Clear bar chart
        ax_bar.set_title('Total Distance Traveled by Fish (Meters)')  # Set title
        ax_bar.set_xlabel('Fish ID')  # Set x label
        ax_bar.set_ylabel('Distance Traveled (Meters)')  # Set y label

        sorted_fish = sorted(distance_traveled.items(), key=lambda item: item[1], reverse=True)  # Sort fish by distance
        sorted_ids = [item[0] for item in sorted_fish]  # Get sorted IDs
        sorted_distances = [item[1] for item in sorted_fish]  # Get sorted distances

        max_dist = max(sorted_distances) if sorted_distances else 1.0  # Get max distance
        text_offset = max_dist * 0.01  # Offset for text

        ax_bar.bar([str(id) for id in sorted_ids], sorted_distances, color='skyblue')  # Draw bar chart

        for i, dist in enumerate(sorted_distances):  # Loop through each fish
            ax_bar.text(i, dist + text_offset, f'{dist:.2f} m', ha='center', fontsize=8)  # Draw distance text

        plt.pause(0.01)  # Pause to update plot

        cv2.imshow("Fish Tracking", annotated_frame)  # Show annotated frame

        if cv2.waitKey(1) == ord('q'):  # If 'q' key pressed
            break  # Exit loop

finally:
    cv2.destroyAllWindows()  # Close OpenCV windows
    cap.release()  # Release video capture
    plt.ioff()  # Turn off interactive mode
    plt.close(fig_line)  # Close line plot
    plt.close(fig_bar)  # Close bar chart
    temp_file = Path(temp_config_file)  # Path to temp YAML
    if temp_file.exists():  # If temp file exists
        temp_file.unlink()  # Delete temp file
    print("Video processing and cleanup complete.")  # Print finish message

