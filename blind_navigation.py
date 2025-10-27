import cv2
import numpy as np
import time
from ultralytics import YOLO
import math
import platform

# Choose the appropriate TTS system based on platform
if platform.system() == 'Darwin':  # macOS
    # Use alternative TTS for macOS to avoid objc error
    import subprocess
    
    def speak_macos(text):
        subprocess.run(['say', text])
else:
    # Use pyttsx3 for other platforms
    import pyttsx3

class BlindNavigationSystem:
    def __init__(self):
        # Initialize text-to-speech engine based on platform
        self.is_macos = platform.system() == 'Darwin'
        
        if not self.is_macos:
            # Use pyttsx3 for non-macOS platforms
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Load YOLO model with optimizations
        self.model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model
        # Set inference parameters for better performance
        self.conf_threshold = 0.35  # Higher confidence threshold for more stable detections
        self.iou_threshold = 0.5  # Higher IoU threshold for better NMS
        self.max_det = 10  # Limit detections for better performance
        self.frame_skip = 1  # Process every frame (0) or skip frames (1+) for better performance
        
        # Frame smoothing parameters for more consistent detection
        self.prev_boxes = []        # Store previous detection boxes for smoothing
        self.smoothing_factor = 0.7 # Weight for current detection (1-factor for previous)
        self.max_tracking_age = 5   # Maximum number of frames to track an object without detection
        
        # Initialize camera with error handling
        self.use_camera = True
        self.use_sample_image = False
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Warning: Could not open video device. Using fallback mode.")
                self.use_camera = False
                self.use_sample_image = True
            else:
                # Set camera resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            print(f"Camera error: {e}")
            print("Switching to fallback mode.")
            self.use_camera = False
            self.use_sample_image = True
            
        # Prepare sample image for fallback mode
        if self.use_sample_image:
            # Create a sample image with some shapes
            self.sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add a red rectangle (simulating an obstacle)
            cv2.rectangle(self.sample_image, (100, 100), (200, 300), (0, 0, 255), -1)
            # Add a blue circle (simulating another obstacle)
            cv2.circle(self.sample_image, (400, 200), 50, (255, 0, 0), -1)
            # Add a green triangle (simulating another obstacle)
            pts = np.array([[300, 300], [400, 300], [350, 200]], np.int32)
            cv2.fillPoly(self.sample_image, [pts], (0, 255, 0))
            
            print("Using sample image for testing. Camera access is required for real usage.")
            print("On macOS, you need to grant camera permission in System Preferences > Security & Privacy > Camera")
        
        # Define classes of interest (obstacles)
        self.obstacle_classes = [0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 56, 57, 58, 60, 62, 63, 66, 67, 73, 74]
        # These correspond to person, bicycle, car, motorcycle, bus, truck, traffic light, fire hydrant, 
        # stop sign, bench, chair, couch, potted plant, dining table, toilet, tv, keyboard, cell phone, book, scissors
        
        # Navigation parameters
        self.last_speech_time = time.time()
        self.speech_cooldown = 2.0  # Seconds between speech commands
        self.distance_threshold = 0.4  # Relative distance threshold for close obstacles
        self.frame_center_x = 320  # Center X coordinate of frame
        self.safe_distance = 0.5  # Safe distance threshold
        
        # Speak initial message
        self.speak("Blind navigation system initialized. Starting obstacle detection.")

    def speak(self, text):
        """Convert text to speech if cooldown period has passed"""
        current_time = time.time()
        if current_time - self.last_speech_time >= self.speech_cooldown:
            print(f"Speaking: {text}")
            if self.is_macos:
                # Use macOS built-in 'say' command
                subprocess.run(['say', text])
            else:
                # Use pyttsx3 for other platforms
                self.engine.say(text)
                self.engine.runAndWait()
            self.last_speech_time = current_time

    def estimate_distance(self, box_height, real_height=1.7):
        """
        Estimate distance based on the height of the bounding box
        This is a simple approximation - in a real system, you would use depth sensors
        """
        # Focal length approximation
        focal_length = 500
        distance = (real_height * focal_length) / box_height
        return distance

    def get_navigation_command(self, x_center, width, distance):
        """Generate detailed navigation command based on obstacle position"""
        # Divide the frame into 5 horizontal sections for more precise guidance
        frame_fifth = self.frame_center_x / 2.5
        
        # Calculate how far the obstacle is from the center as a percentage
        center_offset = abs(x_center - self.frame_center_x) / self.frame_center_x
        
        # Convert distance to approximate steps (assuming average step is ~0.7m)
        steps = max(1, int(distance / 0.7))
        
        if distance < self.safe_distance:
            # Close obstacle - immediate action needed
            if x_center < self.frame_center_x - 2*frame_fifth:
                return f"Obstacle very close on your far left. Take {steps} steps to your right immediately."
            elif x_center < self.frame_center_x - frame_fifth:
                return f"Obstacle close on your left. Take {steps} steps to your right."
            elif x_center > self.frame_center_x + 2*frame_fifth:
                return f"Obstacle very close on your far right. Take {steps} steps to your left immediately."
            elif x_center > self.frame_center_x + frame_fifth:
                return f"Obstacle close on your right. Take {steps} steps to your left."
            else:
                # Directly ahead - suggest best direction to turn based on frame analysis
                if self.find_clearest_path() == "left":
                    return "Obstacle directly ahead. Stop and turn left, then proceed."
                else:
                    return "Obstacle directly ahead. Stop and turn right, then proceed."
        else:
            # Obstacle at safer distance - provide guidance
            if x_center < self.frame_center_x - 2*frame_fifth:
                return f"Obstacle on your far left, approximately {distance:.1f} meters away. Continue straight."
            elif x_center < self.frame_center_x - frame_fifth:
                return f"Obstacle on your left, approximately {distance:.1f} meters away. Slight right recommended."
            elif x_center > self.frame_center_x + 2*frame_fifth:
                return f"Obstacle on your far right, approximately {distance:.1f} meters away. Continue straight."
            elif x_center > self.frame_center_x + frame_fifth:
                return f"Obstacle on your right, approximately {distance:.1f} meters away. Slight left recommended."
            else:
                return f"Obstacle directly ahead, approximately {distance:.1f} meters away. Prepare to change direction."
                
    def find_clearest_path(self):
        """Analyze the current frame to find the clearest path (left or right)"""
        # Get the latest detection results
        if not hasattr(self, 'latest_results') or self.latest_results is None:
            return "right"  # Default if no results available
            
        # Count obstacles on left and right sides of the frame
        left_count = 0
        right_count = 0
        left_proximity = 0
        right_proximity = 0
        
        for r in self.latest_results:
            boxes = r.boxes
            for box in boxes:
                # Get class and confidence
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Check if this is an obstacle class and confidence is high enough
                if cls_id in self.obstacle_classes and conf > 0.5:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x_center = (x1 + x2) / 2
                    
                    # Calculate distance (simplified)
                    box_height = y2 - y1
                    distance = self.estimate_distance(box_height)
                    
                    # Add to left or right counts based on position
                    if x_center < self.frame_center_x:
                        left_count += 1
                        left_proximity += (1.0 / max(0.1, distance))  # Higher value for closer objects
                    else:
                        right_count += 1
                        right_proximity += (1.0 / max(0.1, distance))  # Higher value for closer objects
        
        # Determine clearest path based on obstacle count and proximity
        if left_count == right_count:
            # If equal counts, use proximity
            if left_proximity <= right_proximity:
                return "left"
            else:
                return "right"
        elif left_count < right_count:
            return "left"  # Fewer obstacles on left
        else:
            return "right"  # Fewer obstacles on right

    def smooth_boxes(self, current_boxes):
        """Apply temporal smoothing to bounding boxes for more stable detection"""
        if not self.prev_boxes:
            # First frame, no previous boxes to smooth with
            self.prev_boxes = current_boxes
            return current_boxes
        
        smoothed_boxes = []
        matched_prev_indices = set()
        
        # For each current box, find the best matching previous box
        for curr_box in current_boxes:
            curr_x1, curr_y1, curr_x2, curr_y2, curr_cls, curr_conf, curr_dist = curr_box
            
            best_match_idx = -1
            best_match_iou = 0.3  # Minimum IoU threshold for matching
            
            # Find the best matching previous box
            for i, prev_box in enumerate(self.prev_boxes):
                if i in matched_prev_indices:
                    continue  # Skip already matched boxes
                
                prev_x1, prev_y1, prev_x2, prev_y2, prev_cls, prev_conf, prev_dist = prev_box
                
                # Only match boxes of the same class
                if prev_cls != curr_cls:
                    continue
                
                # Calculate IoU between current and previous box
                x_left = max(curr_x1, prev_x1)
                y_top = max(curr_y1, prev_y1)
                x_right = min(curr_x2, prev_x2)
                y_bottom = min(curr_y2, prev_y2)
                
                if x_right < x_left or y_bottom < y_top:
                    continue  # No overlap
                
                intersection = (x_right - x_left) * (y_bottom - y_top)
                curr_area = (curr_x2 - curr_x1) * (curr_y2 - curr_y1)
                prev_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
                union = curr_area + prev_area - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = i
            
            if best_match_idx >= 0:
                # Found a match, apply smoothing
                prev_box = self.prev_boxes[best_match_idx]
                matched_prev_indices.add(best_match_idx)
                
                # Apply smoothing factor
                alpha = self.smoothing_factor
                smoothed_x1 = int(alpha * curr_x1 + (1 - alpha) * prev_box[0])
                smoothed_y1 = int(alpha * curr_y1 + (1 - alpha) * prev_box[1])
                smoothed_x2 = int(alpha * curr_x2 + (1 - alpha) * prev_box[2])
                smoothed_y2 = int(alpha * curr_y2 + (1 - alpha) * prev_box[3])
                
                # Keep the current class and confidence
                smoothed_box = [smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2, curr_cls, curr_conf, curr_dist]
                smoothed_boxes.append(smoothed_box)
            else:
                # No match found, use current box
                smoothed_boxes.append(curr_box)
        
        # Add unmatched previous boxes with reduced age
        for i, prev_box in enumerate(self.prev_boxes):
            if i not in matched_prev_indices and prev_box[6] > 0:  # Check if box has tracking age
                # Reduce tracking age
                prev_box[6] -= 1
                if prev_box[6] > 0:  # Only keep if still valid
                    smoothed_boxes.append(prev_box)
        
        # Update previous boxes for next frame
        self.prev_boxes = smoothed_boxes
        return smoothed_boxes

    def process_frame(self, frame):
        """Process a single frame for obstacle detection and navigation"""
        # Get original frame dimensions
        orig_height, orig_width = frame.shape[:2]
        
        # Resize frame for faster processing (smaller = faster)
        frame_resized = cv2.resize(frame, (416, 416))
        
        # Run YOLOv8 inference on the frame with optimized parameters
        results = self.model(
            frame_resized, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False
        )
        
        # Store the latest results for path finding
        self.latest_results = results
        
        # Process detection results
        closest_obstacle = None
        min_distance = float('inf')
        
        # Create a copy of the frame for drawing
        frame_with_boxes = frame.copy()
        
        # Scale factors to map from resized frame to original frame
        scale_x = orig_width / 416
        scale_y = orig_height / 416
        
        # Collect current boxes for smoothing
        current_boxes = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class and confidence
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Check if this is an obstacle class and confidence is high enough
                if cls_id in self.obstacle_classes and conf > 0.5:
                    # Get bounding box coordinates (in resized frame)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Scale coordinates to original frame size
                    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                    
                    # Calculate center and dimensions
                    x_center = (x1 + x2) / 2
                    box_height = y2 - y1
                    box_width = x2 - x1
                    
                    # Estimate distance (simplified)
                    distance = self.estimate_distance(box_height)
                    
                    # Add to current boxes with tracking age
                    current_boxes.append([x1, y1, x2, y2, cls_id, conf, self.max_tracking_age])
        
        # Apply temporal smoothing to boxes
        smoothed_boxes = self.smooth_boxes(current_boxes)
        
        # Draw smoothed boxes
        for box in smoothed_boxes:
            x1, y1, x2, y2, cls_id, conf, _ = box
            
            # Draw bounding box with thicker lines and brighter color
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Calculate distance based on box height
            box_height = y2 - y1
            distance = self.estimate_distance(box_height)
            
            # Add label with class name and distance
            class_name = self.model.names[cls_id]
            label = f"{class_name}: {distance:.1f}m"
            
            # Draw background for text for better visibility
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame_with_boxes, 
                         (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), 
                         (0, 0, 0), -1)
            
            # Draw text with larger font in white color
            cv2.putText(frame_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
            # Calculate box width
            box_width = x2 - x1
            
            # Check if this is the closest obstacle
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = {
                    'class': class_name,
                    'distance': distance,
                    'x_center': (x1 + x2) / 2,
                    'width': box_width
                }
        
        # Generate navigation command for the closest obstacle
        if closest_obstacle:
            command = self.get_navigation_command(
                closest_obstacle['x_center'],
                closest_obstacle['width'],
                closest_obstacle['distance']
            )
            
            # Add information about the obstacle
            full_command = f"{closest_obstacle['class']} detected. {command}"
            self.speak(full_command)
            
            # Display command on frame
            cv2.putText(frame_with_boxes, full_command, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame_with_boxes

    def run(self):
        """Main loop for the blind navigation system"""
        try:
            # Initialize frame counter for frame skipping
            frame_count = 0
            # For FPS calculation
            fps_start_time = time.time()
            fps_frame_count = 0
            fps = 0
            
            while True:
                # Start timing this frame
                frame_start_time = time.time()
                
                if self.use_camera:
                    # Capture frame from camera
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break
                else:
                    # Use sample image in fallback mode
                    frame = self.sample_image.copy()
                    # Add some movement to the sample image to simulate changes
                    current_time = time.time()
                    offset_x = int(30 * math.sin(current_time))
                    offset_y = int(20 * math.cos(current_time))
                    
                    # Create a translation matrix
                    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                    # Apply the translation
                    frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
                
                # Process frame based on frame skip setting
                if frame_count % (self.frame_skip + 1) == 0:
                    # Process the frame
                    processed_frame = self.process_frame(frame)
                else:
                    # Skip processing, just use the previous frame
                    processed_frame = frame
                
                # Increment frame counter
                frame_count += 1
                
                # Calculate and display FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Add performance info to the frame
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if not self.use_camera:
                    # Add demo mode indicator
                    cv2.putText(processed_frame, f"DEMO MODE - {time.strftime('%H:%M:%S')}", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display the processed frame
                cv2.imshow('Blind Navigation System', processed_frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Calculate how long this frame took to process
                frame_time = time.time() - frame_start_time
                
                # Dynamically adjust frame skip based on performance
                if frame_time > 0.1:  # If processing takes more than 100ms
                    self.frame_skip = min(3, self.frame_skip + 1)  # Increase skip (max 3)
                elif frame_time < 0.03 and self.frame_skip > 0:  # If processing is fast
                    self.frame_skip = max(0, self.frame_skip - 1)  # Decrease skip (min 0)
                
                # Add a small delay in fallback mode to control the frame rate
                if not self.use_camera:
                    time.sleep(0.1)
                
        finally:
            # Clean up
            if self.use_camera:
                self.cap.release()
            cv2.destroyAllWindows()
            self.speak("Navigation system shutting down.")

if __name__ == "__main__":
    # Create and run the blind navigation system
    try:
        navigation_system = BlindNavigationSystem()
        navigation_system.run()
    except Exception as e:
        print(f"Error: {e}")