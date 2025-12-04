# This import adds OpenCV so we can read, process, display, and record video frames
import cv2
# This import adds FaceAnalysis which bundles a modern face detector we can use directly
from insightface.app import FaceAnalysis
# This import adds type hints so lists and tuples of boxes are easier to understand
from typing import List, Tuple


# This class wraps face detection so callers only deal with simple bounding boxes
class FaceDetector:
    # This method prepares the face detector with a chosen list of execution providers
    def __init__(self, providers: List[str] | None = None) -> None:
        # This sets CPU provider by default so the detector works reliably on most machines
        if providers is None:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        # This stores the providers list so we remember how the detector was configured
        self.providers: List[str] = providers
        # This builds the FaceAnalysis app which includes a fast face detection model
        self.app: FaceAnalysis = FaceAnalysis(name="buffalo_s", providers=self.providers)
        # This prepares the detector for typical video resolutions to balance speed and accuracy
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    # This method takes a frame and returns a list of bounding boxes for all detected faces
    def detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        # This converts the frame from BGR to RGB so the detector reads colors correctly
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # This runs the detector on the RGB frame and returns a list of detected face objects
        faces = self.app.get(rgb_frame)
        # This creates an empty list where we will store bounding boxes from all faces
        boxes: List[Tuple[int, int, int, int]] = []
        # This loops through each detected face so we can extract the bounding box
        for face in faces:
            # This converts the bounding box coordinates to integer pixel positions
            x1, y1, x2, y2 = face.bbox.astype(int)
            # This appends the bounding box tuple to the list for later use in blurring
            boxes.append((x1, y1, x2, y2))
        # This returns the full list of face bounding boxes to the caller
        return boxes


# This class handles only the blurring of face regions on a given frame
class FaceBlurrer:
    # This method applies blur to all faces when enabled and returns the resulting frame
    def blur_faces(self, frame, boxes: List[Tuple[int, int, int, int]], enabled: bool):
        # This immediately returns the original frame unchanged if blurring is turned off
        if not enabled:
            return frame
        # This reads frame height and width so we can clip boxes to valid pixel ranges
        height, width = frame.shape[:2]
        # This creates a copy of the frame so we do not overwrite the source image directly
        output = frame.copy()
        # This loops over every bounding box so each detected face can be processed safely
        for (x1, y1, x2, y2) in boxes:
            # This clips the left coordinate so it never goes outside the image on the left
            x1_clamped = max(0, min(x1, width - 1))
            # This clips the top coordinate so it never goes outside the image at the top
            y1_clamped = max(0, min(y1, height - 1))
            # This clips the right coordinate so it never goes outside the image on the right
            x2_clamped = max(0, min(x2, width))
            # This clips the bottom coordinate so it never goes outside the image at the bottom
            y2_clamped = max(0, min(y2, height))
            # This skips this box if clamping produces an invalid or zero-size region
            if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
                continue
            # This crops the region of the frame where the current face is safely located
            face_region = output[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
            # This skips blurring if the cropped region is still empty for any reason
            if face_region.size == 0:
                continue
            # This applies a strong Gaussian blur to the cropped face region to hide identity
            blurred_region = cv2.GaussianBlur(face_region, (51, 51), 0)
            # This writes the blurred face region back into its original location on the frame
            output[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = blurred_region
        # This returns the final frame where all valid face regions have been blurred
        return output

# This class controls reading video, processing frames, showing them, and recording output
class BlurApp:
    # This method sets up video reading, detection, blurring, and recording destinations
    def __init__(self, source: str, output_path: str) -> None:
        # This stores the path to the input video file so we know which file to process
        self.source: str = source
        # This stores the output video path where processed frames will be recorded
        self.output_path: str = output_path
        # This opens the input video so frames can be read one by one from disk
        self.cap = cv2.VideoCapture(self.source)
        # This starts with blur turned on so the video begins in privacy mode by default
        self.blur_enabled: bool = True
        # This creates a FaceDetector instance so we can locate faces in each frame
        self.detector: FaceDetector = FaceDetector()
        # This creates a FaceBlurrer instance so we can blur any detected faces
        self.blurrer: FaceBlurrer = FaceBlurrer()
        # This starts with no video writer until we inspect the first frame’s size and fps
        self.writer = None
    
     # This method creates the video writer after we see the first frame so sizes and fps match well
    def _init_writer(self, frame) -> None:
        # This reads the frame height, width, and channel count from the image shape
        height, width, _ = frame.shape
        # This reads the frames-per-second value reported by the input video metadata
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # This falls back to a default fps if the input video did not report a valid value
        fps = 25.0
        # This defines the video codec so the writer saves an MP4 file on most systems
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # This creates the VideoWriter so each processed frame can be recorded to disk
        self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))


    # This method is the main loop that reads, processes, shows, and records every frame
    def run(self) -> None:
        # This starts an infinite loop that ends when the video finishes or the user quits
        while True:
            # This reads the next frame from the input video file
            ret, frame = self.cap.read()
            # This stops the loop if no frame is returned, which means the video has ended
            if not ret:
                break
            # This initializes the video writer once we have a real frame to measure
            if self.writer is None:
                self._init_writer(frame)
            # This processes the frame by detecting faces and optionally blurring them
            processed = self._process_frame(frame)
            # This displays the processed frame so the user can watch the live output
            cv2.imshow("Face Blur POC - Video", processed)
            # This records the processed frame so the final file shows blur ON and OFF states
            self._write_frame(processed)
            # This waits briefly for a key press and reads which key the user pressed
            key = cv2.waitKey(1) & 0xFF
            # This handles the key press and decides whether to keep running or stop
            if not self._handle_key(key):
                break
        # This releases the video capture object so the input file is properly closed
        self.cap.release()
        # This releases the writer if it was created so the output file is finalized
        if self.writer is not None:
            self.writer.release()
        # This closes all OpenCV windows that were opened by the application
        cv2.destroyAllWindows()

    # This method records a single processed frame into the output video file
    def _write_frame(self, frame) -> None:
        # This checks that the writer exists before trying to write a frame to disk
        if self.writer is not None:
            # This appends the processed frame to the output video file
            self.writer.write(frame)
    
    # This method wraps detection, blurring, and then delegates HUD drawing to a helper
    def _process_frame(self, frame):
        # This runs the face detector on the frame to find all face bounding boxes
        boxes = self.detector.detect_faces(frame)
        # This applies blur to the detected faces when blur mode is enabled
        processed = self.blurrer.blur_faces(frame, boxes, self.blur_enabled)

         # This draws blue rectangles and labels around each detected face on the processed frame
        processed = self._draw_face_boxes(processed, boxes)

        # This calls a helper method to draw the status HUD on the processed frame
        processed = self._draw_hud(processed)
        # This returns the final processed frame so it can be displayed and recorded
        return processed
    
     # This method draws blue bounding boxes and a 'face' label on each detected face
    def _draw_face_boxes(self, frame, boxes: List[Tuple[int, int, int, int]]):
        # This creates a copy of the frame so drawing does not affect the original reference
        output = frame.copy()
        # This loops over each face bounding box so every detected face can be outlined
        for (x1, y1, x2, y2) in boxes:
            # This draws a blue rectangle around the face area to highlight the detected region
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # This chooses a position slightly above the rectangle to place the label text
            label_position = (x1, max(0, y1 - 10))
            # This draws the word 'face' above the rectangle so viewers know what is being detected
            cv2.putText(output, "face", label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # This returns the frame with all face rectangles and labels drawn for visualization
        return output

        # This method draws a large blur status HUD with a yellow background and white text
    def _draw_hud(self, frame):
        # This reads the frame height and width so we can place the HUD near the top-right corner
        height, width = frame.shape[:2]
        # This builds a short status text that shows whether blur is currently on or off
        status_text = f"BLUR {'ON' if self.blur_enabled else 'OFF'}"
        # This selects the font style to use when drawing the blur status message
        font = cv2.FONT_HERSHEY_SIMPLEX
        # This sets a larger font scale so the HUD text appears big and easy to notice
        font_scale = 1.2
        # This sets the line thickness so the HUD text looks bold and readable
        thickness = 2
        # This measures the size of the status text so we can design an enclosing background box
        (text_w, text_h), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)
        # This sets horizontal padding so the yellow background extends beyond the text edges
        padding_x = 12
        # This sets vertical padding so the yellow background gives breathing room above and below text
        padding_y = 8
        # This computes the total width of the background box including horizontal padding
        box_width = text_w + 2 * padding_x
        # This computes the total height of the background box including vertical padding
        box_height = text_h + 2 * padding_y
        # This computes the x coordinate so the box appears near the top-right with a small margin
        x = max(10, width - box_width - 10)
        # This sets the y coordinate so the box sits slightly below the top border of the frame
        y = 10
        # This defines the top-left corner of the yellow background box
        top_left = (x, y)
        # This defines the bottom-right corner of the yellow background box
        bottom_right = (x + box_width, y + box_height)
        # This draws a filled yellow rectangle as the HUD background so the text stands out clearly
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), -1)
        # This computes the x position where the white text will start inside the yellow box
        text_x = x + padding_x
        # This computes the y position so the text baseline sits nicely within the yellow box
        text_y = y + padding_y + text_h
        # This draws the blur status text in white on top of the yellow background for high contrast
        cv2.putText(frame, status_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        # This returns the frame with the updated HUD so it can be displayed and recorded
        return frame



    # This method handles key presses so the user can toggle blur or exit the app
    def _handle_key(self, key: int) -> bool:
        # This checks if the user pressed 'b' to switch blur between enabled and disabled
        if key == ord('b'):
            # This flips the blur flag so upcoming frames show or hide face blurring
            self.blur_enabled = not self.blur_enabled
        # This checks if the user pressed 'q' to stop processing and close the application
        if key == ord('q'):
            # This returns False to tell the caller to break out of the main loop
            return False
        # This returns True so the main loop continues when no exit key is pressed
        return True


# This block makes sure the app runs only when this script is executed directly
if __name__ == "__main__":
    # This sets the path to the input video file you want to process
    input_video_path = "input.mp4"
    # This sets the path where the processed video with blur toggles will be saved
    output_video_path = "output_blur_poc.mp4"
    # This creates a BlurApp instance configured with the chosen input and output paths
    app = BlurApp(input_video_path, output_video_path)
    # This starts the main loop so you can watch and record blur-on and blur-off behavior
    app.run()
