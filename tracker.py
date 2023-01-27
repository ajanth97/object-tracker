import cv2
import sys
import pandas as pd

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]

def createTracker():
    # Set up tracker.
    # Instead of CSRT, you can also use
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
             tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    return tracker

def openVideoFile(path):
    # Read video
    video = cv2.VideoCapture(path)
    #video = cv2.VideoCapture(0) # for using CAM

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    return video

#Helper function to get center coordinate from bounding box
#bbox has the form (x,y,w,h) where x and y is the top left corner of the bouding box.
# w and h is width and height of bounding box
def getCenterCoordinateFromBbox(bbox):
    center_coordinate = (bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2))        
    return center_coordinate

#Helper function to calculate the change in coordinates
def getCoordinateChange(start_coordinate, final_coordinate):
    return (final_coordinate[0]-start_coordinate[0], final_coordinate[1]-start_coordinate[1])

if __name__ == '__main__' : 
    tracker = createTracker()
    videoFile = openVideoFile("final.mov")  
    # Read first frame.
    #For loop of 30 used to create delay to avoid hand covering the ball
    for i in range(60):
        ok, frame = videoFile.read()
        if not ok:
            print ('Cannot read video file')
            sys.exit()
    initial_bbox = cv2.selectROI(frame, True)
    start = tracker.init(frame, initial_bbox)
    starting_coordinate = getCenterCoordinateFromBbox(initial_bbox)  
    time_elasped = 0
    coordinate_time_data = []
    while True:
        ok, frame = videoFile.read()
        if not ok:
            break
        # Start timer
        startCount = cv2.getTickCount()
        # Update tracker
        ok, bbox = tracker.update(frame)
        time_elasped = time_elasped + ((cv2.getTickCount() - startCount) / cv2.getTickFrequency())
        current_coordinate = getCenterCoordinateFromBbox(bbox)
        coordinate_change = getCoordinateChange(starting_coordinate,current_coordinate)
        coordinate_time_relation = (coordinate_change[0],coordinate_change[1],time_elasped)
        coordinate_time_data.append(coordinate_time_relation)
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - startCount)       
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            #print(bbox)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        # Display result
        cv2.imshow("Tracking", frame)
        # Exit if ESC pressed
        if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
            break      
    videoFile.release()
    # Create pandas dataframe our list of coordinates delta data
    df = pd.DataFrame(coordinate_time_data, columns=["X delta", "Y delta", "Time (s)"])
    print(df)
    #Output CSV file from dataframe
    df.to_csv("data.csv",sep='\t', encoding='utf-8')
    cv2.destroyAllWindows()    