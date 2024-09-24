# Import this main file
from object_tracker import Object_Tracker

vc = Object_Tracker(window_name = 'output', weights='./checkpoints/obj_tracker', video='./1.mp4' )

# Run it
vc.run()