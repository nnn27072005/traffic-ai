from roboflow import Roboflow
rf = Roboflow(api_key="XwtWeFEZTiyYITifFwXW")  # free account
project = rf.workspace("ai-city-challenge").project("fisheye8k")
dataset = project.version(1).download("yolov8")  # compatible với ultralytics