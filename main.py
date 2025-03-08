from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load your trained model
model = YOLO("weights/best.pt")

# Path to the image you want to run inference on
img_path = "a.jpg"

# Run inference (model returns a list of results for each image)
results = model(img_path)

# # Print results summary in the console
# results.print()

# Visualize the results:
# Option 1: Using the built-in plot() method
img_with_boxes = results[0].plot()  # returns an image with bounding boxes drawn
plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Option 2: Save the image with predictions
results.save()  # saves to the default runs directory
