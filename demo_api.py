from src.task3_classifier import api

label = api.predict("path/to/your_image.png")
print("Predicted defect:", label)