# Imports
import os
import tifffile
import SimpleITK as sitk
from spotiflow.model import Spotiflow
import numpy as np
import csv

# Adjustable parameters
folder_path = '/PATH/TO/IMAGES/'     # Folder where images are saved (must terminate with /!)

filter_radius = 2           # Radius of the filter image

model_name = "hybiss"       # Option: general, hybiss, synth_complex
prob_threshold = 0.1        # Probability of detection. Between 0 and 1. Lower probability - more spots; higer probability - less spots.
min_distance = 3            # Minimal distance between two spots

N = 5                       # How many closest spots to find

pixel_size = 20             # Pixel size in nm

# Check the folder content
folder_content = os.listdir(folder_path)

# Remove anything that does not end with .tif
images = list()

for item in folder_content:
    if os.path.isfile(os.path.join(folder_path, item)) and item.endswith('.tif'):
        images.append(item)

# Initialize the spot prediction model
model = Spotiflow.from_pretrained(model_name)

# Function to turn predictions into the numpy format
def get_array_image_from_spotiflow_preds(probs, org_img_size: tuple):
    array = np.zeros(org_img_size, dtype=np.uint8)
    for coord in probs:
        array[int(coord[0]), int(coord[1])] = 1
    
    return array

# Function to calculate the euclidian distance in real units
def euclidean_distance(point1, point2, pixel_size):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)*pixel_size

# Make sure that folders for predictions and results are in place
if not os.path.exists(folder_path + 'Predictions'):
    os.makedirs(folder_path + 'Predictions')

if not os.path.exists(folder_path + 'Results'):
    os.makedirs(folder_path + 'Results')

# Loop through all the images
for img_name in images:
    # Read image
    tif_image = tifffile.imread(folder_path + img_name)
    img = sitk.GetImageFromArray(tif_image)

    # Filter image
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius(filter_radius)  # Adjust the radius for median filtering
    img = median_filter.Execute(img)

    # Mask image
    binary_mask = sitk.BinaryThreshold(img, lowerThreshold=10)
    img = sitk.Mask(img, binary_mask)

    # Convert to np
    img_np = sitk.GetArrayViewFromImage(img)

    # Predict the spots
    points, _ = model.predict(img_np, prob_thresh=prob_threshold, min_distance=min_distance)

    # Convert predictions (point cloud) into image-like array
    img_preds_np = get_array_image_from_spotiflow_preds(points, img_np.shape)   # Numpy image
    img_preds = sitk.GetImageFromArray(img_preds_np)                            # SITK image

    # Write prediction image into Predictions folder
    sitk.WriteImage(img_preds, folder_path + "Predictions/" + img_name[0:-4] + "_preds.nrrd")

    # Connected components
    img_preds_cc = sitk.ConnectedComponent(img_preds)

    # Analyze labels
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(img_preds_cc)
    labels = label_shape_filter.GetLabels()     # Get all labels

    # Iterate over spots/labels and find N closest
    closest_labels = dict()
    for label in labels:
        distances = list()
        for other_label in labels:
            if label != other_label:
                distance = euclidean_distance(label_shape_filter.GetCentroid(label), label_shape_filter.GetCentroid(other_label), pixel_size)
                distances.append(distance)
        
        distances.sort()
        closest_labels[label] = distances[:N]               # Only keep the closest 5

    # Save closest_labels to a CSV file
    with open(folder_path + 'Results/' + img_name[0:-4] + "_results.csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        
        writer.writerow(closest_labels.keys())              # Write the header
        writer.writerows(zip(*closest_labels.values()))     # Write the rows
