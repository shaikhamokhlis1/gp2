import os
import csv

# Set the base dataset folder
base_folder = 'dataset/Test'

# Output CSV file
output_csv = 'testing.csv'

# Initialize the rows list
rows = [['Image_Path', 'Label']]

# Iterate through Cataract and Normal folders
for label in ['Cataract', 'Normal']:
    folder_path = os.path.join(base_folder, label)
    if os.path.exists(folder_path):
        for image in os.listdir(folder_path):
            if image.endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if necessary
                # Create the full image path and normalize it to use forward slashes
                image_path = os.path.join(folder_path, image).replace('\\', '/')
                rows.append([image_path, label])

# Write to the CSV file
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

print(f"CSV file '{output_csv}' has been created with {len(rows) - 1} entries.")
