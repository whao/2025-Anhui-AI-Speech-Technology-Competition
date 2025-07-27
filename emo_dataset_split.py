# This script is used to randomly select a subset of audio dataset from emotion dataset for testing model inference.

# The structure of the dataset is assumed to be:
# - dataset/
#   - angry/
#   - fear/
#   - happy/
#   - neutral/
#   - sad/

# By looking at the audio files in each emotion folder, we can see that there are 200 audio files in each emotion folder.
# And the name of each audio file is in the format of `{201-250}-{emotion}-{speaker}.wav`.
# Numbers {201-250} means that the content of what the speaker is saying, same number means the same content.
# {emotion} and {speaker} are the emotion and speaker of the audio file, respectively.

# We will randomly select 10 audio files from each emotion folder, and save them all to a new folder named `data/emo_dataset_split`.
# Then we will hash the filenames of selected audio and save the each hashed filename and its corresponding {emotion} and original filename to a CSV file.
# The CSV file will be saved in the same folder. The CSV file will be named `emo_dataset_split.csv`.

import os
import random
import hashlib
import pandas as pd
# Set the path to the dataset
dataset_path = "dataset"

# Set the path to the output folder
output_folder = "data/emo_dataset_split"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
# Set the number of audio files to select from each emotion folder
num_files_to_select = 10
# Initialize a list to store the data for the CSV file
data = []
# Iterate through each emotion folder
for emotion in os.listdir(dataset_path):
    emotion_folder = os.path.join(dataset_path, emotion)
    if not os.path.isdir(emotion_folder):
        continue
    # Get all audio files in the emotion folder
    audio_files = [f for f in os.listdir(emotion_folder) if f.endswith('.wav')]
    # Randomly select the specified number of audio files
    selected_files = random.sample(audio_files, min(num_files_to_select, len(audio_files)))
    # Process each selected file
    for audio_file in selected_files:
        # Create the full path to the audio file
        audio_file_path = os.path.join(emotion_folder, audio_file)
        # Hash the filename
        hashed_filename = hashlib.sha256(audio_file.encode()).hexdigest()
        # Save the audio file to the output folder with the hashed filename
        new_audio_file_path = os.path.join(output_folder, f"{hashed_filename}.wav")
        os.rename(audio_file_path, new_audio_file_path)
        # Append the data for the CSV file
        data.append({
            'hashed_filename': hashed_filename,
            'emotion': emotion,
            'original_filename': audio_file
        })
# Create the CSV file
csv_file_path = os.path.join(output_folder, 'emo_dataset_split.csv')
df = pd.DataFrame(data)
df.to_csv(csv_file_path, index=False)
print(f"CSV file saved to {csv_file_path}")