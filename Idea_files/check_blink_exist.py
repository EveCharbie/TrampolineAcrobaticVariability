import os
from pathlib import Path

# Define the directory path
directory_path = "/home/lim/disk/Eye-tracking/PupilData/CloudExport"

# List to hold the names of all .mp4 files
mp4_folder = []
# Walk through the directory and subdirectories to find .mp4 files
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith(".mp4"):
            # Append the file name to the list
            mp4_folder.append(os.path.join(root, file))
mp4_folder

# Dictionary to hold folders and their MP4 files
folder_mp4_files = {}
# Populate the dictionary with folders as keys and a list of their MP4 files as values
for file_path in mp4_folder:
    folder_path = os.path.dirname(file_path)  # Extract the folder path
    folder_name = os.path.basename(folder_path)  # Extract the folder name
    file_name = os.path.basename(file_path)  # Extract the file name
    if folder_name not in folder_mp4_files:
        folder_mp4_files[folder_name] = []
    folder_mp4_files[folder_name].append(file_name)
# Find the name of the single MP4 file in folders where there is exactly one MP4 file
mp4_names_in_folders_with_one_file = [files[0] for files in folder_mp4_files.values() if len(files) == 1]
mp4_names_in_folders_with_one_file

# Filter files to exclude those starting with "PI"
filtered_files_exclude_pi = [file for file in mp4_folder if not os.path.basename(file).startswith("PI")]
# Extracting just the file names for clarity
all_files_mp4 = [os.path.basename(file) for file in filtered_files_exclude_pi]
# Extract the part of the file name before the first underscore
extracted_parts_fils_mp4 = [name.split("_")[0] for name in all_files_mp4]
extracted_parts_fils_mp4


pkl_files = "/home/lim/disk/Eye-tracking/PupilData/blinks_labeled/"
# List to hold the names of all files
all_files = []
# List only files in the given directory
for file in os.listdir(pkl_files):
    if os.path.isfile(os.path.join(pkl_files, file)):
        all_files.append(file)
# Extract the part of the file name before the first underscore
extracted_parts_fils_labeled = [name.split("_")[0] for name in all_files]


# Compare the lists and find identifiers in one list but not in the other
in_mp4_not_in_labeled = set(extracted_parts_fils_mp4) - set(extracted_parts_fils_labeled)
in_labeled_not_in_mp4 = set(extracted_parts_fils_labeled) - set(extracted_parts_fils_mp4)

print(in_mp4_not_in_labeled)
print(mp4_names_in_folders_with_one_file)

# Les identifiants des fichiers à chercher
# Chemin du dossier A
root_dir = Path('/home/lim/disk/Eye-tracking/PupilData/CloudExport')

def find_files(directory, file_ids):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if any(file_id in filename for file_id in file_ids):
                # Extrayez le nom des dossiers B et C à partir du chemin
                relative_path = Path(dirpath).relative_to(root_dir)
                parts = relative_path.parts
                if len(parts) >= 2:
                    # parts[0] est le dossier B et parts[1] est le dossier C
                    print(f"ID de fichier '{filename}' trouvé dans B: {parts[0]}, C: {parts[1]}")

# Exécutez la fonction
tolabeling= find_files(root_dir, in_mp4_not_in_labeled)

print(tolabeling)