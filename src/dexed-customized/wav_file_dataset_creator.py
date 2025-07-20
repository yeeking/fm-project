import os
import sys
import shutil
import numpy as np

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_folder> <output_folder> <percentage>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    percentage = float(sys.argv[3])

    # Check input folder
    if not os.path.isdir(input_folder):
        raise AssertionError(f"Input folder does not exist: {input_folder}")

    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)

    # Search recursively for .wav files
    wav_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(root, file)
                wav_files.append(full_path)

    total_files = len(wav_files)
    if total_files == 0:
        print("No .wav files found.")
        return

    n_select = max(1, int((percentage / 100.0) * total_files))
    selected_files = list(np.random.choice(wav_files, size=n_select, replace=False))

    print(f"Found {total_files} .wav files. Selecting {n_select} ({percentage}%).")

    # Copy selected files to output folder
    for file_path in selected_files:
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_folder, filename)
        shutil.copy2(file_path, out_path)

    print(f"Copied {len(selected_files)} files to {output_folder}")

if __name__ == "__main__":
    main()

