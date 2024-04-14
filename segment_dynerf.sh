#!/bin/bash

# Define the base directory where the folders are located
base_dir="/home/ubuntu/datasets/dynerf/cook_spinach/segments"

# Loop through each segment folder
for segment in {1..5}; do
    # Calculate the start and end frame indices for each segment
    start_index=$(( (segment - 1) * 60 ))
    end_index=$(( start_index + 59 ))

    # Format the folder name
    segment_folder="$base_dir/2s_$(printf "%02d" $segment)"

    # Loop through each camera sub-folder
    for cam_folder in "$segment_folder"/cam*; do
        echo "Processing $cam_folder ..."

        # Move into the camera folder
        cd "$cam_folder/images"

        # Remove all images outside the required range
        for file in *.png; do
            # Extract the frame number from the filename
            frame_num=${file%.png}
            frame_num=${frame_num#0}  # Remove leading zeros for numerical comparison

            # Check if the frame number is outside the required range
            if [ "$frame_num" -lt "$start_index" ] || [ "$frame_num" -gt "$end_index" ]; then
                # Delete the file if outside the range
                rm "$file"
            fi
        done

       # If start_index is not 0, rename the files to start from 0
        if [ "$start_index" -ne 0 ]; then
            for file in *.png; do
                # Extract the frame number from the filename
                frame_num=${file%.png}
                frame_num=$((10#${frame_num#0}))  # Convert to decimal to handle numbers with leading zeros correctly

                # Rename the file to start from 0000.png, using -- to prevent issues with leading -
                new_file_name=$(printf "%04d.png" $((frame_num - start_index)))
                mv -- "$file" "$new_file_name"
            done
        fi

        echo "Completed processing $cam_folder"
    done
done

echo "All images have been organized."
