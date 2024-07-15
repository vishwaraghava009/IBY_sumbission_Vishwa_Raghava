
adasr_repo_base_dir="/home/iby_vishwa/Documents/Simple_Pipeline/AdaSR-TalkingHead" # change the path according to you machine

source_image="$adasr_repo_base_dir/DEMO/Face2.jpeg" #change the path to your custom image
audio_file="$adasr_repo_base_dir/DEMO/audio_file.wav" #change the path to your custom audio
driving_video_1="$adasr_repo_base_dir/DEMO/demo_video_1.mp4" #change the path to your custom driving video
driving_video_2="$adasr_repo_base_dir/DEMO/demo_video_2.mp4"


# Extract base names without extension
source_image_base=$(basename "$source_image" .jpeg)
driving_video_base_1=$(basename "$driving_video_1" .mp4)
driving_video_base_2=$(basename "$driving_video_2" .mp4)
audio_file_base=$(basename "$audio_file" .wav)

# Construct the result video filenames within the repo directory
result_video_1="${adasr_repo_base_dir}/${source_image_base}_${driving_video_base_1}.mp4"
result_video_2="${adasr_repo_base_dir}/${source_image_base}_${driving_video_base_2}.mp4"

# Print debug information
echo "Result video path 1: $result_video_1"
echo "Result video path 2: $result_video_2"
echo "Audio file path: $audio_file"

# Save the generated video filenames and audio file path to a temporary file
echo "$result_video_1" > "$adasr_repo_base_dir/generated_videos.txt"
echo "$result_video_2" >> "$adasr_repo_base_dir/generated_videos.txt"
echo "$audio_file" >> "$adasr_repo_base_dir/generated_videos.txt"

# Run the demo for the first driving video
python "$adasr_repo_base_dir/demo.py" \
    --config "$adasr_repo_base_dir/config/mix-resolution.yml" \
    --checkpoint "$adasr_repo_base_dir/checkpoints/mix-train.pth.tar" \
    --source_image "$source_image" \
    --driving_video "$driving_video_1" \
    --result_video "$result_video_1" \
    --relative

# Uncomment to run the demo for the second driving video
# python "$adasr_repo_base_dir/demo.py" \
#     --config "$adasr_repo_base_dir/config/mix-resolution.yml" \
#     --checkpoint "$adasr_repo_base_dir/checkpoints/mix-train.pth.tar" \
#     --source_image "$source_image" \
#     --driving_video "$driving_video_2" \
#     --result_video "$result_video_2" \
#     --relative
