
musetalk_repo_base_dir="/home/iby_vishwa/Documents/AdaSR/MuseTalk" #change the path

adasr_repo_base_dir="/home/iby_vishwa/Documents/AdaSR/AdaSR-TalkingHead" #chhnge the path

generated_videos_file="$adasr_repo_base_dir/generated_videos.txt"

generated_video_1=$(sed -n '1p' "$generated_videos_file")
generated_video_2=$(sed -n '2p' "$generated_videos_file")
audio_file=$(sed -n '3p' "$generated_videos_file")

musetalk_config_file="$musetalk_repo_base_dir/configs/inference/realtime.yaml"

sed -i "s|video_path: \".*\"|video_path: \"$generated_video_1\"|" "$musetalk_config_file"
sed -i "s|audio_0: \".*\"|audio_0: \"$audio_file\"|" "$musetalk_config_file"
sed -i "/audio_1:/d" "$musetalk_config_file"

echo "Updated config file with video path: $generated_video_1 and audio path: $audio_file"
cat "$musetalk_config_file"

export PYTHONPATH="$musetalk_repo_base_dir:$PYTHONPATH"

python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml --batch_size 4