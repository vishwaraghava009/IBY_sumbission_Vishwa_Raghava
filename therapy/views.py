import os
import subprocess
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

def list_images():
    pics_dir = os.path.join(settings.MEDIA_ROOT, 'pics')
    images = [f for f in os.listdir(pics_dir) if os.path.isfile(os.path.join(pics_dir, f))]
    return images


def check_video_existence(request):
    lip_sync_video_path = os.path.join(settings.MUSETALK_ROOT, 'results', 'avatars', 'avatar1', 'vid_output', 'audio_0.mp4')
    video_exists = os.path.exists(lip_sync_video_path)
    return JsonResponse({'exists': video_exists})


@csrf_exempt
def index(request):
    images = list_images()
    lip_sync_video_url = None
    if request.method == 'POST':
        selected_image = request.POST.get('selected_image')
        if selected_image:
            uploaded_file_url = os.path.join('media', 'pics', selected_image)
            
            # Define paths for source image, driving video, and output video
            source_image_path = os.path.join(settings.MEDIA_ROOT, 'pics', selected_image)
            driving_video_path = os.path.join(settings.MEDIA_ROOT, 'drive_video', 'fem_nod_crop.mp4')  # Example driving video path
            output_video_dir = os.path.join(settings.MEDIA_ROOT, 'motion_video')
            output_video_path = os.path.join(output_video_dir, f'{selected_image.split(".")[0]}_motion.mp4')

            # Ensure the output directory exists
            os.makedirs(output_video_dir, exist_ok=True)
            
            try:
                # Run the motion transfer script
                subprocess.run([
                    'python', 'LIA/run_demo.py',
                    '--model', 'vox',
                    '--source_path', source_image_path,
                    '--driving_path', driving_video_path,
                    '--save_folder', output_video_dir
                ], check=True)
                print("Motion transfer complete. Video saved.")
            except subprocess.CalledProcessError as e:
                print(f"Error running motion transfer: {e}")
                return HttpResponse(f"Error running motion transfer: {e}", status=500)

            # Check if the lip sync video exists
            lip_sync_video_path = os.path.join(settings.MUSETALK_ROOT, 'results', 'avatars', 'avatar1', 'vid_output', 'audio_0.mp4')
            if os.path.exists(lip_sync_video_path):
                lip_sync_video_url = os.path.join(settings.MUSETALK_URL, 'results', 'avatars', 'avatar1', 'vid_output', 'audio_0.mp4')

            return render(request, 'therapy/index.html', {
                'uploaded_file_url': uploaded_file_url,
                'images': images,
                'lip_sync_video_url': lip_sync_video_url
            })

    return render(request, 'therapy/index.html', {'images': images})

def list_images_api(request):
    images = list_images()
    return JsonResponse({'images': images})
