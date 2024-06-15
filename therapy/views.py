from django.shortcuts import render
from django.conf import settings
import os

def list_images():
    pics_dir = os.path.join(settings.MEDIA_ROOT, 'pics')
    images = [f for f in os.listdir(pics_dir) if os.path.isfile(os.path.join(pics_dir, f))]
    return images

def index(request):
    images = list_images()
    if request.method == 'POST':
        selected_image = request.POST.get('selected_image')
        if selected_image:
            uploaded_file_url = os.path.join('media', 'pics', selected_image)
            return render(request, 'therapy/index.html', {'uploaded_file_url': uploaded_file_url, 'images': images})
    return render(request, 'therapy/index.html', {'images': images})
