from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

def index(request):
    if request.method == 'POST' and request.FILES['therapist_image']:
        therapist_image = request.FILES['therapist_image']
        fs = FileSystemStorage()
        filename = fs.save(therapist_image.name, therapist_image)
        uploaded_file_url = fs.url(filename)
        return render(request, 'therapy/index.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'therapy/index.html')
