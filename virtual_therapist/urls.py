"""
URL configuration for virtual_therapist project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from therapy import views
from django.conf import settings
from therapy.views import list_images_api
from therapy.api_views import EmotionDetectionView, SpeechAnalysisView
from django.conf.urls.static import static




urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('api/list-images/', list_images_api, name='list_images_api'),
    path('api/emotion-detection/', EmotionDetectionView.as_view(), name='emotion-detection'),
    path('api/speech-analysis/', SpeechAnalysisView.as_view(), name='speech-analysis'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

