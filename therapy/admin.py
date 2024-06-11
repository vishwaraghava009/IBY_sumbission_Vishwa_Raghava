from django.contrib import admin
from .models import EmotionAnalysis, ToneAnalysis, Transcription, TherapistResponse

admin.site.register(EmotionAnalysis)
admin.site.register(ToneAnalysis)
admin.site.register(Transcription)
admin.site.register(TherapistResponse)

