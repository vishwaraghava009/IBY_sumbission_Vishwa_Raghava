from django.db import models

class EmotionAnalysis(models.Model):
    session_id = models.CharField(max_length=100)
    timestamp = models.FloatField()
    emotion = models.CharField(max_length=100)
    confidence = models.FloatField()

class ToneAnalysis(models.Model):
    session_id = models.CharField(max_length=100)
    timestamp = models.FloatField()
    emotion = models.CharField(max_length=50, default="neutral")

class Transcription(models.Model):
    session_id = models.CharField(max_length=100)
    text = models.TextField()
    timestamp = models.FloatField()

class TherapistResponse(models.Model):
    session_id = models.CharField(max_length=100)
    text = models.TextField()
    audio = models.FileField(upload_to='therapist_responses/', null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
