from rest_framework import serializers

class EmotionSerializer(serializers.Serializer):
    emotion = serializers.CharField(max_length=100)
