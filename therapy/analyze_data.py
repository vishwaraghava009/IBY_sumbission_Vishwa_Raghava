from .models import EmotionAnalysis, ToneAnalysis, Transcription

def analyze_data():
    transcriptions = Transcription.objects.all()
    for transcription in transcriptions:
        aligned_emotions = []
        timestamp = transcription.timestamp
        tone_analysis = ToneAnalysis.objects.filter(timestamp__gte=timestamp).order_by('timestamp')
        emotion_analysis = EmotionAnalysis.objects.filter(timestamp__gte=timestamp).order_by('timestamp')

        for sentence in transcription.text.split('. '):
            sentence_end = timestamp + len(sentence.split()) * 1000  # Assuming 1 second per word

            # Get emotions and tones within the time frame of the sentence
            tones = tone_analysis.filter(timestamp__lte=sentence_end)
            emotions = emotion_analysis.filter(timestamp__lte=sentence_end)

            if tones.exists() and emotions.exists():
                dominant_emotion = emotions.first().emotion
                aligned_emotions.append(f"{{{dominant_emotion}}} {sentence}")

            timestamp = sentence_end

        transcription.text = '. '.join(aligned_emotions)
        transcription.save()

if __name__ == "__main__":
    analyze_data()
