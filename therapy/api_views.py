import os
import json
import subprocess
import torch
import time
import wave
import numpy as np
import cv2
import speech_recognition as sr
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from django.http import JsonResponse
from deepface import DeepFace
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import EmotionAnalysis, ToneAnalysis, Transcription, TherapistResponse
from collections import Counter
from groq import Groq
from dotenv import load_dotenv
import yaml
import logging
import shutil
from django.conf import settings


load_dotenv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
GROQ_API_KEY = "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"  #my groq_api_key replace it with yours if this doen't work(as itvgets exposed on git now)
client = Groq(api_key=GROQ_API_KEY)


class CheckVideoExistenceAPI(APIView):
    def get(self, request, *args, **kwargs):
        session_id = request.GET.get('session_id')
        if session_id:
            video_path = os.path.join(settings.MEDIA_ROOT, 'output_video', f'{session_id}_audio_0.mp4')
            exists = os.path.exists(video_path)
            return JsonResponse({'exists': exists})
        return JsonResponse({'error': 'No session_id provided'}, status=400)


class EmotionDetectionView(APIView):
    def post(self, request, *args, **kwargs):
        if 'frame' not in request.FILES or 'timestamp' not in request.POST or 'session_id' not in request.POST:
            return Response({'error': 'No frame, timestamp, or session_id provided'}, status=400)
        
        file = request.FILES['frame']
        timestamp = request.POST['timestamp']
        session_id = request.POST['session_id']
        print(f"Received timestamp: {timestamp}")
        try:
            timestamp = float(timestamp)
            file_bytes = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Frame could not be decoded')

            print("Analyzing frame...")
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            print(f"Analysis result: {analysis}")

            if analysis and isinstance(analysis, list) and 'dominant_emotion' in analysis[0]:
                dominant_emotion = analysis[0]['dominant_emotion']
                confidence = analysis[0]['emotion'][dominant_emotion]
                #save to database
                EmotionAnalysis.objects.create(session_id=session_id, timestamp=timestamp, emotion=dominant_emotion, confidence=confidence)
            else:
                dominant_emotion = 'No dominant emotion detected'
            
            return Response({'emotion': dominant_emotion})
        except Exception as e:
            print(f"Error: {str(e)}")  
            return Response({'error': str(e)}, status=400)

def segment_transcription(transcription, tone_emotions, facial_emotions, window_size=10000):
    words = transcription.split()
    segments = []
    current_segment = []
    facial_emotion_counts = Counter()
    tone_emotion_counts = Counter()
    current_timestamp = 0

    print("Transcription words:", words)
    print("Tone emotions:", tone_emotions)
    print("Facial emotions:", facial_emotions)

    for i, word in enumerate(words):
        word_timestamp = i * (1000 // len(words))  
        print(f"Word: {word}, Word Timestamp: {word_timestamp}")

        for timestamp, emotion in tone_emotions:
            if word_timestamp >= timestamp:
                tone_emotion_counts[emotion] += 1

        for record in facial_emotions:
            if word_timestamp >= record['timestamp']:
                facial_emotion_counts[record['emotion']] += 1

        if i > 0 and (i % window_size == 0):
            most_common_facial_emotion = facial_emotion_counts.most_common(1)[0][0] if facial_emotion_counts else 'neutral'
            most_common_tone_emotion = tone_emotion_counts.most_common(1)[0][0] if tone_emotion_counts else 'neutral'
            
            segments.append({
                'text': ' '.join(current_segment),
                'timestamp': current_timestamp,
                'facial_emotion': most_common_facial_emotion,
                'tone_emotion': most_common_tone_emotion
            })
            current_segment = []
            facial_emotion_counts.clear()
            tone_emotion_counts.clear()
            current_timestamp = word_timestamp

        current_segment.append(word)

    if current_segment:
        most_common_facial_emotion = facial_emotion_counts.most_common(1)[0][0] if facial_emotion_counts else 'neutral'
        most_common_tone_emotion = tone_emotion_counts.most_common(1)[0][0] if tone_emotion_counts else 'neutral'
        
        segments.append({
            'text': ' '.join(current_segment),
            'timestamp': current_timestamp,
            'facial_emotion': most_common_facial_emotion,
            'tone_emotion': most_common_tone_emotion
        })

    for segment in segments:
        segment['text'] = f"{{{segment['facial_emotion']}, {segment['tone_emotion']}}} {segment['text']}"

    print("Segments:", segments)
    return segments

class SpeechAnalysisView(APIView):
    def post(self, request, *args, **kwargs):
        if 'audio' not in request.FILES or 'timestamps' not in request.POST or 'session_id' not in request.POST:
            return Response({'error': 'No audio file, timestamps, or session_id provided'}, status=400)
        
        audio_file = request.FILES['audio']
        timestamps = json.loads(request.POST['timestamps'])
        session_id = request.POST['session_id']
        print("Received audio file")
        try:
            with open(f"temp_audio_{session_id}.webm", "wb") as f:
                f.write(audio_file.read())
                print("Saved audio file as temp_audio.webm")

            ffmpeg_command = [
                'ffmpeg',
                '-y',  
                '-i', f'temp_audio_{session_id}.webm',
                f'temp_audio_{session_id}.wav'
            ]
            subprocess.run(ffmpeg_command, check=True)
            print(f"Converted audio file to temp_audio_{session_id}.wav")

            try:
                wf = wave.open(f"temp_audio_{session_id}.wav", 'rb')
                print(f"Audio file format: {wf.getnchannels()} channels, {wf.getsampwidth()} bytes per sample, {wf.getframerate()} Hz")
                wf.close()
            except wave.Error as e:
                print(f"Error reading audio file: {e}")
                return Response({'error': 'Error reading audio file'}, status=400)

            transcription = recognize_speech_from_file(f"temp_audio_{session_id}.wav")
            print("Transcription:", transcription)
            
            tone_features = analyze_audio_tone(f"temp_audio_{session_id}.wav")
            print("Tone features:", tone_features)

            if len(tone_features) != len(timestamps):
                print("Length mismatch between tone features and timestamps")
                min_length = min(len(tone_features), len(timestamps))
                tone_features = tone_features[:min_length]
                timestamps = timestamps[:min_length]

            #save tone analysis with timestamps
            tone_emotions = []
            for i, timestamp in enumerate(timestamps):
                emotion = map_tone_features_to_emotion(tone_features[i])
                tone_emotions.append((timestamp, emotion))
                ToneAnalysis.objects.create(session_id=session_id, timestamp=timestamp, emotion=emotion)

            print("Tone emotions:", tone_emotions)

            #facial emotions
            facial_emotions = list(EmotionAnalysis.objects.filter(session_id=session_id).values('timestamp', 'emotion'))
            print("Facial emotions:", facial_emotions)

            #stucture transcription 
            segmented_transcriptions = segment_transcription(transcription, tone_emotions, facial_emotions, window_size=10000)
            print("Segmented transcriptions:", segmented_transcriptions)
            
            for segment in segmented_transcriptions:
                Transcription.objects.create(
                    session_id=session_id,
                    text=segment['text'],
                    timestamp=segment['timestamp']
                )

            #prompt for LLM
            user_transcription = ' '.join([seg['text'] for seg in segmented_transcriptions])
            prompt = f"""Imagine you are playing the "Brain of a therapist" role. You'll be given what your patient is speaking in this conversation, along with the emotion and tone at the beginning of each sentence he speaks in a single reply to you. For example:

            User: {user_transcription}

            For example:
            {{sad, sad}} I'm feeling a bit down today. {{happy, surprise}} But, I met my old friend and spent time with him. {{happy, sarcastic}} We thought of playing cricket.

            Here, the first term inside the brackets is regarding the facial emotion of the patient while speaking the sentence following the brackets, and similarly, the second term inside the brackets is regarding the tone of his speech while uttering the sentence following the brackets.

            Available emotions are:
            [Happy, Sad, Anger, Surprise, Fear, Disgust, Contempt]

            Available tones are:
            [Happy, Sad, Anger, Surprise, Fear, Disgust, Sarcastic]

            You have to analyse the transcription of the sentence he speaks, taking the emotion and tone as the additional attributes to extract the actual meaning of the transcription or how the patient feels while saying a particular sentence.

            You have to write your analysis of the patient's reply, and following it, you also have to craft what the therapist has to reply in 30 words. The reply has to be crafted based on the analysis. Also, the reply has to be in a way that it sounds realistic when spoken, that is, adding some reactional words or terms that have no meaning but make the conversation reactional and realistic, for example, "Aw...", "Umm...", "a term that can introduce pausing as if the therapist thinking a second before speaking just like a human does but donot include instructions like *smiles*, *pauses* etc. We cannot get them spoken by a TTS model, etc.

            The output structure should be as follows:
            Analysis: (analysis of the user's reply)
            Therapist: (therapist reply)

            Okay, now, start the conversation by observing the reply from him below for a greeting, and you'll get replies from him to perform the entire process described above. Do not answer anything else in the output apart from the above structure.

            Therapist: Hello! What do you want to express today? I am here to listen and not judge you in any way.
            Patient: {user_transcription}
            """

            #call the LLM
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "you are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stop=None
            )
            llm_output = response.choices[0].message.content
            print("LLM Response:", llm_output)

            #extract the therapist's reply
            therapist_reply = llm_output.split("Therapist: ")[-1].strip()
            print("Therapist Reply:", therapist_reply)

            #save the LLM output to a file
            llm_output_path = os.path.abspath(f"media/{session_id}_llm_output.txt")
            with open(llm_output_path, "w") as f:
                f.write(therapist_reply)

            #call the inference script
            audio_filename = os.path.abspath(f"media/audio/{session_id}_response.wav")
            inference_command = [
                'python',
                os.path.join(os.path.dirname(__file__), '..', 'StyleTTS2', 'run_inference.py'),
                llm_output_path,
                audio_filename
            ]
            subprocess.run(inference_command, check=True)
            print(f"Audio saved to {audio_filename}")

            #call the MuseTalk inference
            handle_new_audio_file(audio_filename, session_id)
            
            return Response({
                'transcription': transcription,
                'tone_emotions': tone_emotions,
                'llm_response': therapist_reply,
                'audio_url': audio_filename,
                'video_url': f'/MuseTalk/results/avatars/avator_1/vid_output/audio_0.mp4'
            })
        except subprocess.CalledProcessError as e:
            print(f"Error running inference script: {e}")
            return Response({'error': 'Error running inference script'}, status=400)
        except Exception as e:
            print(f"Error: {str(e)}")
            return Response({'error': str(e)}, status=400)

def handle_new_audio_file(audio_file_path, session_id):
    logging.info(f"Handling new audio file: {audio_file_path}")
    motion_video_dir = os.path.abspath("media/motion_video/vox/")
    output_video_dir = os.path.abspath(f"media/output_video/")
    
    while True:
        motion_videos = [f for f in os.listdir(motion_video_dir) if f.endswith('.mp4')]
        if motion_videos:
            motion_video_path = os.path.join(motion_video_dir, motion_videos[0])
            logging.info(f"Motion video found: {motion_video_path}")
            break
        logging.info("Waiting for motion video...")
        time.sleep(5)
    
    avatar_dir = os.path.abspath("MuseTalk/results/avatars/avator_1/")
    preparation_needed = not os.path.exists(avatar_dir)
    
    run_musetalk_inference()

def run_musetalk_inference():
    command = [
        'python', '-m', 'scripts.realtime_inference',
        '--inference_config', 'configs/inference/realtime.yaml'
    ]
    logging.info(f"Running MuseTalk inference: {command}")
    subprocess.run(command, cwd='MuseTalk')

def recognize_speech_from_file(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio)
            return transcription
        except sr.RequestError:
            return "API unavailable"
        except sr.UnknownValueError:
            return "Unable to recognize speech"

def analyze_audio_tone(file_path):
    [Fs, x] = audioBasicIO.read_audio_file(file_path)
    window_size = 0.05  
    step_size = 0.025  
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, window_size * Fs, step_size * Fs)
    return F.T  

def map_tone_features_to_emotion(tone_features):
    mean_val = np.mean(tone_features)
    std_val = np.std(tone_features)

    if mean_val > 0.5 and std_val < 0.2:
        return "happy"
    elif mean_val < 0.5 and std_val > 0.2:
        return "sad"
    elif mean_val < 0.2:
        return "neutral"
    else:
        return "surprised"