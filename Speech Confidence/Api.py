import os
import re
import io 
import boto3
import joblib
import wave 
import shutil
import whisper
import librosa
import webrtcvad
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from flask import Flask, request, jsonify
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score





app = Flask(__name__)

S3_BUCKET = 'your-s3-bucket-name'
S3_REGION = 'your-region'
AWS_ACCESS_KEY = 'your-access-key'
AWS_SECRET_KEY = 'your-secret-key'

s3_client = boto3.client(
    's3',
    region_name=S3_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

directory = r"" #S3 directory path.


MODEL_PATH = r'C:\Users\Technologist\OneDrive - Higher Education Commission\Job Project\Speech Confidence\model_pkl_V2.pkl' # replace the path if required.  
model = joblib.load(MODEL_PATH) 


def preprocess_audio(s3_path):  
    def trim_audio(input_file, output_file, duration_limit=180):
        audio = AudioSegment.from_file(input_file)
        duration = len(audio) / 1000

        if duration > duration_limit:
            trimmed_audio = audio[duration_limit * 1000:]
            trimmed_audio.export(output_file, format="wav")
            print(f"Trimmed {input_file}.")
        else:
            print(f"\n{input_file} is shorter than {duration_limit} seconds. Skipping Trim Function.")

    def process_directory_from_s3(s3_directory, duration_limit=180):
        temp_directory = "temp_audio_files"
        os.makedirs(temp_directory, exist_ok=True)

        result_files = []
        try:
            files = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_directory)

            for file in files.get('Contents', []):
                file_name = file['Key']
                if file_name.endswith(".wav"): 
                    local_path = os.path.join(temp_directory, secure_filename(file_name))
                    s3_client.download_file(S3_BUCKET, file_name, local_path)

                    trimmed_local_path = os.path.join(temp_directory, f"Trimmed_{secure_filename(file_name)}")
                    trim_audio(local_path, trimmed_local_path, duration_limit)

                    trimmed_s3_path = f"{s3_directory}/Trimmed_{os.path.basename(file_name)}"
                    s3_client.upload_file(trimmed_local_path, S3_BUCKET, trimmed_s3_path)
                    result_files.append(trimmed_s3_path)

                    os.remove(local_path)
                    os.remove(trimmed_local_path)

            return result_files

        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            shutil.rmtree(temp_directory)

    
    trimmed_files = process_directory_from_s3(directory)
    print(f"Trimmed files uploaded to S3: {trimmed_files}") 



    def load_trimmed_audio_data_from_s3(s3_directory, target_sr=22050, mono=True):
        s3_client = boto3.client(
            's3',
            region_name=S3_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        
        audio_file_dict = {}
        
        try:
            files = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_directory)
            
            for file in files.get('Contents', []):
                file_name = file['Key']
                if file_name.lower().endswith(".wav") and file_name.startswith("Trimmed_"):
                    try:
                        # Download file into memory
                        file_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=file_name)
                        audio_data, sample_rate = librosa.load(BytesIO(file_obj['Body'].read()), sr=target_sr, mono=mono)

                        if not np.issubdtype(audio_data.dtype, np.floating):
                            audio_data = audio_data.astype(np.float32)

                        audio_file_dict[file_name] = {
                            "audio_data": audio_data,
                            "sample_rate": sample_rate
                        }
                        print(f"Loaded and processed {file_name}.")

                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")
                else:
                    print(f"Skipped {file_name} as it is not a trimmed WAV file.") 
            
        except Exception as e:
            print(f"Error listing files in S3 directory {s3_directory}: {e}")
        
        return audio_file_dict




    def detect_noise_level(audio, sample_rate, noise_threshold=0.01):
        rms_energy = np.sqrt(np.mean(audio**2)) 
        return rms_energy < noise_threshold 

    def noise_reduction(audio, sample_rate, noise_clip_duration=0.5): 
        try:
            noise_clip_samples = int(noise_clip_duration * sample_rate)
            noise_clip = audio[:noise_clip_samples]

            audio_stft = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
            noise_stft = librosa.stft(noise_clip, n_fft=2048, hop_length=512, win_length=2048)

            noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

            audio_mag = np.abs(audio_stft)
            audio_phase = np.angle(audio_stft)
            denoised_mag = np.maximum(audio_mag - noise_mag, 0)

            denoised_stft = denoised_mag * np.exp(1j * audio_phase)
            denoised_audio = librosa.istft(denoised_stft, hop_length=512, win_length=2048)

            return denoised_audio

        except Exception as e:
            print(f"Error in noise reduction: {e}")
            return audio



    def transcribe_audio(audio, sample_rate, segment_duration=30):
        """
        Transcribe audio robustly by splitting into manageable segments if needed.
        """
        try:
            model = whisper.load_model("large")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            return ""

        segment_samples = segment_duration * sample_rate
        segments = [audio[i * segment_samples:(i + 1) * segment_samples] for i in range(len(audio) // segment_samples + (len(audio) % segment_samples != 0))]

        full_transcript = []
        for segment in segments:
            try:
                result = model.transcribe(np.array(segment), language="en", beam_size=5, fp16=True)
                transcript = result.get('text', '').strip()
                full_transcript.append(transcript)
            except Exception as e:
                print(f"Error transcribing segment: {e}")

        return " ".join(full_transcript)



    def detect_repeated_words(transcript, min_repeats=2):
        try:
            normalized_text = re.sub(r'[^\w\s]', '', transcript.lower())
            words = normalized_text.split()

            repeated_words = []
            for i in range(len(words) - 1):
                if words[i] == words[i + 1]:
                    repeated_words.append(words[i])

            repeated_counts = Counter(repeated_words)

            filtered_repeats = {word: count for word, count in repeated_counts.items() if count >= min_repeats}

            return {
                "total_repeated_words": len(repeated_words),
                "repeated_words_counts": filtered_repeats
            }

        except Exception as e:
            print(f"Error detecting repeated words: {e}")
            return {
                "total_repeated_words": 0,
                "repeated_words_counts": {}
            }




    def extract_pauses_from_wav(wav_file_path, vad_mode=3, pause_threshold=0.5, frame_duration=10):
        try:
            # Open WAV file
            with wave.open(wav_file_path, 'rb') as wf:
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()

                # Validate audio format
                if num_channels != 1:
                    raise ValueError("Audio must be mono.")
                if sample_width != 2:
                    raise ValueError("Audio must be 16-bit PCM.")
                if sample_rate != 16000:
                    raise ValueError("Audio must have a sample rate of 16 kHz.")

                # Read frames and convert to int16 NumPy array
                audio = wf.readframes(wf.getnframes())
                audio = np.frombuffer(audio, dtype=np.int16)

            # Initialize WebRTC VAD
            vad = webrtcvad.Vad(vad_mode)

            # Compute frame size in samples
            frame_size = int(sample_rate * frame_duration / 1000)

            # Create frames
            frames = [
                audio[i:i + frame_size]
                for i in range(0, len(audio) - frame_size + 1, frame_size)
            ]
            if len(audio) % frame_size != 0:
                last_frame = audio[len(audio) - len(audio) % frame_size:]
                frames.append(np.pad(last_frame, (0, frame_size - len(last_frame))))

            # Process frames with VAD
            pause_durations = []
            start_silence = None
            for i, frame in enumerate(frames):
                is_speech = vad.is_speech(frame.tobytes(), sample_rate)
                if not is_speech:
                    if start_silence is None:
                        start_silence = i
                else:
                    if start_silence is not None:
                        pause_duration = (i - start_silence) * frame_duration / 1000
                        if pause_duration >= pause_threshold:
                            pause_durations.append(pause_duration)
                        start_silence = None

            # Calculate pause features
            total_pauses = len(pause_durations)
            avg_pause_duration = np.mean(pause_durations) if pause_durations else 0.0

            return {
                "total_pauses": total_pauses,
                "average_pause_duration": avg_pause_duration,
            }

        except Exception as e:
            print(f"Error extracting pauses: {e}")
            return {
                "total_pauses": 0,
                "average_pause_duration": 0.0,
            }



    def extract_pitch(audio_data, sample_rate):
        chunk_size = sample_rate * 10  # e.g., 10 seconds
        num_chunks = len(audio_data) // chunk_size + (len(audio_data) % chunk_size != 0)
        avg_pitches = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(audio_data))
            chunk = audio_data[start:end]
            pitches, _ = librosa.piptrack(y=chunk, sr=sample_rate)
            pitch_values = pitches[pitches > 0]
            pitch_values = pitch_values[(pitch_values > 50) & (pitch_values < 300)]
            avg_pitch = np.mean(pitch_values) if pitch_values.size > 0 else 0
            avg_pitches.append(avg_pitch)

        overall_avg_pitch = np.mean(avg_pitches) if avg_pitches else 0
        return overall_avg_pitch



    def calculate_pitch_variability(audio_data, sr, min_pitch=50, max_pitch=500):
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        refined_pitches = []

        for i in range(pitches.shape[1]):
            pitch_frame = pitches[:, i]
            mag_frame = magnitudes[:, i]
            if mag_frame.max() > 0:
                max_pitch_idx = mag_frame.argmax()
                pitch = pitch_frame[max_pitch_idx]
                if min_pitch <= pitch <= max_pitch:
                    refined_pitches.append(pitch)

        refined_pitches = np.array(refined_pitches)
        pitch_variability = np.std(refined_pitches) if refined_pitches.size > 0 else 0
        return pitch_variability


    def calculate_speech_rate(audio_data, sample_rate, min_duration=0.2):
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        if duration < min_duration:
            return 0

        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_data)
            words = result['text'].split()
            words_count = len(words)
            speech_rate = (words_count / duration) * 60
            return speech_rate
        except Exception as e:
            print(f"Error calculating speech rate: {e}")
            return 0


    def count_filler_words(transcript, filler_words):
        """Counts the total occurrences of filler words in a transcript."""
        if not transcript or not filler_words:
            return 0

        word_list = re.findall(r'\b\w+\b', transcript.lower())
        word_counts = Counter(word_list)
        total_fillers = sum(word_counts.get(word, 0) for word in filler_words)

        print(f"Total filler words found: {total_fillers}")
        return total_fillers

    fillers = [
        "um", "uh", "like", "you know", "so", "actually", "basically",
        "literally", "seriously", "honestly", "I mean", "well", "right",
        "kind of", "sort of", "I guess", "probably", "maybe", "for sure",
        "at the end of the day", "the thing is", "you see", "I think",
        "obviously", "as I said", "essentially",
        "anyway", "to be honest", "just", "okay", "anyhow", "oh"
    ] 



    def extract_features_from_audio_from_s3(s3_directory, fillers, target_sr=22050, mono=True):
        s3_client = boto3.client(
            's3',
            region_name=S3_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        
        audio_data = load_trimmed_audio_data_from_s3(s3_directory, target_sr, mono)  # Load from S3
        features_dict = {}
        

        for filename, data in audio_data.items():
            try:
                if 'audio_data' not in data or 'sample_rate' not in data:
                    print(f"Missing required keys in {filename}. Skipping...")
                    continue
                
                total_pauses = [] 
                average_pause_duration = []

                audio = data['audio_data']
                sample_rate = data['sample_rate']

                if detect_noise_level(audio, sample_rate):
                    audio = noise_reduction(audio, sample_rate)

                features = {"file_name": filename}
                transcript = transcribe_audio(audio, sample_rate)

                features["filler_words"] = count_filler_words(transcript, fillers)
                features["avg_pitch"] = extract_pitch(audio, sample_rate)
                features["pitch_variability"] = calculate_pitch_variability(audio, sample_rate)
                features["speech_rate"] = calculate_speech_rate(audio, sample_rate)
                repeated_words_features = detect_repeated_words(transcript)
                features["total_repeated_words"] = repeated_words_features["total_repeated_words"]
                results = extract_pauses_from_wav(audio) 
                total_pauses.append(results["total_pauses"])
                average_pause_duration.append(results["average_pause_duration"])

                file_label = os.path.splitext(filename)[0] 
                features["label"] = re.sub(r'\d+|\(.*?\)', '', file_label).strip()
                

                for key, value in features.items():
                    if key not in features_dict:
                        features_dict[key] = []
                    features_dict[key].append(value)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return features_dict




    def save_features_to_s3(features_dict, s3_directory):
        try:
            features_df = pd.DataFrame(features_dict)
            
            csv_buffer = io.StringIO()
            features_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            global audio_features_s3 
            audio_features_s3 = f"{s3_directory}/audio_features.csv"
            
            s3_client = boto3.client('s3')
            s3_client.put_object(Bucket=S3_BUCKET, Key=audio_features_s3, Body=csv_buffer.getvalue())
            
            print(f"Feature extraction complete. Data uploaded to {audio_features_s3}")
        except Exception as e:
            print(f"Failed to upload CSV file to S3: {e}")


    features_dict = extract_features_from_audio_from_s3(directory, fillers)
    save_features_to_s3(features_dict, directory) 




def retrain_model(features_csv):
    df = pd.read_csv(features_csv) 
    X = df[['filler_words','avg_pitch','pitch_variability','speech_rate','total_pauses','average_pause_duration','total_repeated_words']]  
    y = df['label'] 

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler() 
    X_Scaled = scaler.fit_transform(X) 

    scores = cross_val_score(model, X, y, cv=10) 

    global model
    model.fit(X_Scaled, y)
    joblib.dump(model, MODEL_PATH) 
    return "Model retrained successfully"


def predict(features_csv):
    global model
    predictions = model.predict(features_csv.reshape(1, -1)) 
    return {"prediction": predictions.tolist()}



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        files = request.json.get('file_paths') # define file_paths 
        if not files:
            return jsonify({"error": "No file paths provided"}), 400

        results = []
        for file_path in files:
            file_name = os.path.basename(file_path)
            local_path = secure_filename(file_name)
            s3_client.download_file(S3_BUCKET, file_path, local_path) 
            features_csv = preprocess_audio(local_path) 
            prediction = predict(features_csv)
            results.append({"File": file_path, "Prediction": prediction})
            os.remove(local_path) 
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/retrain', methods=['POST']) 
def retrain_endpoint():
    try:
        temp_dir = "temp_retrain_files"
        os.makedirs(temp_dir, exist_ok=True)

        files = request.json.get('file_paths')  # Assume these are S3 paths
        if not files:
            return jsonify({"error": "No file paths provided"}), 400

        for file_path in files:
            preprocess_audio(file_path)  

        # Define path to the generated features.csv 
        features_csv = audio_features_s3 
        s3_client.download_file(S3_BUCKET, "audio_features.csv", features_csv)
        retrain_message = retrain_model(features_csv)

        s3_client.delete_object(Bucket=S3_BUCKET, Key="audio_features.csv") 
        shutil.rmtree(temp_dir)

        return jsonify({"message": retrain_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to confirm the API is running.
    """
    return jsonify({"status": "OK"}) 



if __name__ == '__main__':
    app.run(debug=True)
