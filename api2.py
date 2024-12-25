import os
import re
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

MODEL_PATH = r'C:\Users\Technologist\OneDrive - Higher Education Commission\Job Project\Speech Confidence\model_pkl_V2.pkl' # replace the path if required.  
model = joblib.load(MODEL_PATH) 

def preprocess_audio(directory_path):

    def trim_audio(input_file, output_file, duration_limit=180):
        audio = AudioSegment.from_file(input_file)
        duration = len(audio) / 1000  
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(os.path.dirname(output_file), f"Trimmed_{base_name}.wav")

        if duration > duration_limit:
            trimmed_audio = audio[duration_limit * 1000:]  # Trim first 180 seconds
            trimmed_audio.export(output_file, format="wav")
            print(f"Trimmed {input_file}.") 
        else:
            audio.export(output_file, format="wav")  
            print(f"{input_file} is shorter than {duration_limit} seconds. Format changed to WAV.")


    def process_directory(input_directory, output_directory, duration_limit=180):
        os.makedirs(output_directory, exist_ok=True) 

        for filename in os.listdir(input_directory):
            if filename.endswith(".mp3"):
                input_file = os.path.join(input_directory, filename)
                output_file = os.path.join(output_directory, filename)  # Save in new folder
                trim_audio(input_file, output_file, duration_limit)

    directory = r"C:\Users\Technologist\OneDrive - Higher Education Commission\Job Project\Speech Confidence\directory" 
    trimmed_output_directory = r"C:\Users\Technologist\OneDrive - Higher Education Commission\Job Project\Speech Confidence\directory\Trimmed"
    process_directory(directory, trimmed_output_directory)


    def load_audio_data(directory, target_sr=22050, mono=True):
        audio_file_dict = {} 
        for filename in os.listdir(directory):
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(directory, filename)
                try:
                    # Load the audio file with librosa, specifying resampling rate and mono conversion
                    audio_data, sample_rate = librosa.load(file_path, sr=target_sr, mono=mono)

                    # Ensure audio data is in the correct format, i.e., floating point
                    if not np.issubdtype(audio_data.dtype, np.floating):
                        audio_data = audio_data.astype(np.float32)

                    audio_file_dict[filename] = {
                        "audio_data": audio_data,
                        "sample_rate": sample_rate
                    }
                    print(f"Loaded and processed {filename}: {len(audio_data)} samples at {sample_rate} Hz, mono={mono}")

                except Exception as e:
                    print(f"Error loading or processing {filename}: {e}")
            else:
                print(f"Skipped {filename} as it is not a WAV file.")
        return audio_file_dict

    def detect_noise_level(audio, sample_rate, noise_threshold=0.01):
        rms_energy = np.sqrt(np.mean(audio**2))
        return rms_energy < noise_threshold 
    
    def reduce_noise(audio, sample_rate, noise_clip_duration=0.5):
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

            # Filter by minimum repeats
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
        

  
    def extract_pauses_from_wav(input_path, vad_mode=3, pause_threshold=0.5, frame_duration=10):
        temp_wav_path = "temp_preprocessed_audio.wav" 

        try:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and 16kHz
            audio.export(temp_wav_path, format="wav")

            with wave.open(temp_wav_path, 'rb') as wf:
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

                audio = wf.readframes(wf.getnframes())
                audio = np.frombuffer(audio, dtype=np.int16)

            vad = webrtcvad.Vad(vad_mode)

            frame_size = int(sample_rate * frame_duration / 1000)
            frames = [
                audio[i:i + frame_size]
                for i in range(0, len(audio) - frame_size + 1, frame_size)
            ]
            if len(audio) % frame_size != 0:
                last_frame = audio[len(audio) - len(audio) % frame_size:]
                frames.append(np.pad(last_frame, (0, frame_size - len(last_frame))))

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

            total_pauses = len(pause_durations)
            avg_pause_duration = np.mean(pause_durations) if pause_durations else 0.0

            return {
                "total_pauses": total_pauses,
                "pause_durations": pause_durations,
                "average_pause_duration": avg_pause_duration,
            }

        except Exception as e:
            print(f"Error extracting pauses: {e}")
            return {
                "total_pauses": 0,
                "pause_durations": [],
                "average_pause_duration": 0.0,
            }

        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
                print(f"Deleted temporary file: {temp_wav_path}")



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


    def extract_features_from_audio(directory, fillers):
        audio_data = load_audio_data(directory)  
        features_dict = {}

        for filename, data in audio_data.items():
            try:
                if 'audio_data' not in data or 'sample_rate' not in data:
                    print(f"Missing required keys in {filename}. Skipping.")
                    continue

                audio = data['audio_data']
                sample_rate = data['sample_rate']

                # Apply Noise Reduction
                if detect_noise_level(audio, sample_rate):
                    audio = reduce_noise(audio, sample_rate)

                features = {"file_name": filename}

                transcript = transcribe_audio(audio, sample_rate)
                features["filler_words"] = count_filler_words(transcript, fillers)
                features["avg_pitch"] = extract_pitch(audio, sample_rate)
                features["pitch_variability"] = calculate_pitch_variability(audio, sample_rate)
                features["speech_rate"] = calculate_speech_rate(audio, sample_rate)
                repeated_words_features = detect_repeated_words(transcript)
                features["total_repeated_words"] = repeated_words_features["total_repeated_words"]

                file_label = os.path.splitext(filename)[0]
                features["label"] = re.sub(r'\d+|\(.*?\)', '', file_label).strip()

                # Pause extraction
                results = extract_pauses_from_wav(audio)
                features["total_pauses"] = results["total_pauses"]
                features["average_pause_duration"] = results["average_pause_duration"]

                # Collect features
                for key, value in features.items():
                    if key not in features_dict:
                        features_dict[key] = []
                    features_dict[key].append(value)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return features_dict
    
    def save_features_to_csv(features_dict, directory):
        try:
            features_df = pd.DataFrame(features_dict)
            output_file = os.path.join(directory, "audio_features_3.csv")
            csv_features = features_df.to_csv(output_file, index=False)
            print(f"Feature extraction complete. Data saved to {output_file}")
        except Exception as e:
            print(f"Failed to save CSV file: {e}")

        return csv_features

    main_directory = r"C:\Users\Technologist\OneDrive - Higher Education Commission\Job Project\Speech Confidence\directory\Trimmed"
    features_dict = extract_features_from_audio(main_directory, fillers) 
    save_features_to_csv(features_dict, directory)
    



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




@app.route('/predict', methods=['POST'])  
def predict_endpoint():
    try:
        files = request.json.get('file_paths') 
        if not files:
            return jsonify({"error": "No file paths provided"}), 400

        predictions = []

        for file_path in files:
            features_csv = preprocess_audio(file_path)
            if not os.path.exists(features_csv):
                return jsonify({"error": "Failed to process audio file"}), 400

            df = pd.read_csv(features_csv) 
            X = df[['filler_words','avg_pitch','pitch_variability','speech_rate','total_pauses','average_pause_duration','total_repeated_words']]

            prediction = model.predict(X) 

            label_encoder = LabelEncoder() 
            predicted_label = label_encoder.inverse_transform(prediction) 
            predictions.append({"file": file_path, "predicted_label": predicted_label[0]})

            os.remove(features_csv)

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500







@app.route('/retrain', methods=['POST'])
def retrain_endpoint():
    try:
        temp_dir = "temp_retrain_files"
        os.makedirs(temp_dir, exist_ok=True)

        files = request.json.get('file_paths')  # Assume these are local file paths
        if not files:
            return jsonify({"error": "No file paths provided"}), 400

        all_features_csv = []

        for file_path in files:
            features_csv = preprocess_audio(file_path)  
            all_features_csv.append(features_csv)

        if not all_features_csv:
            return jsonify({"error": "No feature CSVs generated"}), 400

        combined_df = pd.DataFrame()

        for features_csv in all_features_csv:
            temp_df = pd.read_csv(features_csv)
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
            os.remove(features_csv)  

        combined_csv_path = os.path.join(temp_dir, "audio_features.csv")
        combined_df.to_csv(combined_csv_path, index=False) 

        # Retrain the model using the CSV
        retrain_message = retrain_model(combined_csv_path)

        shutil.rmtree(temp_dir)  

        return jsonify({"message": retrain_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True)


    