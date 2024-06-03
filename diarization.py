import argparse
import os
import torch
import soundfile as sf
import numpy as np
from pyannote.audio import Pipeline, Model, Inference
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pydub import AudioSegment

# Load HF Token from environment variable
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set")

# Fonction pour convertir MP3 en WAV
def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")
    return wav_file

# Fonction pour ajouter des silences
def add_silence(duration, samplerate):
    return np.zeros(int(duration * samplerate), dtype=np.float32)

def combine_audio_channels(audio_file1, audio_file2, output_dir, original_filename):
    audio1 = AudioSegment.from_wav(audio_file1)
    audio2 = AudioSegment.from_wav(audio_file2)
    combined_audio = AudioSegment.from_mono_audiosegments(audio1, audio2)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{original_filename.split('.')[0]}.mp3")
    combined_audio.export(output_file, format="mp3")
    print(f"Saved combined audio to {output_file}")

def merge_short_segments(segments, threshold):
    merged_segments = []
    for i, segment in enumerate(segments):
        if i == 0:
            merged_segments.append(segment)
        else:
            prev_segment = merged_segments[-1]
            if segment[0] - prev_segment[1] < threshold and prev_segment[2] == segment[2]:
                merged_segments[-1] = (prev_segment[0], segment[1], prev_segment[2])
            else:
                merged_segments.append(segment)
    return merged_segments

def post_process_segments(diarization):
    processed_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if len(processed_segments) > 0 and processed_segments[-1][2] == speaker:
            processed_segments[-1] = (processed_segments[-1][0], turn.end, speaker)
        else:
            processed_segments.append((turn.start, turn.end, speaker))
    return processed_segments

def identify_main_speaker(processed_segments):
    speaker_durations = {}
    for start_time, end_time, speaker in processed_segments:
        if speaker not in speaker_durations:
            speaker_durations[speaker] = 0
        speaker_durations[speaker] += end_time - start_time
    main_speaker = max(speaker_durations, key=speaker_durations.get)
    return main_speaker

def handle_overlapping_segments(segments, samplerate):
    overlapping_segments = []
    for i, (start_time1, end_time1, speaker1) in enumerate(segments):
        for j, (start_time2, end_time2, speaker2) in enumerate(segments):
            if i != j and start_time1 < end_time2 and end_time1 > start_time2:
                overlap_start = max(start_time1, start_time2)
                overlap_end = min(end_time1, end_time2)
                if (overlap_end - overlap_start) > 0:
                    overlapping_segments.append((overlap_start, overlap_end))
    return overlapping_segments

def speaker_diarization(audio_path, output_dir, n_clusters=2, short_segment_merge_threshold=0.2):
    device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if audio_path.endswith('.mp3'):
        wav_file = audio_path.replace('.mp3', '.wav')
        convert_mp3_to_wav(audio_path, wav_file)
        audio_path = wav_file

    try:
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        diarization_pipeline.to(device)
        diarization_pipeline.segmentation.min_duration = 0.5
        print("Loaded diarization pipeline successfully.")
    except Exception as e:
        print(f"Error loading the diarization pipeline: {e}")
        return

    audio_data, samplerate = sf.read(audio_path)
    audio_data = audio_data.astype(np.float32)
    total_duration = len(audio_data) / samplerate
    print(f"Loaded audio file successfully with samplerate: {samplerate}")

    try:
        diarization = diarization_pipeline({"uri": audio_path, "audio": audio_path})
        print("Performed diarization successfully.")
    except Exception as e:
        print(f"Error during diarization: {e}")
        return

    processed_segments = post_process_segments(diarization)
    processed_segments = merge_short_segments(processed_segments, short_segment_merge_threshold)
    print("Post-processed segments successfully.")
    print(f"Processed segments: {processed_segments}")

    main_speaker = identify_main_speaker(processed_segments)
    print(f"Identified main speaker: {main_speaker}")

    try:
        embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        embedding_inference = Inference(embedding_model, window="sliding", duration=0.5, step=0.25).to(device)
        print("Loaded embedding model successfully.")
    except Exception as e:
        print(f"Error loading the embedding model: {e}")
        return

    embeddings = []
    non_main_speaker_segments = []

    for start_time, end_time, speaker in processed_segments:
        start = int(start_time * samplerate)
        end = int(end_time * samplerate)
        segment_audio = audio_data[start:end]

        if speaker != main_speaker:
            non_main_speaker_segments.append((start_time, end_time, segment_audio))
            try:
                segment_embedding = embedding_inference({'waveform': torch.tensor(segment_audio).unsqueeze(0).to(device), 'sample_rate': samplerate})
                segment_embedding_mean = np.mean(segment_embedding.data, axis=0)
                embeddings.append(segment_embedding_mean)
                print(f"Processed segment [{start_time} --> {end_time}] successfully.")
            except Exception as e:
                print(f"Error processing segment [{start_time} --> {end_time}]: {e}")
                return

    if not embeddings:
        print("No embeddings extracted. Please check the audio file and diarization results.")
        return

    embeddings = np.vstack(embeddings)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    # Adjust n_components based on the number of samples and features
    n_components = min(50, min(embeddings.shape))
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    print("Reduced embeddings dimensionality with PCA successfully.")
    
    clustering = KMeans(n_clusters=n_clusters - 1, random_state=42)
    labels = clustering.fit_predict(reduced_embeddings)
    print(f"Clustering results: {labels}")

    speaker_segments = {main_speaker: []}
    for i in range(n_clusters - 1):
        speaker_segments[i] = []

    for i, (start_time, end_time, segment_audio) in enumerate(non_main_speaker_segments):
        label = labels[i]
        speaker_segments[label].append((start_time, end_time, segment_audio))

    for start_time, end_time, speaker in processed_segments:
        if speaker == main_speaker:
            start = int(start_time * samplerate)
            end = int(end_time * samplerate)
            segment_audio = audio_data[start:end]
            speaker_segments[main_speaker].append((start_time, end_time, segment_audio))

    print(f"Speaker segments before merging silence: {speaker_segments}")

    def add_silence(duration_samples, samplerate):
        return np.zeros(duration_samples, dtype=np.float32)

    for label in speaker_segments.keys():
        total_duration_samples = int(total_duration * samplerate)
        total_speaker_audio = np.zeros(total_duration_samples, dtype=np.float32)
        current_time_samples = 0

        for start_time, end_time, segment_audio in speaker_segments[label]:
            start_sample = int(start_time * samplerate)
            end_sample = int(end_time * samplerate)
            segment_duration_samples = end_sample - start_sample

            silence_duration_samples = start_sample - current_time_samples
            if silence_duration_samples > 0:
                silence_audio = add_silence(silence_duration_samples, samplerate)
                total_speaker_audio[current_time_samples:start_sample] = silence_audio

            total_speaker_audio[start_sample:end_sample] = segment_audio
            current_time_samples = end_sample

        remaining_samples = total_duration_samples - current_time_samples
        if remaining_samples > 0:
            silence_audio = add_silence(remaining_samples, samplerate)
            total_speaker_audio[current_time_samples:] = silence_audio

        speaker_segments[label] = total_speaker_audio

    print(f"Speaker segments after merging silence: {speaker_segments}")

    for speaker, audio in speaker_segments.items():
        output_file = os.path.join(output_dir, f"speaker_{speaker}.wav")
        sf.write(output_file, audio, samplerate)
        print(f"Saved speaker {speaker} audio to {output_file}")
    
    original_filename = os.path.basename(audio_path)
    combine_audio_channels(os.path.join(output_dir, f"speaker_{main_speaker}.wav"), os.path.join(output_dir, f"speaker_0.wav"), output_dir, original_filename)

def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization with embeddings and clustering")
    parser.add_argument("--audio_file", type=str, required=True, help="Path to the input audio file (wav or mp3)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output audio segments")
    parser.add_argument("--n_clusters", type=int, default=2, help="Number of speakers (default: 2)")
    parser.add_argument("--short_segment_merge_threshold", type=float, default=0.2, help="Threshold to merge short segments (default: 0.2)")

    args = parser.parse_args()
    
    audio_file = args.audio_file
    output_dir = args.output_dir
    n_clusters = args.n_clusters
    short_segment_merge_threshold = args.short_segment_merge_threshold

    speaker_diarization(audio_file, output_dir, n_clusters, short_segment_merge_threshold)

if __name__ == "__main__":
    main()
