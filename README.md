# Speaker Diarization with Embeddings and Clustering

This repository provides a script for performing speaker diarization using embeddings and clustering. The script is designed to handle audio files in WAV and MP3 formats, convert them as needed, and perform diarization to separate and identify different speakers in the audio.

## Features

- Convert MP3 files to WAV format for processing
- Add silence to audio segments
- Combine mono audio channels into a single stereo file
- Merge short segments based on a specified threshold
- Post-process diarization segments
- Identify the main speaker in the audio
- Handle overlapping audio segments
- Extract embeddings for each segment using a pre-trained model
- Normalize and reduce the dimensionality of embeddings using PCA
- Cluster the embeddings to limit the number of speakers
- Create audio segments for each speaker with appropriate pauses
- Save the processed audio segments for each speaker

## Requirements

To use this script, you need the following Python packages:

- argparse
- os
- torch
- soundfile
- numpy
- pyannote.audio
- sklearn
- pydub

You can install these packages using pip:

```sh
pip install argparse os torch soundfile numpy pyannote.audio sklearn pydub
```

## Usage

To run the script, use the following command:

```sh
python diarization.py --audio_file <path_to_audio_file> --output_dir <output_directory> [--n_clusters <number_of_clusters>] [--short_segment_merge_threshold <threshold>]
```

### Arguments

- `--audio_file`: Path to the input audio file (wav or mp3).
- `--output_dir`: Directory to save the output audio segments.
- `--n_clusters`: (Optional) Number of speakers (default: 2).
- `--short_segment_merge_threshold`: (Optional) Threshold to merge short segments (default: 0.2).

### Example

```sh
python diarization.py --audio_file example.mp3 --output_dir output --n_clusters 2 --short_segment_merge_threshold 0.2
```

## Code Overview

### Convert MP3 to WAV

The `convert_mp3_to_wav` function converts an MP3 file to WAV format for processing.

### Add Silence

The `add_silence` function adds silence to the audio segments.

### Combine Audio Channels

The `combine_audio_channels` function combines mono audio channels into a single stereo file.

### Merge Short Segments

The `merge_short_segments` function merges short segments based on a specified threshold.

### Post-process Segments

The `post_process_segments` function processes the diarization segments to merge consecutive segments of the same speaker.

### Identify Main Speaker

The `identify_main_speaker` function identifies the main speaker based on the duration of their speech segments.

### Handle Overlapping Segments

The `handle_overlapping_segments` function identifies and handles overlapping audio segments.

### Speaker Diarization

The `speaker_diarization` function performs the entire diarization process, including extracting embeddings, normalizing, reducing dimensionality, and clustering.

### Main Function

The `main` function parses command-line arguments and calls the `speaker_diarization` function with the provided parameters.

## Contact

For any questions or issues, please contact Babacar Gningue at gningue.babacar.ita@gmail.com .

## License

This project is licensed under the MIT License.
