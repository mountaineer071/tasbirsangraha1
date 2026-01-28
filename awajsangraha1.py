import streamlit as st
import librosa
import numpy as np
import json
import base64
import io
import soundfile as sf
import tempfile
import os
import subprocess
import sys
import traceback
from typing import Dict, Any, Optional, Tuple
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# -----------------------------
# FFMPEG CHECK & UTILITIES
# -----------------------------
def check_ffmpeg_installed() -> bool:
    """Check if FFmpeg is available in the system."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def extract_audio_with_ffmpeg(video_path: str, audio_path: str) -> bool:
    """Extract audio from video using FFmpeg directly."""
    try:
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # WAV format
            '-ar', '44100',  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            audio_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            return True
        else:
            st.error(f"FFmpeg error: {result.stderr[:500]}")
            return False
    except Exception as e:
        st.error(f"FFmpeg extraction failed: {str(e)}")
        return False

# -----------------------------
# AUDIO FEATURE EXTRACTION
# -----------------------------
def compute_audio_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute comprehensive audio features for computer hearing applications"""
    features = {}
    
    # Time-domain features
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate'] = {
            'mean': float(np.mean(zcr)),
            'std': float(np.std(zcr)),
            'max': float(np.max(zcr)),
            'min': float(np.min(zcr))
        }
    except Exception as e:
        features['zero_crossing_rate'] = {'error': str(e)}
    
    try:
        rms = librosa.feature.rms(y=y)
        features['rms_energy'] = {
            'mean': float(np.mean(rms)),
            'std': float(np.std(rms)),
            'max': float(np.max(rms)),
            'min': float(np.min(rms))
        }
    except Exception as e:
        features['rms_energy'] = {'error': str(e)}
    
    # Spectral features
    spectral_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']
    for feat_name in spectral_features:
        try:
            if feat_name == 'spectral_centroid':
                values = librosa.feature.spectral_centroid(y=y, sr=sr)
            elif feat_name == 'spectral_bandwidth':
                values = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            else:  # spectral_rolloff
                values = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            features[feat_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'max': float(np.max(values)),
                'min': float(np.min(values))
            }
        except Exception as e:
            features[feat_name] = {'error': str(e)}
    
    # MFCCs (13 coefficients + delta + delta2)
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i+1}'] = {
                'mean': float(np.mean(mfccs[i])),
                'std': float(np.std(mfccs[i])),
                'max': float(np.max(mfccs[i])),
                'min': float(np.min(mfccs[i]))
            }
        
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(13):
            features[f'mfcc_delta_{i+1}'] = {
                'mean': float(np.mean(mfcc_delta[i])),
                'std': float(np.std(mfcc_delta[i]))
            }
        
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        for i in range(13):
            features[f'mfcc_delta2_{i+1}'] = {
                'mean': float(np.mean(mfcc_delta2[i])),
                'std': float(np.std(mfcc_delta2[i]))
            }
    except Exception as e:
        features['mfccs'] = {'error': str(e)}
    
    # Chroma features
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i+1}'] = {
                'mean': float(np.mean(chroma[i])),
                'std': float(np.std(chroma[i]))
            }
    except Exception as e:
        features['chroma'] = {'error': str(e)}
    
    # Spectral contrast
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(contrast.shape[0]):
            features[f'spectral_contrast_band_{i+1}'] = {
                'mean': float(np.mean(contrast[i])),
                'std': float(np.std(contrast[i]))
            }
    except Exception as e:
        features['spectral_contrast'] = {'error': str(e)}
    
    # Tonnetz
    try:
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        for i in range(6):
            features[f'tonnetz_{i+1}'] = {
                'mean': float(np.mean(tonnetz[i])),
                'std': float(np.std(tonnetz[i]))
            }
    except Exception as e:
        features['tonnetz'] = {'error': str(e)}
    
    # Additional features for computer hearing
    try:
        # Mel-spectrogram statistics
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spectrogram'] = {
            'mean': float(np.mean(mel_spec_db)),
            'std': float(np.std(mel_spec_db)),
            'max': float(np.max(mel_spec_db)),
            'min': float(np.min(mel_spec_db))
        }
        
        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features['spectral_flatness'] = {
            'mean': float(np.mean(flatness)),
            'std': float(np.std(flatness))
        }
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['onset_strength'] = {
            'mean': float(np.mean(onset_env)),
            'std': float(np.std(onset_env)),
            'max': float(np.max(onset_env))
        }
        
        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        features['beat_count'] = len(beat_frames)
        
    except Exception as e:
        features['additional_features_error'] = str(e)
    
    # Basic metadata
    features['basic_metadata'] = {
        'duration_sec': float(len(y) / sr),
        'sample_rate': int(sr),
        'num_samples': int(len(y)),
        'rms_total': float(np.sqrt(np.mean(y**2))),
        'peak_amplitude': float(np.max(np.abs(y))),
        'dynamic_range_db': float(20 * np.log10(np.max(np.abs(y)) / (np.std(y) + 1e-10)))
    }
    
    return features

def load_audio_file(file_path: str, is_video: bool = False) -> Tuple[Optional[np.ndarray], Optional[int], str]:
    """Load audio from file with robust error handling."""
    try:
        if is_video:
            # First try librosa (requires ffmpeg)
            try:
                y, sr = librosa.load(file_path, sr=None, mono=True)
                return y, sr, "success"
            except Exception as librosa_error:
                # If librosa fails, try extracting with ffmpeg first
                if check_ffmpeg_installed():
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                        tmp_audio_path = tmp_audio.name
                    
                    if extract_audio_with_ffmpeg(file_path, tmp_audio_path):
                        y, sr = librosa.load(tmp_audio_path, sr=None, mono=True)
                        os.unlink(tmp_audio_path)
                        return y, sr, "success_via_ffmpeg"
                    else:
                        os.unlink(tmp_audio_path) if os.path.exists(tmp_audio_path) else None
                        return None, None, f"FFmpeg extraction failed: {librosa_error}"
                else:
                    return None, None, f"FFmpeg not installed and librosa failed: {librosa_error}"
        else:
            # Audio file
            y, sr = librosa.load(file_path, sr=None, mono=True)
            return y, sr, "success"
    except Exception as e:
        return None, None, f"Error loading file: {str(e)}"

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(
    page_title="Audio Characteristics Extractor",
    page_icon="🔊",
    layout="wide"
)

st.title("🔊 Audio Characteristics Extractor for Computer Hearing")
st.markdown("""
Upload an MP4 video or MP3 audio file, select a time segment, and extract:
- **Audio segment** (WAV format embedded in output)
- **Comprehensive acoustic features** (MFCCs, spectral, temporal, chroma, etc.)
- **User-provided description**
- **Metadata** (timestamps, duration, sample rate)

Output is a single JSON file containing everything needed for computer hearing applications.
""")

# Check FFmpeg installation
ffmpeg_available = check_ffmpeg_installed()
if not ffmpeg_available:
    st.warning("""
    ⚠️ **FFmpeg not detected!** 
    
    FFmpeg is required for processing video files (MP4). Audio files (MP3, WAV) should still work.
    
    **To install FFmpeg:**
    - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)
    - **Mac:** `brew install ffmpeg`
    - **Linux:** `sudo apt-get install ffmpeg` or `sudo yum install ffmpeg`
    
    If running in a cloud environment, check the platform's documentation for installing FFmpeg.
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Upload MP4 video or MP3 audio file",
    type=["mp4", "mp3", "wav", "mpeg", "m4a", "flac", "ogg"],
    help="Supports MP4 (video), MP3, WAV, M4A, FLAC, and OGG files"
)

if uploaded_file is not None:
    # Display file info
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📁 **File:** {uploaded_file.name} ({file_size_mb:.2f} MB)")
    
    # Check file size limit
    if file_size_mb > 200:
        st.error("❌ File size exceeds 200MB limit. Please upload a smaller file.")
    else:
        # Create temporary file
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Determine file type
            is_video = suffix in ['.mp4', '.mpeg', '.mov', '.avi', '.mkv']
            file_type = "video" if is_video else "audio"
            
            if is_video:
                st.info("🎥 Processing video file... Extracting audio track")
                if not ffmpeg_available:
                    st.warning("FFmpeg not found. Trying alternative methods...")
            else:
                st.info("🔊 Processing audio file...")
            
            # Load audio
            with st.spinner(f"Loading {'video' if is_video else 'audio'} file..."):
                y, sr, load_status = load_audio_file(tmp_file_path, is_video)
            
            if y is None or sr is None:
                st.error(f"❌ Failed to process file: {load_status}")
                
                # Enhanced troubleshooting
                with st.expander("🔧 Detailed Troubleshooting"):
                    st.markdown("""
                    ### Common Issues & Solutions:
                    
                    1. **FFmpeg not installed for video files:**
                       - Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
                       - Restart the application after installation
                    
                    2. **Corrupted or incompatible file:**
                       - Try converting the file to WAV or MP3 using a tool like Audacity or VLC
                       - Ensure the file isn't corrupted by playing it in another media player
                    
                    3. **Unsupported codec:**
                       - Convert to a standard format: `ffmpeg -i input.mp4 -c:a pcm_s16le output.wav`
                       - Use online converters if FFmpeg isn't available
                    
                    4. **Large file size:**
                       - Extract a shorter segment using video editing software
                       - Compress the audio: `ffmpeg -i input.mp3 -b:a 128k output.mp3`
                    
                    5. **Permission issues:**
                       - Ensure you have read permissions for the file
                       - Try moving the file to a different directory
                    """)
                    
                    if is_video and not ffmpeg_available:
                        st.code("""
                        # Quick FFmpeg installation commands:
                        
                        # Ubuntu/Debian:
                        sudo apt-get update
                        sudo apt-get install ffmpeg
                        
                        # MacOS (with Homebrew):
                        brew install ffmpeg
                        
                        # Windows (with Chocolatey):
                        choco install ffmpeg
                        
                        # Or download from: https://ffmpeg.org/download.html
                        """)
            else:
                total_duration = len(y) / sr
                
                # Time selection UI
                st.subheader("⏱️ Select Time Segment")
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                
                with col1:
                    t1 = st.number_input(
                        "Start time (seconds)",
                        min_value=0.0,
                        max_value=total_duration,
                        value=0.0,
                        step=0.1,
                        format="%.2f"
                    )
                with col2:
                    t2 = st.number_input(
                        "End time (seconds)",
                        min_value=t1 + 0.1,
                        max_value=total_duration,
                        value=min(t1 + 3.0, total_duration),
                        step=0.1,
                        format="%.2f"
                    )
                with col3:
                    st.metric("Duration", f"{t2 - t1:.2f}s")
                with col4:
                    if st.button("🎵 Full Audio"):
                        t1 = 0.0
                        t2 = total_duration
                        st.rerun()
                
                # Duration validation
                if t2 <= t1:
                    st.error("❌ End time must be greater than start time")
                else:
                    segment_duration = t2 - t1
                    
                    # Warn for long segments
                    if segment_duration > 60:
                        st.warning("⚠️ Segment longer than 60 seconds. Consider shorter segments (1-10s) for computer hearing tasks.")
                    elif segment_duration > 30:
                        st.info("💡 Segment is 30-60 seconds. This is acceptable but may create larger files.")
                    
                    # Extract segment
                    start_sample = int(t1 * sr)
                    end_sample = int(t2 * sr)
                    y_segment = y[start_sample:end_sample]
                    
                    # Preview audio segment
                    st.subheader("🔊 Segment Preview")
                    col_preview1, col_preview2 = st.columns([3, 1])
                    
                    with col_preview1:
                        preview_buffer = io.BytesIO()
                        sf.write(preview_buffer, y_segment, sr, format='WAV')
                        preview_buffer.seek(0)
                        st.audio(preview_buffer, format='audio/wav', start_time=0)
                    
                    with col_preview2:
                        st.metric("Sample Rate", f"{sr} Hz")
                        st.metric("Samples", f"{len(y_segment):,}")
                    
                    # Description input
                    st.subheader("📝 Segment Description")
                    description = st.text_area(
                        "Describe this audio segment (for computer hearing context)",
                        placeholder="e.g., 'Dog barking with car passing background', 'Piano melody in C major', 'Speech with emotional emphasis'",
                        height=100,
                        help="This description will be included in the metadata for ML training datasets"
                    )
                    
                    # Advanced options
                    with st.expander("⚙️ Advanced Options"):
                        col_adv1, col_adv2 = st.columns(2)
                        with col_adv1:
                            include_raw_audio = st.checkbox(
                                "Include raw audio in JSON",
                                value=True,
                                help="Embed the WAV audio as base64 in the JSON output"
                            )
                            normalize_audio = st.checkbox(
                                "Normalize audio segment",
                                value=True,
                                help="Normalize audio to prevent clipping"
                            )
                        with col_adv2:
                            compute_detailed_stats = st.checkbox(
                                "Detailed statistics",
                                value=True,
                                help="Compute min/max/median for all features"
                            )
                            segment_format = st.selectbox(
                                "Segment format",
                                ["WAV", "MP3"],
                                help="Format for embedded audio"
                            )
                    
                    # Process button
                    if st.button("✨ Extract Features & Prepare Download", type="primary"):
                        with st.spinner("Computing audio characteristics..."):
                            # Normalize if requested
                            if normalize_audio and len(y_segment) > 0:
                                max_val = np.max(np.abs(y_segment))
                                if max_val > 0:
                                    y_segment = y_segment / max_val * 0.95
                            
                            # Compute features
                            features = compute_audio_features(y_segment, sr)
                            
                            # Prepare audio buffer
                            if include_raw_audio:
                                audio_buffer = io.BytesIO()
                                if segment_format == "WAV":
                                    sf.write(audio_buffer, y_segment, sr, format='WAV')
                                    audio_mime = "audio/wav"
                                else:  # MP3
                                    # Note: MP3 writing requires pydub or similar
                                    sf.write(audio_buffer, y_segment, sr, format='MP3')
                                    audio_mime = "audio/mpeg"
                                
                                audio_buffer.seek(0)
                                audio_b64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
                            else:
                                audio_b64 = None
                                audio_mime = None
                            
                            # Prepare output structure
                            output_data = {
                                "metadata": {
                                    "original_filename": uploaded_file.name,
                                    "file_type": file_type,
                                    "file_size_bytes": uploaded_file.size,
                                    "segment_start_sec": t1,
                                    "segment_end_sec": t2,
                                    "segment_duration_sec": t2 - t1,
                                    "sample_rate_hz": sr,
                                    "num_samples": len(y_segment),
                                    "bit_depth": 16,  # Assuming 16-bit from librosa
                                    "normalized": normalize_audio,
                                    "description": description.strip() if description.strip() else "No description provided",
                                    "feature_extraction_library": "librosa",
                                    "extraction_timestamp": st.session_state.get('run_time', 'N/A'),
                                    "processing_notes": load_status
                                },
                                "audio_characteristics": features,
                                "computer_hearing_notes": {
                                    "recommended_usage": [
                                        "MFCCs and deltas: Speaker identification, speech recognition",
                                        "Spectral features: Music genre classification, sound event detection",
                                        "Chroma features: Music key detection, harmonic analysis",
                                        "Tonnetz: Tonal similarity tasks",
                                        "Onset strength: Rhythm analysis, beat tracking",
                                        "Tempo: Music tempo estimation, activity recognition"
                                    ],
                                    "normalization_required": True,
                                    "typical_preprocessing": [
                                        "Standardize features (zero mean, unit variance) per dataset",
                                        "Log-scaling for spectral features",
                                        "Delta and delta-delta features for temporal dynamics",
                                        "Feature concatenation for neural network input"
                                    ],
                                    "common_applications": [
                                        "Audio classification",
                                        "Sound event detection",
                                        "Music information retrieval",
                                        "Speech emotion recognition",
                                        "Environmental sound analysis"
                                    ]
                                }
                            }
                            
                            if include_raw_audio and audio_b64:
                                output_data["audio_segment"] = {
                                    "format": segment_format.lower(),
                                    "encoding": "base64",
                                    "mime_type": audio_mime,
                                    "size_bytes": len(audio_b64) * 3 // 4,  # Approximate base64 to bytes
                                    "data": audio_b64
                                }
                        
                        # Filename input
                        st.subheader("💾 Download Output")
                        default_name = f"audio_features_{Path(uploaded_file.name).stem}_{int(t1)}s_{int(t2)}s"
                        output_name = st.text_input(
                            "Enter output filename (without extension)",
                            value=default_name,
                            help="Your file will be saved as [name].json"
                        ).strip()
                        
                        if not output_name:
                            output_name = default_name
                        
                        # Create download button
                        json_str = json.dumps(output_data, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download Audio Characteristics Package",
                            data=json_str,
                            file_name=f"{output_name}.json",
                            mime="application/json",
                            help="Contains embedded audio segment + features + metadata"
                        )
                        
                        # Display summary
                        st.subheader("📊 Extraction Summary")
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        
                        with col_sum1:
                            st.metric("Total Features", len(features))
                            st.metric("Duration", f"{t2-t1:.2f}s")
                        with col_sum2:
                            st.metric("MFCC Features", 39)  # 13 + 13 + 13
                            st.metric("Sample Rate", f"{sr} Hz")
                        with col_sum3:
                            st.metric("Spectral Features", 8)
                            st.metric("Tempo", f"{features.get('tempo', 0):.1f} BPM")
                        with col_sum4:
                            st.metric("Chroma Features", 12)
                            st.metric("File Size", f"{len(json_str) / 1024:.1f} KB")
                        
                        # Feature explorer
                        with st.expander("🔍 Explore Features"):
                            feature_categories = {
                                "MFCC Features": [f for f in features.keys() if f.startswith('mfcc')],
                                "Spectral Features": [f for f in features.keys() if f.startswith('spectral') and not f.startswith('spectral_contrast')],
                                "Chroma Features": [f for f in features.keys() if f.startswith('chroma')],
                                "Temporal Features": ['zero_crossing_rate', 'rms_energy', 'onset_strength', 'tempo'],
                                "Other Features": [f for f in features.keys() if f not in ['basic_metadata'] and not any(f.startswith(prefix) for prefix in ['mfcc', 'spectral', 'chroma'])]
                            }
                            
                            selected_category = st.selectbox(
                                "Select feature category",
                                list(feature_categories.keys())
                            )
                            
                            if selected_category:
                                for feat in feature_categories[selected_category]:
                                    if feat in features and isinstance(features[feat], dict):
                                        st.write(f"**{feat}**:")
                                        st.json(features[feat])
                        
                        st.success(f"✅ Feature extraction complete! Ready to download.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            
            # Show detailed error for debugging
            with st.expander("🔧 Show Technical Details"):
                st.code(traceback.format_exc())
            
            st.info("""
            **Troubleshooting tips:**
            1. **FFmpeg Installation**: For video files, ensure FFmpeg is installed and in your PATH
            2. **File Format**: Ensure file is a valid MP4/MP3/WAV file
            3. **File Corruption**: Try playing the file in another media player
            4. **Large Files**: Try shorter segments (1-10 seconds)
            5. **Alternative Method**: Convert video to WAV first using external tools
            """)
        
        finally:
            # Cleanup temp files
            try:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
            except:
                pass

# Footer with installation instructions
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <h4>🛠️ Installation & Setup</h4>
    <p><strong>For Video Processing (MP4 files):</strong></p>
    <code>pip install streamlit librosa numpy soundfile pydub</code><br/>
    <code># Plus install FFmpeg system-wide</code>
    
    <p>💡 <strong>Computer Hearing Tip:</strong> For ML pipelines, use consistent segment lengths (1-3 seconds). 
    Normalize features across your dataset before training models.</p>
    <p>🔒 <strong>Privacy:</strong> All processing happens locally in your browser/server</p>
</div>
""", unsafe_allow_html=True)

# Add runtime info to session state
if 'run_time' not in st.session_state:
    import datetime
    st.session_state.run_time = datetime.datetime.now().isoformat()
