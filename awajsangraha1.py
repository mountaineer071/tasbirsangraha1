import streamlit as st
import librosa
import numpy as np
import json
import base64
import io
import soundfile as sf
import tempfile
import os
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# AUDIO FEATURE EXTRACTION
# -----------------------------
def compute_audio_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute comprehensive audio features for computer hearing applications"""
    features = {}
    
    # Time-domain features
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate'] = {
        'mean': float(np.mean(zcr)),
        'std': float(np.std(zcr))
    }
    
    rms = librosa.feature.rms(y=y)
    features['rms_energy'] = {
        'mean': float(np.mean(rms)),
        'std': float(np.std(rms))
    }
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = {
        'mean': float(np.mean(spectral_centroids)),
        'std': float(np.std(spectral_centroids))
    }
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = {
        'mean': float(np.mean(spectral_bandwidth)),
        'std': float(np.std(spectral_bandwidth))
    }
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff'] = {
        'mean': float(np.mean(spectral_rolloff)),
        'std': float(np.std(spectral_rolloff))
    }
    
    # MFCCs (13 coefficients + delta + delta2)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}'] = {
            'mean': float(np.mean(mfccs[i])),
            'std': float(np.std(mfccs[i]))
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
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f'chroma_{i+1}'] = {
            'mean': float(np.mean(chroma[i])),
            'std': float(np.std(chroma[i]))
        }
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        features[f'spectral_contrast_band_{i+1}'] = {
            'mean': float(np.mean(contrast[i])),
            'std': float(np.std(contrast[i]))
        }
    
    # Tonnetz (tonal centroid features)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    for i in range(6):
        features[f'tonnetz_{i+1}'] = {
            'mean': float(np.mean(tonnetz[i])),
            'std': float(np.std(tonnetz[i]))
        }
    
    # Temporal features
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features['onset_strength'] = {
        'mean': float(np.mean(onset_env)),
        'std': float(np.std(onset_env))
    }
    
    # Basic metadata
    features['basic_metadata'] = {
        'duration_sec': float(len(y) / sr),
        'sample_rate': int(sr),
        'num_samples': int(len(y)),
        'rms_total': float(np.sqrt(np.mean(y**2)))
    }
    
    return features

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

# File uploader
uploaded_file = st.file_uploader(
    "Upload MP4 video or MP3 audio file",
    type=["mp4", "mp3", "wav"],
    help="Supports MP4 (video), MP3 and WAV (audio) files"
)

if uploaded_file is not None:
    # Create temporary file
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Process file based on type
        if suffix == ".mp4":
            st.info("🎥 Processing video file... Extracting audio track")
            # Use librosa to load audio directly from video (requires ffmpeg)
            y, sr = librosa.load(tmp_file_path, sr=None, mono=True)
            file_type = "video"
        else:
            st.info("🔊 Processing audio file...")
            y, sr = librosa.load(tmp_file_path, sr=None, mono=True)
            file_type = "audio"
        
        total_duration = len(y) / sr
        
        # Time selection UI
        st.subheader("⏱️ Select Time Segment")
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            t1 = st.number_input(
                "Start time (seconds)",
                min_value=0.0,
                max_value=total_duration,
                value=0.0,
                step=0.1,
                format="%.1f"
            )
        with col2:
            t2 = st.number_input(
                "End time (seconds)",
                min_value=t1 + 0.1,
                max_value=total_duration,
                value=min(t1 + 3.0, total_duration),
                step=0.1,
                format="%.1f"
            )
        with col3:
            st.write("**Duration:**")
            st.write(f"{t2 - t1:.1f} seconds")
        
        if t2 <= t1:
            st.error("❌ End time must be greater than start time")
        elif (t2 - t1) > 30:
            st.warning("⚠️ Segment longer than 30 seconds may create large output files. Consider shorter segments for computer hearing tasks.")
        else:
            # Extract segment
            start_sample = int(t1 * sr)
            end_sample = int(t2 * sr)
            y_segment = y[start_sample:end_sample]
            
            # Preview audio segment
            st.subheader("🔊 Segment Preview")
            preview_buffer = io.BytesIO()
            sf.write(preview_buffer, y_segment, sr, format='WAV')
            preview_buffer.seek(0)
            st.audio(preview_buffer, format='audio/wav', start_time=0)
            
            # Description input
            st.subheader("📝 Segment Description")
            description = st.text_area(
                "Describe this audio segment (for computer hearing context)",
                placeholder="e.g., 'Dog barking with car passing background', 'Piano melody in C major', 'Speech with emotional emphasis'",
                height=100
            )
            
            # Process button
            if st.button("✨ Extract Features & Prepare Download"):
                with st.spinner("Computing audio characteristics..."):
                    # Compute features
                    features = compute_audio_features(y_segment, sr)
                    
                    # Create WAV buffer for embedding
                    wav_buffer = io.BytesIO()
                    sf.write(wav_buffer, y_segment, sr, format='WAV')
                    wav_buffer.seek(0)
                    audio_b64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
                    
                    # Prepare output structure
                    output_data = {
                        "metadata": {
                            "original_filename": uploaded_file.name,
                            "file_type": file_type,
                            "segment_start_sec": t1,
                            "segment_end_sec": t2,
                            "segment_duration_sec": t2 - t1,
                            "sample_rate_hz": sr,
                            "num_samples": len(y_segment),
                            "description": description.strip() if description.strip() else "No description provided",
                            "feature_extraction_library": "librosa",
                            "extraction_timestamp": str(st.session_state.get('run_time', 'N/A'))
                        },
                        "audio_characteristics": features,
                        "audio_segment": {
                            "format": "wav",
                            "encoding": "base64",
                            "data": audio_b64
                        },
                        "computer_hearing_notes": {
                            "recommended_usage": [
                                "MFCCs and deltas: Speaker identification, speech recognition",
                                "Spectral features: Music genre classification, sound event detection",
                                "Chroma features: Music key detection, harmonic analysis",
                                "Tonnetz: Tonal similarity tasks",
                                "Onset strength: Rhythm analysis, beat tracking"
                            ],
                            "normalization_required": True,
                            "typical_preprocessing": "Standardize features per dataset; consider log-scaling spectral features"
                        }
                    }
                
                # Filename input
                st.subheader("💾 Download Output")
                default_name = f"audio_segment_{int(t1)}s_to_{int(t2)}s"
                output_name = st.text_input(
                    "Enter output filename (without extension)",
                    value=default_name,
                    help="Your file will be saved as [name].json"
                ).strip()
                
                if not output_name:
                    output_name = default_name
                
                # Create download button
                json_str = json.dumps(output_data, indent=2)
                st.download_button(
                    label="📥 Download Audio Characteristics Package",
                    data=json_str,
                    file_name=f"{output_name}.json",
                    mime="application/json",
                    help="Contains embedded audio segment + features + metadata"
                )
                
                # Display feature summary
                st.subheader("📊 Extracted Features Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Features", len(features))
                    st.metric("MFCC Coefficients", 39)  # 13 base + 13 delta + 13 delta2
                with col2:
                    st.metric("Spectral Features", 15)
                    st.metric("Temporal Features", 3)
                with col3:
                    st.metric("Chroma Features", 12)
                    st.metric("Tonnetz Features", 6)
                
                with st.expander("🔍 View Full Feature List"):
                    st.json({k: v for k, v in features.items() if not k.startswith('basic_metadata')})
                
                st.success(f"✅ Ready to download! File contains: Audio segment ({t2-t1:.1f}s) + {len(features)} acoustic features + your description")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Troubleshooting tips:\n- Ensure file is valid MP4/MP3/WAV\n- For videos: FFmpeg must be installed in environment\n- Try shorter audio segments\n- Check file isn't corrupted")
    
    finally:
        # Cleanup temp files
        try:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        except:
            pass

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 <strong>Computer Hearing Tip:</strong> For ML pipelines, extract features from consistent segment lengths (e.g., 1-3 seconds). 
    Normalize features across your dataset before training models.</p>
    <p>🔒 Your data never leaves your browser - all processing happens locally in this Streamlit app</p>
</div>
""", unsafe_allow_html=True)
