import streamlit as st
from pathlib import Path
from PIL import Image
import re

# Page config
st.set_page_config(
    page_title="Photo Album",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for elegance and responsiveness
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .album-header {
        text-align: center;
        margin-bottom: 2.5rem;
    }
    .person-section {
        background: white;
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    .person-section:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    .person-name {
        font-size: 1.9rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1.4rem;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 1.4rem;
    }
    .image-container {
        aspect-ratio: 1/1;
        overflow: hidden;
        border-radius: 10px;
        background: #f7fafc;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .image-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.35s ease;
    }
    .image-container:hover img {
        transform: scale(1.04);
    }
    @media (max-width: 768px) {
        .image-grid {
            grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 1rem;
        }
        .person-name {
            font-size: 1.6rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def format_name(folder_name: str) -> str:
    """Convert 'firstname-lastname' to 'Firstname Lastname'"""
    parts = folder_name.split('-')
    return ' '.join(part.capitalize() for part in parts)

def main():
    st.title("✨ Personal Photo Album")
    st.markdown("<div class='album-header'>Browse memories by person</div>", unsafe_allow_html=True)
    
    data_dir = Path("data")
    if not data_dir.exists():
        st.error("❌ Folder `data` not found. Please create it with subfolders named like `firstname-lastname`.")
        return

    # Get all subdirectories in data/
    person_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not person_dirs:
        st.warning("📭 No person folders found in `data/`. Create folders like `john-doe` to get started!")
        return

    # Sort alphabetically by formatted name
    person_dirs.sort(key=lambda d: format_name(d.name).lower())

    valid_dirs = []
    for d in person_dirs:
        # Skip folders that don't match firstname-lastname pattern (at least one hyphen)
        if '-' not in d.name or d.name.startswith('.') or d.name.startswith('_'):
            continue
        valid_dirs.append(d)

    if not valid_dirs:
        st.warning("📭 No valid `firstname-lastname` folders found. Example: `anna-mueller`")
        return

    # Process each person
    for person_dir in valid_dirs:
        display_name = format_name(person_dir.name)
        
        # Find image files (case-insensitive)
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        image_files = [
            f for f in person_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            continue  # Skip if no images

        # Sort images by filename
        image_files.sort(key=lambda x: x.name.lower())

        # Render person section
        st.markdown('<div class="person-section">', unsafe_allow_html=True)
        st.markdown(f'<div class="person-name">{display_name}</div>', unsafe_allow_html=True)
        
        # Create responsive image grid
        cols = st.columns(min(4, len(image_files)))
        for idx, img_path in enumerate(image_files):
            col = cols[idx % len(cols)]
            with col:
                try:
                    img = Image.open(img_path)
                    # Resize for performance (max 1000px on longest side)
                    img.thumbnail((1000, 1000))
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.caption(f"⚠️ {img_path.name}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
