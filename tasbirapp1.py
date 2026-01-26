import streamlit as st
from pathlib import Path
import os
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Photo Album",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .album-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .person-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .person-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .person-name {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #2c3e50;
        text-align: center;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 1rem;
    }
    .image-container {
        position: relative;
        overflow: hidden;
        border-radius: 8px;
        aspect-ratio: 1/1;
    }
    .image-container img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.3s ease;
    }
    .image-container:hover img {
        transform: scale(1.05);
    }
    .stImage > img {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📸 Personal Photo Album")
    st.markdown("---")
    
    # Get data directory
    data_dir = Path("data")
    if not data_dir.exists():
        st.error("❌ 'data' folder not found. Please create it with subfolders named 'FirstName LastName'")
        return
    
    # Get all person directories
    person_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not person_dirs:
        st.warning("⚠️ No person folders found in 'data' directory")
        return
    
    # Sort alphabetically by full name
    person_dirs.sort(key=lambda x: x.name)
    
    # Process each person
    for person_dir in person_dirs:
        # Get image files (case-insensitive)
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp')
        image_files = []
        for ext in image_extensions:
            image_files.extend(person_dir.glob(ext))
            image_files.extend(person_dir.glob(ext.upper()))
        
        if not image_files:
            continue
            
        # Display person section
        st.markdown(f'<div class="person-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="person-name">{person_dir.name}</div>', unsafe_allow_html=True)
        
        # Create image grid
        cols = st.columns(min(4, len(image_files)))  # Max 4 columns
        for idx, img_path in enumerate(sorted(image_files)):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                try:
                    # Open and resize image for better performance
                    img = Image.open(img_path)
                    # Resize to max 800px width while maintaining aspect ratio
                    img.thumbnail((800, 800))
                    
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading {img_path.name}: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

if __name__ == "__main__":
    main()
