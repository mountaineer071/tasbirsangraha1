"""
COMPREHENSIVE WEB PHOTO ALBUM APPLICATION WITH IMAGE EDITING
Version: 3.0.0 - Extended with Image Upload & Digital Modification
Features: Table of Contents, Image Gallery, Comments, Ratings, Metadata Management, Search, Image Upload & Editing
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import base64
import json
import datetime
import uuid
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import csv
import io
import time
import math
from dataclasses import dataclass, asdict
from enum import Enum
import random
import string
from collections import defaultdict
import re
import os
from contextlib import contextmanager

# ==================== NEW IMPORTS FOR IMAGE EDITING ====================
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV not installed. Some editing features will be limited. Run: pip install opencv-python")

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    st.warning("Rembg not installed. Background removal feature unavailable. Run: pip install rembg")

try:
    from skimage import exposure, filters, restoration
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    st.warning("scikit-image not installed. Advanced filters unavailable. Run: pip install scikit-image")

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
class Config:
    """Application configuration constants"""
    APP_NAME = "MemoryVault Pro"
    VERSION = "3.0.0"
    # Get absolute paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    UPLOADS_DIR = BASE_DIR / "uploads"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Files and paths
    METADATA_FILE = METADATA_DIR / "album_metadata.json"
    DB_FILE = DB_DIR / "album.db"
    
    # Image settings
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB (increased for uploads)
    
    # Gallery settings
    ITEMS_PER_PAGE = 20
    GRID_COLUMNS = 4
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    MAX_COMMENT_LENGTH = 500
    MAX_CAPTION_LENGTH = 200
    
    # Upload settings
    MAX_UPLOAD_SIZE_MB = 15
    ALLOWED_UPLOAD_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp']
    
    # Cache
    CACHE_TTL = 3600  # 1 hour
    
    @classmethod
    def init_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.THUMBNAIL_DIR,
            cls.METADATA_DIR,
            cls.DB_DIR,
            cls.EXPORT_DIR,
            cls.UPLOADS_DIR,
            cls.TEMP_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create sample structure if no data exists
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_sample_structure()
    
    @classmethod
    def create_sample_structure(cls):
        """Create sample directory structure with example photos"""
        sample_people = ["john-smith", "sarah-johnson", "michael-brown"]
        for person in sample_people:
            person_dir = cls.DATA_DIR / person
            person_dir.mkdir(exist_ok=True)
            
            # Create a sample README file
            readme_file = person_dir / "README.txt"
            readme_file.write_text(f"Photos of {person.replace('-', ' ').title()}\nAdd your photos here!")
            
            # Add a simple sample image
            sample_image_path = person_dir / "sample.jpg"
            if not sample_image_path.exists():
                try:
                    from PIL import Image, ImageDraw
                    # Create image with gradient background
                    img = Image.new('RGB', (400, 300), color='#667eea')
                    draw = ImageDraw.Draw(img)
                    # Draw a simple person icon
                    draw.ellipse((150, 50, 250, 150), fill='#ffffff', outline='#4a5568')
                    draw.rectangle((150, 150, 250, 280), fill='#ffffff', outline='#4a5568')
                    # Add text
                    draw.text((120, 250), person.replace('-', ' ').title(), fill='#2d3748')
                    img.save(sample_image_path)
                    st.info(f"Created sample image for {person}")
                except Exception as e:
                    st.warning(f"Could not create sample image for {person}: {str(e)}")


class UserRoles(Enum):
    """User permission levels"""
    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    EDITOR = "editor"
    ADMIN = "admin"


# ============================================================================
# IMAGE EDITING TOOLS
# ============================================================================
class ImageEditor:
    """Comprehensive image editing and modification tools"""
    
    @staticmethod
    def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
        """Adjust brightness (0.0 to 2.0)"""
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
        """Adjust contrast (0.0 to 2.0)"""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_saturation(img: Image.Image, factor: float) -> Image.Image:
        """Adjust saturation (0.0 to 2.0)"""
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_sharpness(img: Image.Image, factor: float) -> Image.Image:
        """Adjust sharpness (0.0 to 2.0)"""
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_filter(img: Image.Image, filter_name: str) -> Image.Image:
        """Apply PIL filters"""
        filters = {
            'BLUR': ImageFilter.BLUR,
            'CONTOUR': ImageFilter.CONTOUR,
            'DETAIL': ImageFilter.DETAIL,
            'EDGE_ENHANCE': ImageFilter.EDGE_ENHANCE,
            'EDGE_ENHANCE_MORE': ImageFilter.EDGE_ENHANCE_MORE,
            'EMBOSS': ImageFilter.EMBOSS,
            'FIND_EDGES': ImageFilter.FIND_EDGES,
            'SMOOTH': ImageFilter.SMOOTH,
            'SMOOTH_MORE': ImageFilter.SMOOTH_MORE,
            'SHARPEN': ImageFilter.SHARPEN
        }
        if filter_name in filters:
            return img.filter(filters[filter_name])
        return img
    
    @staticmethod
    def rotate_image(img: Image.Image, angle: float) -> Image.Image:
        """Rotate image by specified angle"""
        return img.rotate(angle, expand=True)
    
    @staticmethod
    def flip_image(img: Image.Image, direction: str) -> Image.Image:
        """Flip image horizontally or vertically"""
        if direction == 'horizontal':
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == 'vertical':
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
    
    @staticmethod
    def resize_image(img: Image.Image, width: int, height: int, maintain_aspect: bool = True) -> Image.Image:
        """Resize image"""
        if maintain_aspect:
            img.thumbnail((width, height), Image.Resampling.LANCZOS)
            return img
        else:
            return img.resize((width, height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def crop_image(img: Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
        """Crop image to specified coordinates"""
        return img.crop((left, top, right, bottom))
    
    @staticmethod
    def convert_mode(img: Image.Image, mode: str) -> Image.Image:
        """Convert image mode (RGB, L/grayscale, RGBA, etc.)"""
        if mode == 'grayscale':
            return img.convert('L')
        elif mode == 'sepia':
            # Convert to sepia tone
            sepia_img = img.convert('RGB')
            width, height = sepia_img.size
            pixels = sepia_img.load()
            
            for py in range(height):
                for px in range(width):
                    r, g, b = sepia_img.getpixel((px, py))
                    
                    tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                    tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                    tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                    
                    if tr > 255:
                        tr = 255
                    if tg > 255:
                        tg = 255
                    if tb > 255:
                        tb = 255
                    
                    pixels[px, py] = (tr, tg, tb)
            
            return sepia_img
        elif mode == 'rgba':
            return img.convert('RGBA')
        else:
            return img.convert(mode)
    
    @staticmethod
    def add_text(img: Image.Image, text: str, position: tuple, font_size: int = 20, 
                 color: tuple = (255, 255, 255), font_path: str = None) -> Image.Image:
        """Add text to image"""
        img_editable = img.copy()
        draw = ImageDraw.Draw(img_editable)
        
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        draw.text(position, text, color, font=font)
        return img_editable
    
    @staticmethod
    def add_border(img: Image.Image, border_size: int, border_color: tuple = (0, 0, 0)) -> Image.Image:
        """Add border to image"""
        new_width = img.width + 2 * border_size
        new_height = img.height + 2 * border_size
        
        if img.mode == 'RGBA':
            border_img = Image.new('RGBA', (new_width, new_height), border_color)
        else:
            border_img = Image.new('RGB', (new_width, new_height), border_color)
        
        border_img.paste(img, (border_size, border_size))
        return border_img
    
    @staticmethod
    def apply_vignette(img: Image.Image, intensity: float = 0.5) -> Image.Image:
        """Apply vignette effect to image"""
        if not CV2_AVAILABLE:
            st.warning("OpenCV not available. Vignette effect requires opencv-python")
            return img
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        rows, cols = img_cv.shape[:2]
        
        # Create vignette mask
        kernel_x = cv2.getGaussianKernel(cols, cols * intensity)
        kernel_y = cv2.getGaussianKernel(rows, rows * intensity)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        
        # Apply mask
        mask = np.dstack((mask, mask, mask))
        output = img_cv * mask
        
        # Convert back to PIL
        output_rgb = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb)
    
    @staticmethod
    def remove_background(img: Image.Image) -> Image.Image:
        """Remove background from image using rembg"""
        if not REMBG_AVAILABLE:
            st.error("Background removal requires rembg library. Install with: pip install rembg")
            return img
        
        # Convert PIL to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Remove background
        output_bytes = remove(img_byte_arr.read())
        
        # Convert back to PIL
        return Image.open(io.BytesIO(output_bytes))
    
    @staticmethod
    def apply_histogram_equalization(img: Image.Image) -> Image.Image:
        """Apply histogram equalization for better contrast"""
        if not SKIMAGE_AVAILABLE:
            st.warning("scikit-image not available. Using basic enhancement.")
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(1.5)
        
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            # Grayscale
            equalized = exposure.equalize_hist(img_array)
        else:
            # RGB - convert to LAB and equalize L channel
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = exposure.equalize_hist(lab[:,:,0]) * 255
            equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(equalized.astype(np.uint8))
    
    @staticmethod
    def apply_denoise(img: Image.Image, strength: float = 1.0) -> Image.Image:
        """Apply noise reduction"""
        if not CV2_AVAILABLE:
            st.warning("OpenCV not available. Denoising requires opencv-python")
            return img
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, strength, strength, 7, 21)
        output_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb)
    
    @staticmethod
    def apply_edge_detection(img: Image.Image, method: str = 'canny') -> Image.Image:
        """Apply edge detection"""
        if not CV2_AVAILABLE:
            st.warning("OpenCV not available. Edge detection requires opencv-python")
            return img
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        if method == 'canny':
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif method == 'sobel':
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.sqrt(cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5,
                                            cv2.convertScaleAbs(sobely), 0.5, 0))
            edges_colored = cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            edges_colored = img_cv
        
        output_rgb = cv2.cvtColor(edges_colored, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb)
    
    @staticmethod
    def apply_color_balance(img: Image.Image, red: float, green: float, blue: float) -> Image.Image:
        """Adjust color balance"""
        img_array = np.array(img).astype(np.float32)
        
        # Apply color adjustments
        img_array[:,:,0] *= red
        img_array[:,:,1] *= green
        img_array[:,:,2] *= blue
        
        # Clip values to valid range
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    @staticmethod
    def apply_temperature(img: Image.Image, temperature: float) -> Image.Image:
        """Adjust color temperature (-100 to 100)"""
        img_array = np.array(img).astype(np.float32)
        
        if temperature > 0:
            # Warmer (more red/yellow)
            img_array[:,:,0] *= (1 + temperature / 100 * 0.3)
            img_array[:,:,1] *= (1 + temperature / 100 * 0.1)
        else:
            # Cooler (more blue)
            img_array[:,:,2] *= (1 - temperature / 100 * 0.3)
            img_array[:,:,1] *= (1 - temperature / 100 * 0.1)
        
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    
    @staticmethod
    def apply_vintage_filter(img: Image.Image, intensity: float = 0.8) -> Image.Image:
        """Apply vintage/old photo filter"""
        # Convert to sepia
        sepia = ImageEditor.convert_mode(img, 'sepia')
        
        # Add noise/grain
        if CV2_AVAILABLE:
            sepia_array = np.array(sepia)
            noise = np.random.normal(0, 25 * intensity, sepia_array.shape).astype(np.uint8)
            noisy = cv2.addWeighted(sepia_array, 1 - intensity * 0.3, noise, intensity * 0.3, 0)
            return Image.fromarray(noisy)
        else:
            return sepia


# ============================================================================
# UPLOAD MANAGER
# ============================================================================
class UploadManager:
    """Handle image uploads and processing"""
    
    @staticmethod
    def validate_file(file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if file is None:
            return False, "No file uploaded"
        
        # Check file size
        if file.size > Config.MAX_IMAGE_SIZE:
            return False, f"File too large. Maximum size is {Config.MAX_UPLOAD_SIZE_MB}MB"
        
        # Check file extension
        file_ext = Path(file.name).suffix.lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        
        # Check MIME type
        if hasattr(file, 'type'):
            if file.type not in Config.ALLOWED_UPLOAD_TYPES:
                return False, f"Invalid MIME type: {file.type}"
        
        return True, "Valid file"
    
    @staticmethod
    def save_uploaded_file(file, person_folder: str = None) -> Tuple[Optional[Path], str]:
        """Save uploaded file to appropriate location"""
        is_valid, message = UploadManager.validate_file(file)
        if not is_valid:
            return None, message
        
        try:
            # Determine save location
            if person_folder:
                save_dir = Config.DATA_DIR / person_folder
                save_dir.mkdir(exist_ok=True)
            else:
                save_dir = Config.UPLOADS_DIR
                save_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_ext = Path(file.name).suffix.lower()
            unique_filename = f"upload_{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"
            file_path = save_dir / unique_filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            
            return file_path, "File uploaded successfully"
        except Exception as e:
            return None, f"Error saving file: {str(e)}"
    
    @staticmethod
    def process_uploaded_image(file_path: Path, metadata: Dict = None) -> Tuple[bool, str]:
        """Process uploaded image and add to database"""
        try:
            from album_manager import ImageMetadata, AlbumEntry, PersonProfile
            
            # Extract metadata
            img_metadata = ImageMetadata.from_image(file_path)
            
            # Create thumbnail
            thumbnail_path = ImageProcessor.create_thumbnail(file_path)
            thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
            
            # Add to database
            db = DatabaseManager()
            db.add_image(img_metadata, thumbnail_path_str)
            
            # Create album entry
            person_id = metadata.get('person_id') if metadata else None
            if not person_id:
                # Create default person if none specified
                people = db.get_all_people()
                if people:
                    person_id = people[0]['person_id']
                else:
                    # Create a default person
                    default_person = PersonProfile(
                        person_id=str(uuid.uuid4()),
                        folder_name="uploaded-photos",
                        display_name="Uploaded Photos",
                        bio="Photos uploaded through the web interface",
                        birth_date=None,
                        relationship="Other",
                        contact_info="",
                        social_links={},
                        profile_image=None,
                        created_at=datetime.datetime.now()
                    )
                    db.add_person(default_person)
                    person_id = default_person.person_id
            
            album_entry = AlbumEntry(
                entry_id=str(uuid.uuid4()),
                image_id=img_metadata.image_id,
                person_id=person_id,
                caption=metadata.get('caption', file_path.stem.replace('_', ' ').title()),
                description=metadata.get('description', 'Uploaded photo'),
                location=metadata.get('location', ''),
                date_taken=metadata.get('date_taken', img_metadata.created_date),
                tags=metadata.get('tags', ['uploaded', 'photo']),
                privacy_level=metadata.get('privacy_level', 'public'),
                created_by='upload',
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )
            
            db.add_album_entry(album_entry)
            
            return True, "Image processed and added to album"
        except Exception as e:
            return False, f"Error processing image: {str(e)}"


# ============================================================================
# UI COMPONENTS - EXTENDED
# ============================================================================
class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def rating_stars(rating: float, max_rating: int = 5, size: int = 20) -> str:
        """Generate HTML for rating stars"""
        if rating is None or rating <= 0:
            rating = 0
        stars_html = []
        full_stars = int(rating)
        has_half_star = rating - full_stars >= 0.5
        for i in range(max_rating):
            if i < full_stars:
                stars_html.append('⭐')
            elif i == full_stars and has_half_star:
                stars_html.append('⭐')
            else:
                stars_html.append('☆')
        return f"""
        <div style="color: #FFD700; font-size: {size}px; letter-spacing: 1px; display: inline-block;">
        {''.join(stars_html)}
        <span style="color: #666; font-size: 14px; margin-left: 8px; vertical-align: middle;">
        {rating:.1f}/{max_rating}
        </span>
        </div>
        """
    
    @staticmethod
    def tag_badges(tags: List[str], max_display: int = 5) -> str:
        """Generate HTML for tag badges"""
        if not tags:
            return ""
        displayed_tags = tags[:max_display]
        extra_count = len(tags) - max_display if len(tags) > max_display else 0
        badges = []
        for tag in displayed_tags:
            badges.append(f'''
            <span style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin: 2px;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-transform: capitalize;
            ">
            {tag.replace('-', ' ')}
            </span>
            ''')
        html = ' '.join(badges)
        if extra_count > 0:
            html += f'''
            <span style="
            background: #f0f0f0;
            color: #666;
            padding: 4px 8px;
            border-radius: 20px;
            font-size: 12px;
            margin: 2px;
            display: inline-block;
            ">
            +{extra_count} more
            </span>
            '''
        return html
    
    @staticmethod
    def color_picker(label: str, default_color: str = "#ffffff") -> str:
        """Color picker widget"""
        return st.color_picker(label, default_color)
    
    @staticmethod
    def image_preview(img: Image.Image, title: str = None, width: int = 400):
        """Display image preview with optional title"""
        if title:
            st.subheader(title)
        st.image(img, use_column_width=True)


# ============================================================================
# EXTENDED ALBUM MANAGER WITH EDITING CAPABILITIES
# ============================================================================
class ExtendedAlbumManager(AlbumManager):
    """Extended album manager with image editing capabilities"""
    
    def __init__(self):
        super().__init__()
        self.editor = ImageEditor()
    
    def edit_image(self, image_path: Path, edits: Dict) -> Tuple[Optional[Image.Image], str]:
        """Apply multiple edits to an image"""
        try:
            with Image.open(image_path) as img:
                # Handle EXIF rotation
                img = ImageOps.exif_transpose(img)
                
                # Convert to RGB if necessary for editing
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                edited_img = img.copy()
                
                # Apply edits in order
                if edits.get('brightness') is not None:
                    edited_img = self.editor.apply_brightness(edited_img, edits['brightness'])
                
                if edits.get('contrast') is not None:
                    edited_img = self.editor.apply_contrast(edited_img, edits['contrast'])
                
                if edits.get('saturation') is not None:
                    edited_img = self.editor.apply_saturation(edited_img, edits['saturation'])
                
                if edits.get('sharpness') is not None:
                    edited_img = self.editor.apply_sharpness(edited_img, edits['sharpness'])
                
                if edits.get('temperature') is not None:
                    edited_img = self.editor.apply_temperature(edited_img, edits['temperature'])
                
                if edits.get('color_balance'):
                    cb = edits['color_balance']
                    edited_img = self.editor.apply_color_balance(
                        edited_img, cb.get('red', 1.0), cb.get('green', 1.0), cb.get('blue', 1.0)
                    )
                
                if edits.get('filter'):
                    edited_img = self.editor.apply_filter(edited_img, edits['filter'])
                
                if edits.get('mode'):
                    edited_img = self.editor.convert_mode(edited_img, edits['mode'])
                
                if edits.get('rotate'):
                    edited_img = self.editor.rotate_image(edited_img, edits['rotate'])
                
                if edits.get('flip'):
                    edited_img = self.editor.flip_image(edited_img, edits['flip'])
                
                if edits.get('resize'):
                    rs = edits['resize']
                    edited_img = self.editor.resize_image(
                        edited_img, rs['width'], rs['height'], rs.get('maintain_aspect', True)
                    )
                
                if edits.get('crop'):
                    cr = edits['crop']
                    edited_img = self.editor.crop_image(
                        edited_img, cr['left'], cr['top'], cr['right'], cr['bottom']
                    )
                
                if edits.get('border'):
                    br = edits['border']
                    edited_img = self.editor.add_border(
                        edited_img, br['size'], br.get('color', (0, 0, 0))
                    )
                
                if edits.get('vignette'):
                    edited_img = self.editor.apply_vignette(edited_img, edits['vignette'])
                
                if edits.get('denoise'):
                    edited_img = self.editor.apply_denoise(edited_img, edits['denoise'])
                
                if edits.get('equalize'):
                    edited_img = self.editor.apply_histogram_equalization(edited_img)
                
                if edits.get('remove_bg'):
                    edited_img = self.editor.remove_background(edited_img)
                
                if edits.get('edge_detection'):
                    edited_img = self.editor.apply_edge_detection(edited_img, edits['edge_detection'])
                
                if edits.get('vintage'):
                    edited_img = self.editor.apply_vintage_filter(edited_img, edits['vintage'])
                
                if edits.get('text'):
                    txt = edits['text']
                    edited_img = self.editor.add_text(
                        edited_img, txt['content'], (txt['x'], txt['y']),
                        txt.get('font_size', 20), txt.get('color', (255, 255, 255))
                    )
                
                return edited_img, "Image edited successfully"
        except Exception as e:
            return None, f"Error editing image: {str(e)}"
    
    def save_edited_image(self, edited_img: Image.Image, original_path: Path, suffix: str = "_edited") -> Tuple[Optional[Path], str]:
        """Save edited image"""
        try:
            # Generate new filename
            original_stem = original_path.stem
            file_ext = original_path.suffix
            new_filename = f"{original_stem}{suffix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
            new_path = original_path.parent / new_filename
            
            # Save image
            edited_img.save(new_path, quality=95, optimize=True)
            
            return new_path, "Image saved successfully"
        except Exception as e:
            return None, f"Error saving image: {str(e)}"


# ============================================================================
# NEW UI PAGES FOR UPLOAD AND EDITING
# ============================================================================
class ExtendedPhotoAlbumApp(PhotoAlbumApp):
    """Extended photo album app with upload and editing features"""
    
    def __init__(self):
        super().__init__()
        self.extended_manager = ExtendedAlbumManager()
    
    def render_upload_page(self):
        """Render image upload page"""
        st.title("📤 Upload Photos")
        
        # Upload options
        upload_type = st.radio(
            "Upload Type",
            ["Single Photo", "Multiple Photos", "Bulk Upload to Person"],
            horizontal=True
        )
        
        if upload_type == "Single Photo":
            self._render_single_upload()
        elif upload_type == "Multiple Photos":
            self._render_multiple_upload()
        else:
            self._render_bulk_upload()
    
    def _render_single_upload(self):
        """Render single photo upload interface"""
        st.subheader("📷 Upload Single Photo")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=[ext[1:] for ext in Config.ALLOWED_EXTENSIONS],
            accept_multiple_files=False,
            help=f"Maximum size: {Config.MAX_UPLOAD_SIZE_MB}MB"
        )
        
        if uploaded_file:
            # Validate file
            is_valid, message = UploadManager.validate_file(uploaded_file)
            if not is_valid:
                st.error(message)
                return
            
            # Display preview
            st.image(uploaded_file, caption=f"Preview: {uploaded_file.name}", use_column_width=True)
            
            # Metadata form
            with st.form("upload_metadata_form"):
                st.subheader("📝 Photo Information")
                
                # Select person
                people = self.manager.db.get_all_people()
                person_options = {p['display_name']: p['person_id'] for p in people}
                selected_person = st.selectbox(
                    "Assign to Person",
                    options=list(person_options.keys()),
                    help="Choose which person this photo belongs to"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    caption = st.text_input(
                        "Caption",
                        value=Path(uploaded_file.name).stem.replace('_', ' ').title(),
                        max_chars=Config.MAX_CAPTION_LENGTH
                    )
                    location = st.text_input("Location")
                
                with col2:
                    description = st.text_area("Description", height=100)
                    tags = st.text_input("Tags (comma-separated)", "photo, memory")
                
                privacy = st.selectbox(
                    "Privacy Level",
                    ["public", "private"],
                    help="Public photos are visible to all users"
                )
                
                submitted = st.form_submit_button("📤 Upload Photo")
                
                if submitted:
                    with st.spinner("Uploading and processing photo..."):
                        # Save file
                        person_folder = None
                        if selected_person:
                            person_id = person_options[selected_person]
                            for p in people:
                                if p['person_id'] == person_id:
                                    person_folder = p['folder_name']
                                    break
                        
                        file_path, save_message = UploadManager.save_uploaded_file(
                            uploaded_file, person_folder
                        )
                        
                        if file_path:
                            # Process and add to database
                            metadata = {
                                'person_id': person_id if selected_person else None,
                                'caption': caption,
                                'description': description,
                                'location': location,
                                'tags': [tag.strip() for tag in tags.split(',') if tag.strip()],
                                'privacy_level': privacy,
                                'date_taken': datetime.datetime.now()
                            }
                            
                            success, process_message = UploadManager.process_uploaded_image(
                                file_path, metadata
                            )
                            
                            if success:
                                st.success(f"✅ {save_message}")
                                st.success(f"✅ {process_message}")
                                st.balloons()
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(process_message)
                        else:
                            st.error(save_message)
    
    def _render_multiple_upload(self):
        """Render multiple photo upload interface"""
        st.subheader("📸 Upload Multiple Photos")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=[ext[1:] for ext in Config.ALLOWED_EXTENSIONS],
            accept_multiple_files=True,
            help=f"Maximum {Config.MAX_UPLOAD_SIZE_MB}MB per file"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files")
            
            # Quick preview of first few images
            preview_cols = st.columns(min(4, len(uploaded_files)))
            for idx, file in enumerate(uploaded_files[:4]):
                with preview_cols[idx]:
                    st.image(file, caption=file.name, use_column_width=True)
            
            # Common metadata for all uploads
            with st.form("bulk_upload_form"):
                st.subheader("📋 Common Information for All Photos")
                
                people = self.manager.db.get_all_people()
                person_options = {p['display_name']: p['person_id'] for p in people}
                selected_person = st.selectbox(
                    "Assign all photos to",
                    options=list(person_options.keys())
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    location = st.text_input("Common Location")
                    privacy = st.selectbox("Privacy Level", ["public", "private"])
                
                with col2:
                    default_tags = st.text_input("Common Tags", "photo, memory, uploaded")
                    add_timestamp = st.checkbox("Add timestamp to captions", value=True)
                
                submitted = st.form_submit_button("📤 Upload All Photos")
                
                if submitted:
                    with st.spinner(f"Uploading {len(uploaded_files)} photos..."):
                        progress_bar = st.progress(0)
                        success_count = 0
                        error_count = 0
                        
                        for idx, file in enumerate(uploaded_files):
                            progress = (idx) / len(uploaded_files)
                            progress_bar.progress(progress)
                            
                            # Validate
                            is_valid, _ = UploadManager.validate_file(file)
                            if not is_valid:
                                error_count += 1
                                continue
                            
                            # Save file
                            person_folder = None
                            if selected_person:
                                person_id = person_options[selected_person]
                                for p in people:
                                    if p['person_id'] == person_id:
                                        person_folder = p['folder_name']
                                        break
                            
                            file_path, _ = UploadManager.save_uploaded_file(file, person_folder)
                            
                            if file_path:
                                # Generate caption
                                caption = file_path.stem.replace('_', ' ').title()
                                if add_timestamp:
                                    caption = f"{caption} {datetime.datetime.now().strftime('%Y-%m-%d')}"
                                
                                # Process
                                metadata = {
                                    'person_id': person_id if selected_person else None,
                                    'caption': caption,
                                    'description': f"Uploaded on {datetime.datetime.now().strftime('%Y-%m-%d')}",
                                    'location': location,
                                    'tags': [tag.strip() for tag in default_tags.split(',') if tag.strip()],
                                    'privacy_level': privacy,
                                    'date_taken': datetime.datetime.now()
                                }
                                
                                success, _ = UploadManager.process_uploaded_image(file_path, metadata)
                                if success:
                                    success_count += 1
                                else:
                                    error_count += 1
                        
                        progress_bar.progress(1.0)
                        st.success(f"✅ Uploaded {success_count} photos successfully!")
                        if error_count > 0:
                            st.warning(f"⚠️ {error_count} photos failed to upload")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
    
    def _render_bulk_upload(self):
        """Render bulk upload to specific person folder"""
        st.subheader("📁 Bulk Upload to Person")
        
        people = self.manager.db.get_all_people()
        if not people:
            st.warning("No people found. Please add a person first.")
            if st.button("Go to People Management"):
                st.session_state['current_page'] = 'people'
                st.rerun()
            return
        
        selected_person = st.selectbox(
            "Select Person",
            options=[p['display_name'] for p in people]
        )
        
        person_data = next((p for p in people if p['display_name'] == selected_person), None)
        
        if person_data:
            st.info(f"Photos will be uploaded to: `{person_data['folder_name']}` folder")
            
            uploaded_files = st.file_uploader(
                "Choose photos",
                type=[ext[1:] for ext in Config.ALLOWED_EXTENSIONS],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.info(f"Ready to upload {len(uploaded_files)} photos to {selected_person}")
                
                if st.button("📤 Upload All Photos", type="primary"):
                    with st.spinner("Uploading..."):
                        success_count = 0
                        for file in uploaded_files:
                            file_path, _ = UploadManager.save_uploaded_file(
                                file, person_data['folder_name']
                            )
                            if file_path:
                                # Quick process without detailed metadata
                                metadata = {
                                    'person_id': person_data['person_id'],
                                    'caption': file_path.stem.replace('_', ' ').title(),
                                    'description': f"Photo of {selected_person}",
                                    'tags': [selected_person.lower().replace(' ', '-'), 'photo'],
                                    'privacy_level': 'public'
                                }
                                success, _ = UploadManager.process_uploaded_image(file_path, metadata)
                                if success:
                                    success_count += 1
                        
                        st.success(f"✅ Uploaded {success_count} photos to {selected_person}'s album!")
                        st.balloons()
    
    def render_image_editor_page(self):
        """Render image editor page"""
        st.title("🎨 Image Editor")
        
        # Check if we have a selected image
        entry_id = st.session_state.get('selected_image_for_edit')
        if not entry_id:
            st.info("Select an image from the gallery to edit, or upload a new one.")
            
            # Option to upload directly to editor
            uploaded_file = st.file_uploader(
                "Or upload an image to edit",
                type=[ext[1:] for ext in Config.ALLOWED_EXTENSIONS]
            )
            
            if uploaded_file:
                # Save temporarily and set as selected
                temp_path = Config.TEMP_DIR / f"temp_{uuid.uuid4().hex}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.session_state['temp_edit_image'] = temp_path
                st.rerun()
            
            return
        
        # Get image
        if 'temp_edit_image' in st.session_state:
            image_path = st.session_state['temp_edit_image']
            original_img = Image.open(image_path)
        else:
            entry = self.extended_manager.get_entry_with_details(entry_id)
            if not entry:
                st.error("Image not found")
                return
            
            if entry.get('filepath'):
                image_path = Config.DATA_DIR / entry['filepath']
                original_img = Image.open(image_path)
            else:
                st.error("Image file not found")
                return
        
        # Display original
        st.subheader("Original Image")
        st.image(original_img, use_column_width=True)
        
        # Editing tools in tabs
        st.divider()
        st.subheader("🔧 Editing Tools")
        
        tabs = st.tabs([
            "🎨 Basic Adjustments",
            "🎭 Filters & Effects",
            "✂️ Transform",
            "✏️ Text & Draw",
            "⚡ AI Tools"
        ])
        
        edits = {}
        
        with tabs[0]:  # Basic Adjustments
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.1)
                if brightness != 1.0:
                    edits['brightness'] = brightness
            
            with col2:
                contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.1)
                if contrast != 1.0:
                    edits['contrast'] = contrast
            
            with col3:
                saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
                if saturation != 1.0:
                    edits['saturation'] = saturation
            
            with col4:
                sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)
                if sharpness != 1.0:
                    edits['sharpness'] = sharpness
            
            # Color temperature
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperature", -100, 100, 0, 5)
                if temperature != 0:
                    edits['temperature'] = temperature
            
            with col2:
                st.info("Color Balance")
                col_r, col_g, col_b = st.columns(3)
                with col_r:
                    red = st.slider("Red", 0.0, 2.0, 1.0, 0.1)
                with col_g:
                    green = st.slider("Green", 0.0, 2.0, 1.0, 0.1)
                with col_b:
                    blue = st.slider("Blue", 0.0, 2.0, 1.0, 0.1)
                
                if red != 1.0 or green != 1.0 or blue != 1.0:
                    edits['color_balance'] = {'red': red, 'green': green, 'blue': blue}
        
        with tabs[1]:  # Filters & Effects
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("PIL Filters")
                filter_options = [
                    'None', 'BLUR', 'CONTOUR', 'DETAIL', 'EDGE_ENHANCE',
                    'EDGE_ENHANCE_MORE', 'EMBOSS', 'FIND_EDGES', 'SMOOTH',
                    'SMOOTH_MORE', 'SHARPEN'
                ]
                selected_filter = st.selectbox("Select Filter", filter_options)
                if selected_filter != 'None':
                    edits['filter'] = selected_filter
                
                st.subheader("Color Modes")
                mode_options = ['None', 'grayscale', 'sepia']
                selected_mode = st.selectbox("Color Mode", mode_options)
                if selected_mode != 'None':
                    edits['mode'] = selected_mode
            
            with col2:
                st.subheader("Advanced Effects")
                
                if REMBG_AVAILABLE:
                    if st.checkbox("Remove Background"):
                        edits['remove_bg'] = True
                        st.success("✅ Background removal enabled")
                else:
                    st.warning("Background removal requires: pip install rembg")
                
                if CV2_AVAILABLE:
                    vignette = st.slider("Vignette Effect", 0.0, 1.0, 0.0, 0.1)
                    if vignette > 0:
                        edits['vignette'] = vignette
                    
                    denoise = st.slider("Denoise Strength", 0.0, 10.0, 0.0, 0.5)
                    if denoise > 0:
                        edits['denoise'] = denoise
                    
                    if st.checkbox("Histogram Equalization"):
                        edits['equalize'] = True
                    
                    edge_method = st.selectbox("Edge Detection", ['None', 'canny', 'sobel'])
                    if edge_method != 'None':
                        edits['edge_detection'] = edge_method
                else:
                    st.warning("Advanced effects require: pip install opencv-python")
                
                vintage = st.slider("Vintage Effect", 0.0, 1.0, 0.0, 0.1)
                if vintage > 0:
                    edits['vintage'] = vintage
        
        with tabs[2]:  # Transform
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Rotation & Flip")
                rotate_angle = st.slider("Rotate (degrees)", -180, 180, 0, 15)
                if rotate_angle != 0:
                    edits['rotate'] = rotate_angle
                
                flip_option = st.selectbox("Flip", ['None', 'horizontal', 'vertical'])
                if flip_option != 'None':
                    edits['flip'] = flip_option
            
            with col2:
                st.subheader("Resize")
                maintain_aspect = st.checkbox("Maintain Aspect Ratio", value=True)
                new_width = st.number_input("Width (px)", min_value=100, value=original_img.width)
                new_height = st.number_input("Height (px)", min_value=100, value=original_img.height)
                
                if st.button("Apply Resize"):
                    edits['resize'] = {
                        'width': int(new_width),
                        'height': int(new_height),
                        'maintain_aspect': maintain_aspect
                    }
                
                st.subheader("Crop")
                st.info("Specify crop coordinates (left, top, right, bottom)")
                crop_left = st.number_input("Left", 0, original_img.width, 0)
                crop_top = st.number_input("Top", 0, original_img.height, 0)
                crop_right = st.number_input("Right", 0, original_img.width, original_img.width)
                crop_bottom = st.number_input("Bottom", 0, original_img.height, original_img.height)
                
                if st.button("Apply Crop"):
                    edits['crop'] = {
                        'left': crop_left,
                        'top': crop_top,
                        'right': crop_right,
                        'bottom': crop_bottom
                    }
        
        with tabs[3]:  # Text & Draw
            st.subheader("Add Text")
            
            col1, col2 = st.columns(2)
            with col1:
                text_content = st.text_input("Text to add")
                text_x = st.number_input("X position", 0, original_img.width, 10)
                text_y = st.number_input("Y position", 0, original_img.height, 10)
                font_size = st.slider("Font Size", 10, 100, 20)
            
            with col2:
                text_color = UIComponents.color_picker("Text Color", "#ffffff")
                # Convert hex to RGB
                text_color_rgb = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                if text_content:
                    edits['text'] = {
                        'content': text_content,
                        'x': text_x,
                        'y': text_y,
                        'font_size': font_size,
                        'color': text_color_rgb
                    }
            
            st.subheader("Add Border")
            border_size = st.slider("Border Size (px)", 0, 50, 0)
            if border_size > 0:
                border_color = UIComponents.color_picker("Border Color", "#000000")
                border_color_rgb = tuple(int(border_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                edits['border'] = {
                    'size': border_size,
                    'color': border_color_rgb
                }
        
        with tabs[4]:  # AI Tools
            st.subheader("🤖 AI-Powered Enhancements")
            
            st.info("""
            **Note:** These features require additional libraries:
            - Background removal: `pip install rembg`
            - Advanced filters: `pip install scikit-image`
            - Denoising/Effects: `pip install opencv-python`
            """)
            
            if REMBG_AVAILABLE:
                if st.button("✨ Auto Remove Background"):
                    edits['remove_bg'] = True
                    st.success("Background removal will be applied!")
            
            if CV2_AVAILABLE:
                if st.button("✨ Auto Enhance (Equalize)"):
                    edits['equalize'] = True
                    st.success("Histogram equalization will be applied!")
                
                if st.button("✨ Reduce Noise"):
                    edits['denoise'] = 2.0
                    st.success("Denoising will be applied!")
            
            # Preset filters
            st.subheader("🎨 Quick Presets")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Vintage"):
                    edits.update({
                        'mode': 'sepia',
                        'vintage': 0.7,
                        'contrast': 1.2
                    })
            
            with col2:
                if st.button("B&W High Contrast"):
                    edits.update({
                        'mode': 'grayscale',
                        'contrast': 1.5,
                        'brightness': 1.1
                    })
            
            with col3:
                if st.button("Vivid Colors"):
                    edits.update({
                        'saturation': 1.4,
                        'contrast': 1.2,
                        'sharpness': 1.3
                    })
        
        # Preview and apply
        st.divider()
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            if st.button("🔄 Reset All Edits"):
                st.session_state['image_edits'] = {}
                st.rerun()
        
        with col2:
            st.subheader("Preview")
        
        with col3:
            if st.button("💾 Save Edited Image", type="primary"):
                if edits:
                    with st.spinner("Applying edits and saving..."):
                        edited_img, message = self.extended_manager.edit_image(image_path, edits)
                        if edited_img:
                            save_path, save_message = self.extended_manager.save_edited_image(
                                edited_img, image_path
                            )
                            if save_path:
                                st.success("✅ Image saved successfully!")
                                
                                # Option to add to album
                                if st.checkbox("Add edited image to album"):
                                    entry = self.extended_manager.get_entry_with_details(entry_id)
                                    if entry:
                                        metadata = {
                                            'person_id': entry['person_id'],
                                            'caption': f"{entry.get('caption', 'Edited Photo')} (Edited)",
                                            'description': entry.get('description', ''),
                                            'location': entry.get('location', ''),
                                            'tags': entry.get('tags', []) + ['edited'],
                                            'privacy_level': entry.get('privacy_level', 'public')
                                        }
                                        success, msg = UploadManager.process_uploaded_image(save_path, metadata)
                                        if success:
                                            st.success("✅ Added to album!")
                                
                                st.balloons()
                            else:
                                st.error(save_message)
                        else:
                            st.error(message)
        
        # Live preview
        if edits:
            with st.spinner("Generating preview..."):
                preview_img, _ = self.extended_manager.edit_image(image_path, edits)
                if preview_img:
                    st.image(preview_img, caption="Edited Preview", use_column_width=True)
        else:
            st.info("Adjust settings above to see preview")
    
    def render_sidebar(self):
        """Extended sidebar with upload and edit options"""
        with st.sidebar:
            st.title(f"📸 {Config.APP_NAME}")
            st.caption(f"v{Config.VERSION}")
            
            # User info
            st.divider()
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**{st.session_state['username']}**")
                st.caption(st.session_state['user_role'].title())
            
            # Navigation
            st.divider()
            st.subheader("Navigation")
            nav_options = {
                "🏠 Dashboard": "dashboard",
                "📤 Upload": "upload",
                "👥 People": "people",
                "📁 Gallery": "gallery",
                "🎨 Editor": "editor",
                "⭐ Favorites": "favorites",
                "🔍 Search": "search",
                "📊 Statistics": "statistics",
                "⚙️ Settings": "settings",
                "📤 Import/Export": "import_export"
            }
            
            for label, key in nav_options.items():
                if st.button(label, use_container_width=True, key=f"nav_{key}"):
                    st.session_state['current_page'] = key
                    # Clear editor state when navigating away
                    if key != 'editor':
                        st.session_state.pop('selected_image_for_edit', None)
                    st.rerun()
            
            # Quick actions
            st.divider()
            st.subheader("Quick Actions")
            if st.button("🔄 Scan Directory", use_container_width=True):
                with st.spinner("Scanning directory..."):
                    results = self.manager.scan_directory()
                    if results['new_images'] > 0 or results['updated_images'] > 0:
                        st.success(f"Found {results['new_images']} new and {results['updated_images']} updated images")
                    st.rerun()
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                self.manager.cache.clear()
                st.success("Cache cleared!")
                st.rerun()
            
            # Display info
            st.divider()
            st.subheader("Info")
            people = self.manager.db.get_all_people()
            total_images = 0
            for person in people:
                stats = self.manager.get_person_stats(person['person_id'])
                total_images += stats['image_count']
            
            st.metric("People", len(people))
            st.metric("Total Images", total_images)
            
            # Library status
            st.divider()
            st.subheader("Library Status")
            if CV2_AVAILABLE:
                st.success("✅ OpenCV")
            else:
                st.error("❌ OpenCV")
            
            if REMBG_AVAILABLE:
                st.success("✅ Rembg")
            else:
                st.error("❌ Rembg")
            
            if SKIMAGE_AVAILABLE:
                st.success("✅ scikit-image")
            else:
                st.error("❌ scikit-image")
    
    def render_main(self):
        """Extended main renderer"""
        # Render sidebar
        self.render_sidebar()
        
        # Get current page
        current_page = st.session_state.get('current_page', 'dashboard')
        
        # Render appropriate page
        if current_page == 'dashboard':
            self.render_dashboard()
        elif current_page == 'upload':
            self.render_upload_page()
        elif current_page == 'people':
            self.render_people_page()
        elif current_page == 'gallery':
            self.render_gallery_page()
        elif current_page == 'image_detail':
            self.render_image_detail_page()
        elif current_page == 'editor':
            self.render_image_editor_page()
        elif current_page == 'favorites':
            self.render_favorites_page()
        elif current_page == 'search':
            self.render_search_page()
        elif current_page == 'statistics':
            self.render_statistics_page()
        elif current_page == 'settings':
            self.render_settings_page()
        elif current_page == 'import_export':
            self.render_import_export_page()
        else:
            self.render_dashboard()
        
        # Footer
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} v{Config.VERSION}")
            st.caption("A comprehensive photo album management system with advanced editing")
            st.caption(f"Libraries: {'OpenCV, ' if CV2_AVAILABLE else ''}{'Rembg, ' if REMBG_AVAILABLE else ''}{'scikit-image' if SKIMAGE_AVAILABLE else ''}")


# ============================================================================
# INSTALLATION INSTRUCTIONS
# ============================================================================
def show_installation_guide():
    """Display installation guide for required libraries"""
    st.title("🔧 Installation Guide")
    st.markdown("""
    ## Required Libraries for Image Editing Features
    
    ### Basic Installation (Core Features)
    ```bash
    pip install streamlit pillow numpy pandas
    ```
    
    ### Full Installation (All Editing Features)
    ```bash
    pip install streamlit pillow numpy pandas
    pip install opencv-python        # For advanced filters, denoising, edge detection
    pip install rembg                # For background removal
    pip install scikit-image         # For histogram equalization and advanced filters
    ```
    
    ### Optional Libraries
    ```bash
    pip install insightface         # For face recognition
    pip install torch torchvision   # For AI-powered enhancements
    ```
    
    ## Quick Start
    ```bash
    # Install all at once
    pip install streamlit pillow numpy pandas opencv-python rembg scikit-image
    
    # Run the application
    streamlit run photo_album_app.py
    ```
    
    ## Feature Availability
    
    | Feature | Library | Command |
    |---------|---------|---------|
    | Basic Adjustments | Pillow (built-in) | ✅ Included |
    | Filters (Blur, Sharpen, etc.) | Pillow | ✅ Included |
    | Background Removal | rembg | `pip install rembg` |
    | Denoising, Edge Detection | opencv-python | `pip install opencv-python` |
    | Histogram Equalization | scikit-image | `pip install scikit-image` |
    | Advanced Color Adjustments | opencv-python | `pip install opencv-python` |
    
    ## Troubleshooting
    
    **OpenCV import error:**
    ```bash
    pip uninstall opencv-python opencv-python-headless opencv-contrib-python
    pip install opencv-python
    ```
    
    **Rembg slow performance:**
    ```bash
    pip install rembg[gpu]  # If you have CUDA GPU
    ```
    
    **Memory issues with large images:**
    - Reduce image size before uploading
    - Increase system RAM or use cloud deployment
    """)
    
    if st.button("Continue to Application"):
        st.session_state['show_install_guide'] = False
        st.rerun()


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================
def main():
    """Main application entry point"""
    try:
        # Show installation guide on first run
        if 'show_install_guide' not in st.session_state:
            st.session_state['show_install_guide'] = True
        
        if st.session_state.get('show_install_guide'):
            show_installation_guide()
            return
        
        # Initialize app
        app = ExtendedPhotoAlbumApp()
        
        # Check initialization
        if not app.initialized:
            st.error("Application failed to initialize. Check directory permissions.")
            return
        
        # Render main app
        app.render_main()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the logs for details.")
        
        # Show error details in expander
        with st.expander("Error Details"):
            st.exception(e)
        
        # Recovery options
        if st.button("Try Again"):
            st.rerun()
        
        if st.button("Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
