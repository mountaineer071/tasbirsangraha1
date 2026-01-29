"""
COMPREHENSIVE WEB PHOTO ALBUM APPLICATION WITH AI ENHANCEMENTS
Version: 3.0.0 - AI Enhanced with LLM & Diffusion Models
Features: Table of Contents, Image Gallery, Comments, Ratings, Metadata Management, Search,
          AI Image Analysis, AI Style Transfer, AI Editing, Batch Processing
"""

import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ExifTags, ImageFilter, ImageEnhance
import base64
import json
import datetime
import uuid
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
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

# ============================================================================
# AI IMPORTS
# ============================================================================
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    from transformers import (
        AutoModelForSeq2SeqLM, 
        AutoTokenizer,
        BlipProcessor, 
        BlipForConditionalGeneration,
        CLIPProcessor, 
        CLIPModel,
        pipeline
    )
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler
    )
    import cv2
    from scipy import ndimage
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("AI libraries not installed. Running in basic mode.")

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
class Config:
    """Application configuration constants"""
    APP_NAME = "MemoryVault Pro AI"
    VERSION = "3.0.0"
    
    # Get absolute paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    AI_MODELS_DIR = BASE_DIR / "ai_models"
    
    # Files and paths
    METADATA_FILE = METADATA_DIR / "album_metadata.json"
    DB_FILE = DB_DIR / "album.db"
    
    # Image settings
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Gallery settings
    ITEMS_PER_PAGE = 20
    GRID_COLUMNS = 4
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    MAX_COMMENT_LENGTH = 500
    MAX_CAPTION_LENGTH = 200
    
    # Cache
    CACHE_TTL = 3600  # 1 hour
    
    # AI Settings
    AI_ENABLED = AI_AVAILABLE
    
    @classmethod
    def init_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.THUMBNAIL_DIR,
            cls.METADATA_DIR,
            cls.DB_DIR,
            cls.EXPORT_DIR,
            cls.AI_MODELS_DIR
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
                    print(f"Created sample image for {person}")
                except Exception as e:
                    print(f"Could not create sample image for {person}: {str(e)}")

class UserRoles(Enum):
    """User permission levels"""
    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    EDITOR = "editor"
    ADMIN = "admin"

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class ImageMetadata:
    """Metadata for a single image"""
    image_id: str
    filename: str
    filepath: str
    file_size: int
    dimensions: Tuple[int, int]
    format: str
    created_date: datetime.datetime
    modified_date: datetime.datetime
    exif_data: Optional[Dict]
    checksum: str
    
    @classmethod
    def from_image(cls, image_path: Path) -> 'ImageMetadata':
        """Create metadata from image file"""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with Image.open(image_path) as img:
            stats = image_path.stat()
            return cls(
                image_id=str(uuid.uuid4()),
                filename=image_path.name,
                filepath=str(image_path.relative_to(Config.DATA_DIR)),
                file_size=stats.st_size,
                dimensions=img.size,
                format=img.format,
                created_date=datetime.datetime.fromtimestamp(stats.st_ctime),
                modified_date=datetime.datetime.fromtimestamp(stats.st_mtime),
                exif_data=cls._extract_exif(img),
                checksum=cls._calculate_checksum(image_path)
            )
    
    @staticmethod
    def _extract_exif(img: Image.Image) -> Optional[Dict]:
        """Extract EXIF data from image"""
        try:
            exif = {}
            if hasattr(img, '_getexif') and img._getexif():
                raw_exif = img._getexif()
                for tag_id, value in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    # Skip large binary data
                    if not isinstance(value, (bytes, np.ndarray)):
                        exif[tag] = str(value)
            return exif if exif else None
        except Exception as e:
            print(f"Error extracting EXIF data: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_checksum(image_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        if not image_path.exists():
            raise FileNotFoundError(f"File not found for checksum: {image_path}")
            
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

@dataclass
class AlbumEntry:
    """Album entry with user content"""
    entry_id: str
    image_id: str
    person_id: str
    caption: str
    description: str
    location: str
    date_taken: Optional[datetime.datetime]
    tags: List[str]
    privacy_level: str
    created_by: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['date_taken'] = self.date_taken.isoformat() if self.date_taken else None
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class Comment:
    """Comment on an album entry"""
    comment_id: str
    entry_id: str
    user_id: str
    username: str
    content: str
    created_at: datetime.datetime
    is_edited: bool
    parent_comment_id: Optional[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class Rating:
    """Rating for an album entry"""
    rating_id: str
    entry_id: str
    user_id: str
    rating_value: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class PersonProfile:
    """Person profile information"""
    person_id: str
    folder_name: str
    display_name: str
    bio: str
    birth_date: Optional[datetime.date]
    relationship: str
    contact_info: str
    social_links: Dict[str, str]
    profile_image: Optional[str]
    created_at: datetime.datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['birth_date'] = self.birth_date.isoformat() if self.birth_date else None
        data['created_at'] = self.created_at.isoformat()
        return data

# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================
class DatabaseManager:
    """Manage SQLite database operations with proper error handling"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Config.DB_FILE
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except sqlite3.Error as e:
            st.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _init_database(self):
        """Initialize database with tables"""
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    created_date TIMESTAMP,
                    modified_date TIMESTAMP,
                    exif_data TEXT,
                    checksum TEXT UNIQUE,
                    thumbnail_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS people (
                    person_id TEXT PRIMARY KEY,
                    folder_name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    bio TEXT,
                    birth_date DATE,
                    relationship TEXT,
                    contact_info TEXT,
                    social_links TEXT,
                    profile_image TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS album_entries (
                    entry_id TEXT PRIMARY KEY,
                    image_id TEXT NOT NULL,
                    person_id TEXT NOT NULL,
                    caption TEXT,
                    description TEXT,
                    location TEXT,
                    date_taken TIMESTAMP,
                    tags TEXT,
                    privacy_level TEXT DEFAULT 'public',
                    created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images (image_id),
                    FOREIGN KEY (person_id) REFERENCES people (person_id)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_edited BOOLEAN DEFAULT 0,
                    parent_comment_id TEXT,
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ratings (
                    rating_id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating_value INTEGER CHECK (rating_value BETWEEN 1 AND 5),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entry_id, user_id),
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_favorites (
                    user_id TEXT,
                    entry_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, entry_id),
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_person ON album_entries(person_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_created ON album_entries(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_entry ON comments(entry_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_entry ON ratings(entry_id)')
                
                conn.commit()
                st.success("✅ Database initialized successfully!")
                
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {str(e)}")
            raise
    
    def add_image(self, metadata: ImageMetadata, thumbnail_path: str = None):
        """Add image metadata to database"""
        if not isinstance(metadata, ImageMetadata):
            raise TypeError("metadata must be an ImageMetadata instance")
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO images
                (image_id, filename, filepath, file_size, width, height, format,
                created_date, modified_date, exif_data, checksum, thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.image_id,
                    metadata.filename,
                    metadata.filepath,
                    metadata.file_size,
                    metadata.dimensions[0],
                    metadata.dimensions[1],
                    metadata.format,
                    metadata.created_date,
                    metadata.modified_date,
                    json.dumps(metadata.exif_data) if metadata.exif_data else None,
                    metadata.checksum,
                    thumbnail_path
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding image to database: {str(e)}")
            raise
    
    def get_image(self, image_id: str) -> Optional[Dict]:
        """Retrieve image metadata"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM images WHERE image_id = ?', (image_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            st.error(f"Error retrieving image: {str(e)}")
            return None
    
    def add_person(self, person: PersonProfile):
        """Add person to database"""
        if not isinstance(person, PersonProfile):
            raise TypeError("person must be a PersonProfile instance")
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO people
                (person_id, folder_name, display_name, bio, birth_date,
                relationship, contact_info, social_links, profile_image)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    person.person_id,
                    person.folder_name,
                    person.display_name,
                    person.bio,
                    person.birth_date.isoformat() if person.birth_date else None,
                    person.relationship,
                    person.contact_info,
                    json.dumps(person.social_links),
                    person.profile_image
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding person: {str(e)}")
            raise
    
    def get_all_people(self) -> List[Dict]:
        """Get all people from database"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM people ORDER BY display_name')
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving people: {str(e)}")
            return []
    
    def add_album_entry(self, entry: AlbumEntry):
        """Add album entry to database"""
        if not isinstance(entry, AlbumEntry):
            raise TypeError("entry must be an AlbumEntry instance")
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO album_entries
                (entry_id, image_id, person_id, caption, description, location,
                date_taken, tags, privacy_level, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.entry_id,
                    entry.image_id,
                    entry.person_id,
                    entry.caption,
                    entry.description,
                    entry.location,
                    entry.date_taken,
                    ','.join(entry.tags) if entry.tags else None,
                    entry.privacy_level,
                    entry.created_by
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding album entry: {str(e)}")
            raise
    
    def add_comment(self, comment: Comment):
        """Add comment to database"""
        if not isinstance(comment, Comment):
            raise TypeError("comment must be a Comment instance")
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO comments
                (comment_id, entry_id, user_id, username, content, parent_comment_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    comment.comment_id,
                    comment.entry_id,
                    comment.user_id,
                    comment.username,
                    comment.content,
                    comment.parent_comment_id
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding comment: {str(e)}")
            raise
    
    def add_rating(self, rating: Rating):
        """Add or update rating"""
        if not isinstance(rating, Rating):
            raise TypeError("rating must be a Rating instance")
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO ratings
                (rating_id, entry_id, user_id, rating_value)
                VALUES (?, ?, ?, ?)
                ''', (
                    rating.rating_id,
                    rating.entry_id,
                    rating.user_id,
                    rating.rating_value
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding rating: {str(e)}")
            raise
    
    def get_entry_comments(self, entry_id: str) -> List[Dict]:
        """Get comments for an entry"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM comments
                WHERE entry_id = ?
                ORDER BY created_at DESC
                ''', (entry_id,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving comments: {str(e)}")
            return []
    
    def get_entry_ratings(self, entry_id: str) -> Tuple[float, int]:
        """Get average rating and count for an entry"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT AVG(rating_value), COUNT(*)
                FROM ratings
                WHERE entry_id = ?
                ''', (entry_id,))
                result = cursor.fetchone()
                avg_rating = result[0] if result and result[0] is not None else 0.0
                count = result[1] if result else 0
                return (float(avg_rating), int(count))
        except sqlite3.Error as e:
            st.error(f"Error retrieving ratings: {str(e)}")
            return (0.0, 0)
    
    def search_entries(self, query: str, person_id: str = None) -> List[Dict]:
        """Search album entries"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                search_pattern = f'%{query}%'
                
                if person_id:
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.person_id = ? AND
                    (ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)
                    ORDER BY ae.created_at DESC
                    ''', (person_id, search_pattern, search_pattern, search_pattern))
                else:
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
                    ORDER BY ae.created_at DESC
                    ''', (search_pattern, search_pattern, search_pattern))
                
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error searching entries: {str(e)}")
            return []
    
    def get_entry_details(self, entry_id: str) -> Optional[Dict]:
        """Get detailed information for an entry"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT
                    ae.entry_id,
                    ae.image_id,
                    ae.person_id,
                    ae.caption,
                    ae.description,
                    ae.location,
                    ae.date_taken,
                    ae.tags,
                    ae.privacy_level,
                    ae.created_by,
                    ae.created_at,
                    ae.updated_at,
                    p.display_name,
                    p.folder_name,
                    i.filename,
                    i.filepath,
                    i.file_size,
                    i.format,
                    i.created_date,
                    i.exif_data,
                    i.thumbnail_path,
                    (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count,
                    (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                    (SELECT COUNT(*) FROM ratings r WHERE r.entry_id = ae.entry_id) as rating_count
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN images i ON ae.image_id = i.image_id
                WHERE ae.entry_id = ?
                ''', (entry_id,))
                
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    # Parse JSON fields
                    if result.get('exif_data'):
                        try:
                            result['exif_data'] = json.loads(result['exif_data'])
                        except Exception as e:
                            result['exif_data'] = {}
                            st.warning(f"Error parsing exif_data: {str(e)}")
                    # Parse tags
                    if result.get('tags'):
                        result['tags'] = [tag.strip() for tag in result['tags'].split(',') if tag.strip()]
                    else:
                        result['tags'] = []
                    return result
                return None
        except sqlite3.Error as e:
            st.error(f"Database error retrieving entry details: {str(e)}")
            return None

# ============================================================================
# IMAGE PROCESSING AND THUMBNAIL MANAGEMENT
# ============================================================================
class ImageProcessor:
    """Handle image processing operations"""
    
    @staticmethod
    def create_thumbnail(image_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        """Create thumbnail for image"""
        if not image_path.exists():
            st.warning(f"Image not found: {image_path}")
            return None
            
        thumbnail_dir = thumbnail_dir or Config.THUMBNAIL_DIR
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        thumbnail_path = thumbnail_dir / f"{image_path.stem}_thumb.jpg"
        
        try:
            with Image.open(image_path) as img:
                # Handle image rotation based on EXIF
                img = ImageOps.exif_transpose(img)
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                
                # Create thumbnail
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                
                # Save thumbnail as JPG
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                
            return thumbnail_path
        except Exception as e:
            st.error(f"Error creating thumbnail for {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_image_data_url(image_path: Path) -> str:
        """Convert image to data URL for embedding"""
        try:
            if not image_path.exists():
                return ""
                
            with open(image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                mime_type = "image/jpeg" if image_path.suffix.lower() in ['.jpg', '.jpeg'] else f"image/{image_path.suffix[1:]}"
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            st.error(f"Error encoding image {image_path}: {str(e)}")
            return ""

# ============================================================================
# AI CONFIGURATION (ONLY IF AVAILABLE)
# ============================================================================
if AI_AVAILABLE:
    class AIConfig:
        """AI configuration settings"""
        # Model paths and names
        CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
        CLIP_MODEL = "openai/clip-vit-large-patch14"
        DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
        INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
        
        # Device settings
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        PRECISION = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Cache settings
        AI_CACHE_TTL = 3600 * 24
        
        @classmethod
        def load_secrets(cls):
            """Load API keys"""
            try:
                import os
                cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
                cls.REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN") or st.secrets.get("REPLICATE_API_TOKEN", None)
            except:
                pass
    
    # ============================================================================
    # AI MODEL MANAGER
    # ============================================================================
    class AIModelManager:
        """Manage AI models with caching"""
        
        def __init__(self):
            self.model_cache = {}
            self.device = AIConfig.DEVICE
            AIConfig.load_secrets()
        
        def _get_cache_key(self, model_name, task):
            return f"{model_name}_{task}"
        
        def load_model(self, model_name, task="default"):
            """Load model with caching"""
            cache_key = self._get_cache_key(model_name, task)
            
            if cache_key in self.model_cache:
                return self.model_cache[cache_key]
            
            try:
                if model_name == "caption":
                    processor = BlipProcessor.from_pretrained(AIConfig.CAPTION_MODEL)
                    model = BlipForConditionalGeneration.from_pretrained(
                        AIConfig.CAPTION_MODEL,
                        torch_dtype=AIConfig.PRECISION
                    ).to(self.device)
                    self.model_cache[cache_key] = (processor, model)
                    
                elif model_name == "clip":
                    processor = CLIPProcessor.from_pretrained(AIConfig.CLIP_MODEL)
                    model = CLIPModel.from_pretrained(AIConfig.CLIP_MODEL).to(self.device)
                    self.model_cache[cache_key] = (processor, model)
                    
                elif model_name == "diffusion":
                    pipe = StableDiffusionPipeline.from_pretrained(
                        AIConfig.DIFFUSION_MODEL,
                        torch_dtype=AIConfig.PRECISION,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to(self.device)
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                    self.model_cache[cache_key] = pipe
                    
                elif model_name == "inpaint":
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        AIConfig.INPAINT_MODEL,
                        torch_dtype=AIConfig.PRECISION,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to(self.device)
                    self.model_cache[cache_key] = pipe
                    
                elif model_name == "img2img":
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        AIConfig.DIFFUSION_MODEL,
                        torch_dtype=AIConfig.PRECISION,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to(self.device)
                    self.model_cache[cache_key] = pipe
                    
                return self.model_cache[cache_key]
                
            except Exception as e:
                st.error(f"Error loading model {model_name}: {str(e)}")
                return None
        
        def unload_model(self, model_name, task="default"):
            """Unload model from cache"""
            cache_key = self._get_cache_key(model_name, task)
            if cache_key in self.model_cache:
                del self.model_cache[cache_key]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        def clear_all_models(self):
            """Clear all loaded models"""
            self.model_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # ============================================================================
    # LLM IMAGE ENHANCER
    # ============================================================================
    class LLMImageEnhancer:
        """LLM-based image analysis and enhancement"""
        
        def __init__(self, model_manager):
            self.model_manager = model_manager
            self._init_session_state()
        
        def _init_session_state(self):
            """Initialize session state for LLM features"""
            if 'llm_captions' not in st.session_state:
                st.session_state.llm_captions = {}
            if 'llm_tags' not in st.session_state:
                st.session_state.llm_tags = {}
            if 'llm_descriptions' not in st.session_state:
                st.session_state.llm_descriptions = {}
            if 'llm_analysis_cache' not in st.session_state:
                st.session_state.llm_analysis_cache = {}
        
        def generate_caption(self, image: Image.Image, max_length: int = 30) -> str:
            """Generate caption for image using BLIP model"""
            try:
                # Create cache key
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                cache_key = f"caption_{image_hash}_{max_length}"
                
                # Check cache
                if cache_key in st.session_state.llm_analysis_cache:
                    return st.session_state.llm_analysis_cache[cache_key]
                
                # Load model
                model_result = self.model_manager.load_model("caption")
                if model_result is None:
                    return "Could not load caption model"
                
                processor, model = model_result
                
                # Prepare image
                inputs = processor(image, return_tensors="pt").to(self.model_manager.device)
                
                # Generate caption
                with torch.no_grad():
                    out = model.generate(**inputs, max_length=max_length, num_beams=5)
                
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                # Cache result
                st.session_state.llm_analysis_cache[cache_key] = caption
                st.session_state.llm_captions[image_hash] = caption
                
                return caption
                
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")
                return "Error generating caption"
        
        def generate_detailed_description(self, image: Image.Image) -> str:
            """Generate detailed description"""
            try:
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                cache_key = f"description_{image_hash}"
                
                if cache_key in st.session_state.llm_analysis_cache:
                    return st.session_state.llm_analysis_cache[cache_key]
                
                # Get initial caption
                caption = self.generate_caption(image, max_length=50)
                
                # Use CLIP to get image features
                model_result = self.model_manager.load_model("clip")
                if model_result is None:
                    return caption
                
                processor, model = model_result
                
                inputs = processor(images=image, return_tensors="pt").to(self.model_manager.device)
                
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                
                # Generate enhanced description
                enhanced_desc = caption
                if "person" in caption.lower():
                    enhanced_desc += ". The photo shows a person in the scene."
                if "landscape" in caption.lower() or "nature" in caption.lower():
                    enhanced_desc += " This appears to be an outdoor scene with natural elements."
                
                # Cache result
                st.session_state.llm_analysis_cache[cache_key] = enhanced_desc
                st.session_state.llm_descriptions[image_hash] = enhanced_desc
                
                return enhanced_desc
                
            except Exception as e:
                st.error(f"Error generating description: {str(e)}")
                return "Error generating description"
        
        def generate_tags(self, image: Image.Image, num_tags: int = 10) -> List[str]:
            """Generate tags for image"""
            try:
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                cache_key = f"tags_{image_hash}_{num_tags}"
                
                if cache_key in st.session_state.llm_analysis_cache:
                    return st.session_state.llm_analysis_cache[cache_key]
                
                # Get caption
                caption = self.generate_caption(image)
                
                # Extract keywords from caption
                words = caption.lower().split()
                
                # Common photo tags
                common_tags = [
                    'photo', 'image', 'picture', 'photography',
                    'color', 'black and white', 'vintage', 'modern',
                    'portrait', 'landscape', 'urban', 'nature',
                    'day', 'night', 'sunset', 'sunrise',
                    'indoor', 'outdoor', 'travel', 'memory'
                ]
                
                # Generate tags based on caption
                tags = set()
                
                # Add words from caption
                for word in words:
                    if len(word) > 3 and word not in ['this', 'that', 'with', 'from']:
                        tags.add(word)
                
                # Add relevant common tags
                for common_tag in common_tags:
                    if any(word in common_tag for word in words[:5]):
                        tags.add(common_tag)
                
                # Limit number of tags
                tags_list = list(tags)[:num_tags]
                
                # Cache result
                st.session_state.llm_analysis_cache[cache_key] = tags_list
                st.session_state.llm_tags[image_hash] = tags_list
                
                return tags_list
                
            except Exception as e:
                st.error(f"Error generating tags: {str(e)}")
                return []
        
        def analyze_image_composition(self, image: Image.Image) -> Dict:
            """Analyze image composition and provide feedback"""
            try:
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                cache_key = f"composition_{image_hash}"
                
                if cache_key in st.session_state.llm_analysis_cache:
                    return st.session_state.llm_analysis_cache[cache_key]
                
                analysis = {
                    'brightness': 'optimal',
                    'contrast': 'good',
                    'sharpness': 'acceptable',
                    'color_balance': 'balanced',
                    'composition': 'standard',
                    'suggestions': []
                }
                
                # Basic image analysis
                img_array = np.array(image)
                
                # Brightness analysis
                brightness = np.mean(img_array)
                if brightness < 50:
                    analysis['brightness'] = 'dark'
                    analysis['suggestions'].append('Consider increasing brightness')
                elif brightness > 200:
                    analysis['brightness'] = 'bright'
                    analysis['suggestions'].append('Consider decreasing brightness')
                
                # Contrast analysis
                contrast = np.std(img_array)
                if contrast < 30:
                    analysis['contrast'] = 'low'
                    analysis['suggestions'].append('Consider increasing contrast')
                
                # Color balance analysis
                if len(img_array.shape) == 3:  # Color image
                    r, g, b = np.mean(img_array, axis=(0, 1))
                    if abs(r - g) > 30 or abs(r - b) > 30:
                        analysis['color_balance'] = 'unbalanced'
                        analysis['suggestions'].append('Consider adjusting color balance')
                
                # Rule of thirds analysis (simplified)
                height, width = img_array.shape[:2]
                analysis['composition'] = f"{width}x{height}"
                
                # Cache result
                st.session_state.llm_analysis_cache[cache_key] = analysis
                
                return analysis
                
            except Exception as e:
                st.error(f"Error analyzing composition: {str(e)}")
                return {}
        
        def generate_ai_caption_variants(self, image: Image.Image, num_variants: int = 3) -> List[str]:
            """Generate multiple caption variants"""
            try:
                base_caption = self.generate_caption(image)
                
                variants = [base_caption]
                
                # Create variants by modifying the base caption
                if num_variants > 1:
                    # Short version
                    short_words = base_caption.split()[:8]
                    variants.append(' '.join(short_words) + '...')
                    
                    # Creative version
                    creative = base_caption.replace('a ', 'an artistic ').replace('the ', 'the stunning ')
                    variants.append(creative)
                    
                    # Descriptive version
                    descriptive = f"A captivating photo showing {base_caption.lower()}"
                    variants.append(descriptive)
                
                return variants[:num_variants]
                
            except Exception as e:
                st.error(f"Error generating caption variants: {str(e)}")
                return ["No caption available"]
    
    # ============================================================================
    # DIFFUSION IMAGE STYLER
    # ============================================================================
    class DiffusionImageStyler:
        """Diffusion model-based image styling and modification"""
        
        def __init__(self, model_manager):
            self.model_manager = model_manager
            self._init_session_state()
        
        def _init_session_state(self):
            """Initialize session state for diffusion features"""
            if 'diffusion_results' not in st.session_state:
                st.session_state.diffusion_results = {}
            if 'style_presets' not in st.session_state:
                st.session_state.style_presets = self._get_default_presets()
            if 'generation_history' not in st.session_state:
                st.session_state.generation_history = []
            if 'current_modification' not in st.session_state:
                st.session_state.current_modification = None
        
        def _get_default_presets(self):
            """Get default style presets"""
            return {
                'realistic': {
                    'prompt_suffix': 'photorealistic, 8k, detailed, sharp focus',
                    'negative_prompt': 'blurry, cartoon, painting, drawing',
                    'guidance_scale': 7.5,
                    'num_inference_steps': 30
                },
                'painting': {
                    'prompt_suffix': 'oil painting, masterpiece, art gallery',
                    'negative_prompt': 'photograph, photo, realistic',
                    'guidance_scale': 8.0,
                    'num_inference_steps': 40
                },
                'sketch': {
                    'prompt_suffix': 'pencil sketch, drawing, black and white',
                    'negative_prompt': 'color, painting, photograph',
                    'guidance_scale': 9.0,
                    'num_inference_steps': 25
                },
                'fantasy': {
                    'prompt_suffix': 'fantasy art, magical, dreamlike, surreal',
                    'negative_prompt': 'realistic, ordinary, mundane',
                    'guidance_scale': 10.0,
                    'num_inference_steps': 50
                },
                'vintage': {
                    'prompt_suffix': 'vintage photograph, 1950s, retro style',
                    'negative_prompt': 'modern, digital, hd',
                    'guidance_scale': 7.0,
                    'num_inference_steps': 35
                }
            }
        
        def apply_style_transfer(self, image: Image.Image, style_name: str, 
                               strength: float = 0.7) -> Optional[Image.Image]:
            """Apply style transfer using diffusion"""
            try:
                # Create cache key
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                cache_key = f"style_{image_hash}_{style_name}_{strength}"
                
                # Check cache
                if cache_key in st.session_state.diffusion_results:
                    return st.session_state.diffusion_results[cache_key]
                
                # Get style preset
                if style_name not in st.session_state.style_presets:
                    st.error(f"Style '{style_name}' not found")
                    return None
                
                preset = st.session_state.style_presets[style_name]
                
                # Generate prompt from image
                llm_enhancer = LLMImageEnhancer(self.model_manager)
                caption = llm_enhancer.generate_caption(image)
                
                # Create final prompt
                base_prompt = f"{caption}, {preset['prompt_suffix']}"
                
                # Load img2img pipeline
                pipe = self.model_manager.load_model("img2img")
                if pipe is None:
                    return None
                
                # Prepare image
                init_image = image.convert("RGB").resize((512, 512))
                
                # Generate
                with torch.autocast(self.model_manager.device):
                    result = pipe(
                        prompt=base_prompt,
                        negative_prompt=preset['negative_prompt'],
                        image=init_image,
                        strength=strength,
                        guidance_scale=preset['guidance_scale'],
                        num_inference_steps=preset['num_inference_steps']
                    ).images[0]
                
                # Resize back to original dimensions
                result = result.resize(image.size)
                
                # Cache result
                st.session_state.diffusion_results[cache_key] = result
                st.session_state.generation_history.append({
                    'type': 'style_transfer',
                    'style': style_name,
                    'original_hash': image_hash,
                    'timestamp': datetime.datetime.now()
                })
                
                return result
                
            except Exception as e:
                st.error(f"Error applying style transfer: {str(e)}")
                return None
        
        def text_guided_edit(self, image: Image.Image, prompt: str, 
                            negative_prompt: str = "", strength: float = 0.6) -> Optional[Image.Image]:
            """Edit image based on text prompt"""
            try:
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                cache_key = f"edit_{image_hash}_{hash(prompt)}_{strength}"
                
                if cache_key in st.session_state.diffusion_results:
                    return st.session_state.diffusion_results[cache_key]
                
                # Load img2img pipeline
                pipe = self.model_manager.load_model("img2img")
                if pipe is None:
                    return None
                
                # Prepare image
                init_image = image.convert("RGB").resize((512, 512))
                
                # Generate
                with torch.autocast(self.model_manager.device):
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=init_image,
                        strength=strength,
                        guidance_scale=7.5,
                        num_inference_steps=30
                    ).images[0]
                
                result = result.resize(image.size)
                
                # Cache result
                st.session_state.diffusion_results[cache_key] = result
                st.session_state.generation_history.append({
                    'type': 'text_edit',
                    'prompt': prompt,
                    'original_hash': image_hash,
                    'timestamp': datetime.datetime.now()
                })
                
                return result
                
            except Exception as e:
                st.error(f"Error in text-guided edit: {str(e)}")
                return None
        
        def generate_variations(self, image: Image.Image, num_variations: int = 4,
                              variation_strength: float = 0.3) -> List[Image.Image]:
            """Generate variations of an image"""
            try:
                image_hash = hashlib.md5(image.tobytes()).hexdigest()
                cache_key = f"variations_{image_hash}_{num_variations}_{variation_strength}"
                
                if cache_key in st.session_state.diffusion_results:
                    return st.session_state.diffusion_results[cache_key]
                
                # Load model
                pipe = self.model_manager.load_model("img2img")
                if pipe is None:
                    return []
                
                # Generate caption for prompt
                llm_enhancer = LLMImageEnhancer(self.model_manager)
                caption = llm_enhancer.generate_caption(image)
                
                # Prepare image
                init_image = image.convert("RGB").resize((512, 512))
                
                variations = []
                
                # Generate variations
                with torch.autocast(self.model_manager.device):
                    for i in range(num_variations):
                        # Add slight variation to prompt
                        variation_prompt = f"{caption}, variation {i+1}"
                        
                        result = pipe(
                            prompt=variation_prompt,
                            image=init_image,
                            strength=variation_strength,
                            guidance_scale=7.0,
                            num_inference_steps=25,
                            generator=torch.Generator(device=self.model_manager.device).manual_seed(i)
                        ).images[0]
                        
                        result = result.resize(image.size)
                        variations.append(result)
                
                # Cache result
                st.session_state.diffusion_results[cache_key] = variations
                st.session_state.generation_history.append({
                    'type': 'variations',
                    'count': num_variations,
                    'original_hash': image_hash,
                    'timestamp': datetime.datetime.now()
                })
                
                return variations
                
            except Exception as e:
                st.error(f"Error generating variations: {str(e)}")
                return []
        
        def upscale_image(self, image: Image.Image, scale_factor: float = 2.0) -> Image.Image:
            """Upscale image"""
            try:
                # Simple upscaling for demo
                new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
                return image.resize(new_size, Image.Resampling.LANCZOS)
                
            except Exception as e:
                st.error(f"Error upscaling image: {str(e)}")
                return image
        
        def create_custom_style(self, name: str, prompt_suffix: str, 
                              negative_prompt: str = "", guidance_scale: float = 7.5,
                              num_steps: int = 30):
            """Create custom style preset"""
            st.session_state.style_presets[name] = {
                'prompt_suffix': prompt_suffix,
                'negative_prompt': negative_prompt,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_steps
            }
    
    # ============================================================================
    # AI IMAGE PROCESSOR
    # ============================================================================
    class AIImageProcessor:
        """Extended image processor with AI capabilities"""
        
        def __init__(self):
            self.image_processor = ImageProcessor()
            self.model_manager = AIModelManager()
            self.llm_enhancer = LLMImageEnhancer(self.model_manager)
            self.diffusion_styler = DiffusionImageStyler(self.model_manager)
        
        def create_thumbnail(self, image_path, thumbnail_dir=None):
            """Delegate to original processor"""
            return self.image_processor.create_thumbnail(image_path, thumbnail_dir)
        
        def get_image_data_url(self, image_path):
            """Delegate to original processor"""
            return self.image_processor.get_image_data_url(image_path)
        
        @staticmethod
        def apply_basic_filters(image: Image.Image, filter_type: str, intensity: float = 1.0) -> Image.Image:
            """Apply basic image filters"""
            try:
                if filter_type == 'blur':
                    return image.filter(ImageFilter.GaussianBlur(radius=intensity))
                elif filter_type == 'sharpen':
                    return image.filter(ImageFilter.SHARPEN)
                elif filter_type == 'edge_enhance':
                    return image.filter(ImageFilter.EDGE_ENHANCE)
                elif filter_type == 'contrast':
                    enhancer = ImageEnhance.Contrast(image)
                    return enhancer.enhance(intensity)
                elif filter_type == 'brightness':
                    enhancer = ImageEnhance.Brightness(image)
                    return enhancer.enhance(intensity)
                elif filter_type == 'saturation':
                    enhancer = ImageEnhance.Color(image)
                    return enhancer.enhance(intensity)
                elif filter_type == 'grayscale':
                    return image.convert('L').convert('RGB')
                elif filter_type == 'sepia':
                    # Apply sepia tone
                    sepia_filter = np.array([
                        [0.393, 0.769, 0.189],
                        [0.349, 0.686, 0.168],
                        [0.272, 0.534, 0.131]
                    ])
                    img_array = np.array(image)
                    sepia_img = np.dot(img_array[..., :3], sepia_filter.T)
                    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
                    return Image.fromarray(sepia_img)
                else:
                    return image
                    
            except Exception as e:
                st.error(f"Error applying filter {filter_type}: {str(e)}")
                return image
        
        def analyze_image_with_llm(self, image: Image.Image) -> Dict:
            """Comprehensive LLM analysis of image"""
            analysis = {}
            
            # Generate caption
            analysis['caption'] = self.llm_enhancer.generate_caption(image)
            
            # Generate description
            analysis['description'] = self.llm_enhancer.generate_detailed_description(image)
            
            # Generate tags
            analysis['tags'] = self.llm_enhancer.generate_tags(image)
            
            # Analyze composition
            analysis['composition'] = self.llm_enhancer.analyze_image_composition(image)
            
            # Generate caption variants
            analysis['caption_variants'] = self.llm_enhancer.generate_ai_caption_variants(image)
            
            return analysis
        
        def apply_ai_style(self, image: Image.Image, style_type: str, **kwargs) -> Optional[Image.Image]:
            """Apply AI-based style to image"""
            return self.diffusion_styler.apply_style_transfer(image, style_type, **kwargs)
        
        def edit_with_prompt(self, image: Image.Image, prompt: str, **kwargs) -> Optional[Image.Image]:
            """Edit image using text prompt"""
            return self.diffusion_styler.text_guided_edit(image, prompt, **kwargs)
        
        def generate_variations(self, image: Image.Image, num_variations: int = 4) -> List[Image.Image]:
            """Generate AI variations of image"""
            return self.diffusion_styler.generate_variations(image, num_variations)
        
        def auto_enhance_image(self, image: Image.Image) -> Image.Image:
            """Auto-enhance image using AI analysis"""
            try:
                # Analyze image
                analysis = self.analyze_image_with_llm(image)
                
                # Apply enhancements based on analysis
                enhanced = image.copy()
                
                if analysis['composition'].get('brightness') == 'dark':
                    enhanced = self.apply_basic_filters(enhanced, 'brightness', 1.3)
                
                if analysis['composition'].get('contrast') == 'low':
                    enhanced = self.apply_basic_filters(enhanced, 'contrast', 1.2)
                
                if analysis['composition'].get('sharpness') == 'acceptable':
                    enhanced = self.apply_basic_filters(enhanced, 'sharpen')
                
                return enhanced
                
            except Exception as e:
                st.error(f"Error in auto-enhancement: {str(e)}")
                return image

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================
class CacheManager:
    """Manage application caching"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str, default=None):
        """Get cached value if not expired"""
        if key in self._cache:
            if time.time() - self._timestamps[key] < Config.CACHE_TTL:
                return self._cache[key]
            else:
                del self._cache[key]
                del self._timestamps[key]
        return default
    
    def set(self, key: str, value):
        """Set cache value"""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self, key: str = None):
        """Clear cache or specific key"""
        if key:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
        else:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_or_set(self, key: str, generator_func):
        """Get cached value or generate and cache it"""
        cached = self.get(key)
        if cached is not None:
            return cached
        try:
            value = generator_func()
            self.set(key, value)
            return value
        except Exception as e:
            st.error(f"Error generating cached value for {key}: {str(e)}")
            return None

# ============================================================================
# UI COMPONENTS
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

# ============================================================================
# ENHANCED UI COMPONENTS WITH AI FEATURES
# ============================================================================
if AI_AVAILABLE:
    class EnhancedUIComponents(UIComponents):
        """Enhanced UI components with AI features"""
        
        @staticmethod
        def ai_analysis_panel(analysis: Dict):
            """Display AI analysis results"""
            with st.expander("🤖 AI Analysis", expanded=True):
                if analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📝 Caption**")
                        st.info(analysis.get('caption', 'No caption generated'))
                        
                        st.markdown("**🏷️ Suggested Tags**")
                        tags = analysis.get('tags', [])
                        if tags:
                            st.markdown(UIComponents.tag_badges(tags[:10]), unsafe_allow_html=True)
                        else:
                            st.text("No tags generated")
                    
                    with col2:
                        st.markdown("**📊 Composition Analysis**")
                        comp = analysis.get('composition', {})
                        if comp:
                            cols = st.columns(2)
                            with cols[0]:
                                st.metric("Brightness", comp.get('brightness', 'N/A'))
                                st.metric("Contrast", comp.get('contrast', 'N/A'))
                            with cols[1]:
                                st.metric("Sharpness", comp.get('sharpness', 'N/A'))
                                st.metric("Color Balance", comp.get('color_balance', 'N/A'))
                        
                        # Suggestions
                        suggestions = comp.get('suggestions', [])
                        if suggestions:
                            st.markdown("**💡 Suggestions**")
                            for suggestion in suggestions:
                                st.text(f"• {suggestion}")
                    
                    # Caption variants
                    variants = analysis.get('caption_variants', [])
                    if len(variants) > 1:
                        st.markdown("**🔄 Caption Variants**")
                        for i, variant in enumerate(variants[:3]):
                            st.text_area(f"Variant {i+1}", variant, height=50, key=f"variant_{i}")
        
        @staticmethod
        def ai_modification_interface(image_path: Path, album_manager):
            """Interface for AI image modifications"""
            st.subheader("✨ AI Image Modifications")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Style Transfer", "Text Edit", "Auto Enhance", "Batch Process"])
            
            with tab1:
                st.markdown("**🎨 Apply Artistic Style**")
                
                # Style selection
                style_presets = st.session_state.get('style_presets', {})
                style_names = list(style_presets.keys())
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_style = st.selectbox("Select Style", style_names, key="style_select")
                    strength = st.slider("Style Strength", 0.1, 1.0, 0.7, 0.1, key="style_strength")
                
                with col2:
                    if selected_style and style_presets.get(selected_style):
                        preset = style_presets[selected_style]
                        st.markdown(f"**Preset Details:**")
                        st.text(f"Prompt: {preset.get('prompt_suffix', '')}")
                        st.text(f"Guidance: {preset.get('guidance_scale', 7.5)}")
                
                if st.button("Apply Style Transfer", key="apply_style"):
                    with st.spinner(f"Applying {selected_style} style..."):
                        result_path = album_manager.apply_ai_modification(
                            image_path, 
                            'style_transfer',
                            style=selected_style,
                            strength=strength
                        )
                        
                        if result_path:
                            st.success(f"Style applied! Saved as {result_path.name}")
                            st.image(str(result_path), caption=f"{selected_style.capitalize()} Style", use_column_width=True)
                            
                            # Offer download
                            with open(result_path, 'rb') as f:
                                st.download_button(
                                    label="Download Modified Image",
                                    data=f,
                                    file_name=result_path.name,
                                    mime="image/jpeg"
                                )
            
            with tab2:
                st.markdown("**📝 Edit with Text Prompt**")
                
                prompt = st.text_area(
                    "Describe the changes you want",
                    placeholder="e.g., 'make it look like a painting', 'add more vibrant colors', 'convert to black and white with color highlights'",
                    height=100,
                    key="edit_prompt"
                )
                
                negative_prompt = st.text_input(
                    "What to avoid",
                    placeholder="e.g., 'blurry, distorted, unrealistic'",
                    key="negative_prompt"
                )
                
                strength = st.slider("Edit Strength", 0.1, 0.9, 0.6, 0.1, key="edit_strength")
                
                if st.button("Apply Edit", key="apply_edit") and prompt:
                    with st.spinner("Applying text-guided edit..."):
                        result_path = album_manager.apply_ai_modification(
                            image_path,
                            'text_edit',
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            strength=strength
                        )
                        
                        if result_path:
                            st.success("Edit applied successfully!")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(str(image_path), caption="Original", use_column_width=True)
                            with col2:
                                st.image(str(result_path), caption="Edited", use_column_width=True)
            
            with tab3:
                st.markdown("**⚡ Auto Enhance**")
                st.markdown("""
                Automatically enhances your image based on AI analysis:
                - Adjusts brightness and contrast
                - Improves sharpness
                - Optimizes colors
                """)
                
                if st.button("Run Auto Enhancement", key="auto_enhance"):
                    with st.spinner("Analyzing and enhancing image..."):
                        # First analyze
                        analysis = album_manager.analyze_image_with_ai(image_path)
                        
                        # Then enhance
                        result_path = album_manager.apply_ai_modification(
                            image_path,
                            'auto_enhance'
                        )
                        
                        if result_path:
                            st.success("Enhancement complete!")
                            
                            # Show comparison
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(str(image_path), caption="Original", use_column_width=True)
                            with col2:
                                st.image(str(result_path), caption="Enhanced", use_column_width=True)
                            
                            # Show what was improved
                            if analysis.get('composition'):
                                st.markdown("**📊 Improvements Applied:**")
                                comp = analysis['composition']
                                improvements = []
                                
                                if comp.get('brightness') == 'dark':
                                    improvements.append("• Increased brightness")
                                if comp.get('contrast') == 'low':
                                    improvements.append("• Enhanced contrast")
                                if comp.get('sharpness') == 'acceptable':
                                    improvements.append("• Improved sharpness")
                                
                                if improvements:
                                    for imp in improvements:
                                        st.text(imp)
                                else:
                                    st.text("Image was already well balanced")
            
            with tab4:
                st.markdown("**🔀 Batch Processing**")
                
                # Get all images in the same directory
                image_dir = image_path.parent
                all_images = [
                    f for f in image_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS
                ]
                
                if len(all_images) > 1:
                    st.markdown(f"Found {len(all_images)} images in this folder")
                    
                    # Select images for batch processing
                    selected_images = st.multiselect(
                        "Select images for batch processing",
                        [img.name for img in all_images],
                        default=[image_path.name],
                        key="batch_images"
                    )
                    
                    # Select operation
                    operation = st.selectbox(
                        "Operation",
                        ["Analyze All", "Auto Enhance All", "Apply Style to All"],
                        key="batch_operation"
                    )
                    
                    if operation == "Apply Style to All":
                        batch_style = st.selectbox("Style", list(style_presets.keys()), key="batch_style")
                    
                    if st.button("Run Batch Processing", key="run_batch"):
                        # Process selected images
                        selected_paths = [image_dir / name for name in selected_images]
                        
                        with st.spinner(f"Processing {len(selected_paths)} images..."):
                            if operation == "Analyze All":
                                results = album_manager.batch_analyze_images(selected_paths)
                                
                                # Display summary
                                st.success(f"Analyzed {len(results)} images")
                                
                                # Show some insights
                                all_tags = []
                                for result in results.values():
                                    if isinstance(result, dict):
                                        tags = result.get('tags', [])
                                        all_tags.extend(tags)
                                
                                if all_tags:
                                    from collections import Counter
                                    tag_counts = Counter(all_tags)
                                    st.markdown("**Most Common Tags:**")
                                    for tag, count in tag_counts.most_common(10):
                                        st.text(f"{tag}: {count} images")
                            
                            elif operation == "Auto Enhance All":
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                enhanced_count = 0
                                for i, img_path in enumerate(selected_paths):
                                    status_text.text(f"Processing {img_path.name}...")
                                    result = album_manager.apply_ai_modification(img_path, 'auto_enhance')
                                    if result:
                                        enhanced_count += 1
                                    
                                    progress = (i + 1) / len(selected_paths)
                                    progress_bar.progress(progress)
                                
                                progress_bar.empty()
                                status_text.empty()
                                st.success(f"Enhanced {enhanced_count} of {len(selected_paths)} images")
                            
                            elif operation == "Apply Style to All":
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                styled_count = 0
                                for i, img_path in enumerate(selected_paths):
                                    status_text.text(f"Applying {batch_style} style to {img_path.name}...")
                                    result = album_manager.apply_ai_modification(
                                        img_path,
                                        'style_transfer',
                                        style=batch_style
                                    )
                                    if result:
                                        styled_count += 1
                                    
                                    progress = (i + 1) / len(selected_paths)
                                    progress_bar.progress(progress)
                                
                                progress_bar.empty()
                                status_text.empty()
                                st.success(f"Applied {batch_style} style to {styled_count} images")
                else:
                    st.info("No other images found in this folder for batch processing")

# ============================================================================
# ALBUM MANAGER
# ============================================================================
class AlbumManager:
    """Main album management class with robust error handling"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.image_processor = ImageProcessor()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'initialized': True,
                'current_page': 1,
                'selected_person': None,
                'selected_image': None,
                'search_query': '',
                'view_mode': 'grid',
                'sort_by': 'date',
                'sort_order': 'desc',
                'selected_tags': [],
                'user_id': str(uuid.uuid4()),
                'username': 'Guest',
                'user_role': UserRoles.VIEWER.value,
                'favorites': set(),
                'recently_viewed': [],
                'toc_page': 1,
                'gallery_page': 1,
                'show_directory_info': True
            })
    
    def scan_directory(self, data_dir: Path = None) -> Dict:
        """Scan directory for images and update database"""
        data_dir = data_dir or Config.DATA_DIR
        results = {
            'total_images': 0,
            'new_images': 0,
            'updated_images': 0,
            'people_found': 0,
            'errors': []
        }
        
        try:
            if not data_dir.exists():
                data_dir.mkdir(parents=True)
                st.warning(f"Created directory: {data_dir}")
                return results
            
            # Find all person directories
            person_dirs = [d for d in data_dir.iterdir() 
                          if d.is_dir() and not d.name.startswith('.') 
                          and d.name != 'thumbnails']
            
            if not person_dirs:
                st.warning("No person folders found. Creating sample structure...")
                Config.create_sample_structure()
                person_dirs = [d for d in data_dir.iterdir() 
                              if d.is_dir() and not d.name.startswith('.') 
                              and d.name != 'thumbnails']
            
            results['people_found'] = len(person_dirs)
            
            progress_bar = st.progress(0)
            total_files = 0
            processed_files = 0
            
            # Count total files
            for person_dir in person_dirs:
                image_files = [
                    f for f in person_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS
                ]
                total_files += len(image_files)
            
            if total_files == 0:
                st.info("No images found in any folders. Please add some images to your person folders.")
                progress_bar.empty()
                return results
            
            for person_dir in person_dirs:
                # Handle person name formatting flexibly
                if '-' in person_dir.name:
                    display_name = ' '.join(part.capitalize() for part in person_dir.name.split('-'))
                elif '_' in person_dir.name:
                    display_name = ' '.join(part.capitalize() for part in person_dir.name.split('_'))
                else:
                    display_name = ' '.join(word.capitalize() for word in person_dir.name.split())
                
                # Create or update person profile
                existing_person = None
                for p in self.db.get_all_people():
                    if p['folder_name'] == person_dir.name:
                        existing_person = p
                        break
                
                if not existing_person:
                    person_profile = PersonProfile(
                        person_id=str(uuid.uuid4()),
                        folder_name=person_dir.name,
                        display_name=display_name,
                        bio=f"Photos of {display_name}",
                        birth_date=None,
                        relationship="Family/Friend",
                        contact_info="",
                        social_links={},
                        profile_image=None,
                        created_at=datetime.datetime.now()
                    )
                    self.db.add_person(person_profile)
                    person_id = person_profile.person_id
                else:
                    person_id = existing_person['person_id']
                
                # Process images
                image_files = [
                    f for f in person_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS
                ]
                
                for img_path in image_files:
                    try:
                        processed_files += 1
                        progress = processed_files / max(total_files, 1)
                        progress_bar.progress(progress)
                        
                        # Calculate checksum
                        checksum = ImageMetadata._calculate_checksum(img_path)
                        
                        # Check if image exists
                        with sqlite3.connect(self.db.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                'SELECT image_id FROM images WHERE checksum = ?',
                                (checksum,)
                            )
                            existing = cursor.fetchone()
                        
                        if existing:
                            results['updated_images'] += 1
                            continue
                        
                        # Create thumbnail
                        thumbnail_path = self.image_processor.create_thumbnail(img_path)
                        thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
                        
                        # Extract metadata
                        try:
                            metadata = ImageMetadata.from_image(img_path)
                        except Exception as e:
                            results['errors'].append(f"Error processing {img_path.name}: {str(e)}")
                            continue
                        
                        # Add to database
                        self.db.add_image(metadata, thumbnail_path_str)
                        
                        # Create album entry
                        album_entry = AlbumEntry(
                            entry_id=str(uuid.uuid4()),
                            image_id=metadata.image_id,
                            person_id=person_id,
                            caption=img_path.stem.replace('_', ' ').title(),
                            description=f"Photo of {display_name}",
                            location="",
                            date_taken=metadata.created_date,
                            tags=[display_name.lower().replace(' ', '-'), 'photo', 'memory'],
                            privacy_level='public',
                            created_by='system',
                            created_at=datetime.datetime.now(),
                            updated_at=datetime.datetime.now()
                        )
                        self.db.add_album_entry(album_entry)
                        
                        results['new_images'] += 1
                        results['total_images'] += 1
                        
                    except Exception as e:
                        error_msg = f"Error processing {img_path}: {str(e)}"
                        results['errors'].append(error_msg)
                        st.error(error_msg)
            
            progress_bar.empty()
            self.cache.clear()
            return results
            
        except Exception as e:
            st.error(f"Critical error during directory scan: {str(e)}")
            results['errors'].append(f"Critical error: {str(e)}")
            return results
    
    def get_person_stats(self, person_id: str) -> Dict:
        """Get statistics for a person with error handling"""
        cache_key = f"person_stats_{person_id}"
        
        def generate_stats():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('SELECT COUNT(*) FROM album_entries WHERE person_id = ?', (person_id,))
                    image_count = cursor.fetchone()[0]
                    
                    cursor.execute('''
                    SELECT COUNT(DISTINCT c.comment_id)
                    FROM comments c
                    JOIN album_entries ae ON c.entry_id = ae.entry_id
                    WHERE ae.person_id = ?
                    ''', (person_id,))
                    comment_count = cursor.fetchone()[0]
                    
                    cursor.execute('''
                    SELECT AVG(r.rating_value)
                    FROM ratings r
                    JOIN album_entries ae ON r.entry_id = ae.entry_id
                    WHERE ae.person_id = ?
                    ''', (person_id,))
                    avg_rating = cursor.fetchone()[0] or 0.0
                    
                    cursor.execute('SELECT MAX(ae.created_at) FROM album_entries ae WHERE ae.person_id = ?', (person_id,))
                    last_activity = cursor.fetchone()[0]
                    
                    return {
                        'image_count': image_count,
                        'comment_count': comment_count,
                        'avg_rating': float(avg_rating),
                        'last_activity': last_activity
                    }
            except sqlite3.Error as e:
                st.error(f"Error getting person stats: {str(e)}")
                return {
                    'image_count': 0,
                    'comment_count': 0,
                    'avg_rating': 0.0,
                    'last_activity': None
                }
        
        return self.cache.get_or_set(cache_key, generate_stats)
    
    def add_to_favorites(self, entry_id: str):
        """Add entry to user favorites"""
        user_id = st.session_state['user_id']
        
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR IGNORE INTO user_favorites (user_id, entry_id)
                VALUES (?, ?)
                ''', (user_id, entry_id))
                conn.commit()
            
            if 'favorites' not in st.session_state:
                st.session_state.favorites = set()
            st.session_state.favorites.add(entry_id)
        except sqlite3.Error as e:
            st.error(f"Error adding to favorites: {str(e)}")
    
    def remove_from_favorites(self, entry_id: str):
        """Remove entry from user favorites"""
        user_id = st.session_state['user_id']
        
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                DELETE FROM user_favorites WHERE user_id = ? AND entry_id = ?
                ''', (user_id, entry_id))
                conn.commit()
            
            if 'favorites' in st.session_state and entry_id in st.session_state.favorites:
                st.session_state.favorites.remove(entry_id)
        except sqlite3.Error as e:
            st.error(f"Error removing from favorites: {str(e)}")
    
    def get_user_favorites(self) -> List[Dict]:
        """Get user's favorite entries"""
        user_id = st.session_state['user_id']
        
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                SELECT ae.*, p.display_name, i.filename, i.thumbnail_path
                FROM user_favorites uf
                JOIN album_entries ae ON uf.entry_id = ae.entry_id
                JOIN people p ON ae.person_id = p.person_id
                JOIN images i ON ae.image_id = i.image_id
                WHERE uf.user_id = ?
                ORDER BY uf.created_at DESC
                ''', (user_id,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving favorites: {str(e)}")
            return []
    
    def add_comment_to_entry(self, entry_id: str, content: str, parent_comment_id: str = None):
        """Add comment to an entry"""
        if not content or len(content.strip()) == 0:
            st.warning("Comment cannot be empty")
            return False
        
        if len(content) > Config.MAX_COMMENT_LENGTH:
            st.warning(f"Comment is too long. Maximum length is {Config.MAX_COMMENT_LENGTH} characters.")
            return False
        
        try:
            comment = Comment(
                comment_id=str(uuid.uuid4()),
                entry_id=entry_id,
                user_id=st.session_state['user_id'],
                username=st.session_state['username'],
                content=content.strip(),
                created_at=datetime.datetime.now(),
                is_edited=False,
                parent_comment_id=parent_comment_id
            )
            
            self.db.add_comment(comment)
            
            # Clear cache for this entry's comments
            cache_key = f"comments_{entry_id}"
            self.cache.clear(cache_key)
            
            st.success("Comment added successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error adding comment: {str(e)}")
            return False
    
    def add_rating_to_entry(self, entry_id: str, rating_value: int):
        """Add or update rating for an entry"""
        if rating_value < 1 or rating_value > 5:
            st.warning("Rating must be between 1 and 5")
            return False
        
        try:
            rating = Rating(
                rating_id=str(uuid.uuid4()),
                entry_id=entry_id,
                user_id=st.session_state['user_id'],
                rating_value=rating_value,
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now()
            )
            
            self.db.add_rating(rating)
            
            # Clear cache for this entry's ratings
            cache_key = f"ratings_{entry_id}"
            self.cache.clear(cache_key)
            
            st.success(f"Rating of {rating_value} stars added!")
            return True
            
        except Exception as e:
            st.error(f"Error adding rating: {str(e)}")
            return False
    
    def get_all_people_with_stats(self) -> List[Dict]:
        """Get all people with their statistics"""
        cache_key = "all_people_with_stats"
        
        def generate_data():
            try:
                people = self.db.get_all_people()
                result = []
                
                for person in people:
                    stats = self.get_person_stats(person['person_id'])
                    person_with_stats = {**person, **stats}
                    
                    # Get profile image if exists
                    if person.get('profile_image'):
                        profile_image_path = Config.DATA_DIR / person['folder_name'] / person['profile_image']
                        if profile_image_path.exists():
                            person_with_stats['profile_image_data'] = self.image_processor.get_image_data_url(profile_image_path)
                    
                    result.append(person_with_stats)
                
                return result
            except Exception as e:
                st.error(f"Error getting people with stats: {str(e)}")
                return []
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def get_entries_by_person(self, person_id: str, page: int = 1, search_query: str = None) -> Dict:
        """Get entries for a specific person with pagination"""
        cache_key = f"entries_person_{person_id}_page_{page}_query_{search_query}"
        
        def generate_data():
            try:
                offset = (page - 1) * Config.ITEMS_PER_PAGE
                
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    if search_query:
                        search_pattern = f'%{search_query}%'
                        cursor.execute('''
                        SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                               (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                               (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count
                        FROM album_entries ae
                        JOIN people p ON ae.person_id = p.person_id
                        JOIN images i ON ae.image_id = i.image_id
                        WHERE ae.person_id = ? AND
                        (ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)
                        ORDER BY ae.created_at DESC
                        LIMIT ? OFFSET ?
                        ''', (person_id, search_pattern, search_pattern, search_pattern, Config.ITEMS_PER_PAGE, offset))
                    else:
                        cursor.execute('''
                        SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                               (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                               (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count
                        FROM album_entries ae
                        JOIN people p ON ae.person_id = p.person_id
                        JOIN images i ON ae.image_id = i.image_id
                        WHERE ae.person_id = ?
                        ORDER BY ae.created_at DESC
                        LIMIT ? OFFSET ?
                        ''', (person_id, Config.ITEMS_PER_PAGE, offset))
                    
                    entries = [dict(row) for row in cursor.fetchall()]
                    
                    # Get total count
                    if search_query:
                        search_pattern = f'%{search_query}%'
                        cursor.execute('''
                        SELECT COUNT(*)
                        FROM album_entries ae
                        WHERE ae.person_id = ? AND
                        (ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)
                        ''', (person_id, search_pattern, search_pattern, search_pattern))
                    else:
                        cursor.execute('SELECT COUNT(*) FROM album_entries WHERE person_id = ?', (person_id,))
                    
                    total_count = cursor.fetchone()[0]
                    total_pages = max(1, math.ceil(total_count / Config.ITEMS_PER_PAGE))
                    
                    return {
                        'entries': entries,
                        'total_count': total_count,
                        'total_pages': total_pages,
                        'current_page': page
                    }
                    
            except sqlite3.Error as e:
                st.error(f"Error getting entries by person: {str(e)}")
                return {
                    'entries': [],
                    'total_count': 0,
                    'total_pages': 1,
                    'current_page': page
                }
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def get_recent_entries(self, limit: int = 10) -> List[Dict]:
        """Get most recent entries"""
        cache_key = f"recent_entries_{limit}"
        
        def generate_data():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                           (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    ORDER BY ae.created_at DESC
                    LIMIT ?
                    ''', (limit,))
                    
                    return [dict(row) for row in cursor.fetchall()]
            except sqlite3.Error as e:
                st.error(f"Error getting recent entries: {str(e)}")
                return []
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def get_top_rated_entries(self, limit: int = 10) -> List[Dict]:
        """Get top rated entries"""
        cache_key = f"top_rated_entries_{limit}"
        
        def generate_data():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                           (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                           (SELECT COUNT(*) FROM ratings r WHERE r.entry_id = ae.entry_id) as rating_count
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.entry_id IN (
                        SELECT r.entry_id
                        FROM ratings r
                        GROUP BY r.entry_id
                        HAVING AVG(r.rating_value) >= 4.0
                        ORDER BY AVG(r.rating_value) DESC
                    )
                    ORDER BY avg_rating DESC
                    LIMIT ?
                    ''', (limit,))
                    
                    return [dict(row) for row in cursor.fetchall()]
            except sqlite3.Error as e:
                st.error(f"Error getting top rated entries: {str(e)}")
                return []
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def get_entry_with_details(self, entry_id: str) -> Optional[Dict]:
        """Get complete entry details including comments and ratings"""
        cache_key = f"entry_details_{entry_id}"
        
        def generate_data():
            try:
                entry = self.db.get_entry_details(entry_id)
                if not entry:
                    return None
                
                # Get comments
                comments = self.db.get_entry_comments(entry_id)
                entry['comments'] = comments
                
                # Get ratings
                avg_rating, rating_count = self.db.get_entry_ratings(entry_id)
                entry['avg_rating'] = avg_rating
                entry['rating_count'] = rating_count
                
                # Get image path
                if entry.get('filepath'):
                    image_path = Config.DATA_DIR / entry['filepath']
                    if image_path.exists():
                        entry['image_data_url'] = self.image_processor.get_image_data_url(image_path)
                
                # Get thumbnail
                if entry.get('thumbnail_path'):
                    thumbnail_path = Path(entry['thumbnail_path'])
                    if thumbnail_path.exists():
                        entry['thumbnail_data_url'] = self.image_processor.get_image_data_url(thumbnail_path)
                
                # Check if user has favorited
                user_id = st.session_state['user_id']
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT 1 FROM user_favorites WHERE user_id = ? AND entry_id = ?
                    ''', (user_id, entry_id))
                    entry['is_favorited'] = cursor.fetchone() is not None
                
                # Check user rating
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT rating_value FROM ratings WHERE user_id = ? AND entry_id = ?
                    ''', (user_id, entry_id))
                    user_rating = cursor.fetchone()
                    entry['user_rating'] = user_rating[0] if user_rating else 0
                
                return entry
                
            except Exception as e:
                st.error(f"Error getting entry details: {str(e)}")
                return None
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def update_entry(self, entry_id: str, updates: Dict) -> bool:
        """Update album entry"""
        try:
            # Get current entry
            entry = self.get_entry_with_details(entry_id)
            if not entry:
                st.error("Entry not found")
                return False
            
            # Prepare update values
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Build update query
                set_clauses = []
                values = []
                
                if 'caption' in updates:
                    set_clauses.append("caption = ?")
                    values.append(updates['caption'][:Config.MAX_CAPTION_LENGTH])
                
                if 'description' in updates:
                    set_clauses.append("description = ?")
                    values.append(updates['description'])
                
                if 'location' in updates:
                    set_clauses.append("location = ?")
                    values.append(updates['location'])
                
                if 'tags' in updates:
                    if isinstance(updates['tags'], list):
                        tags_str = ','.join(updates['tags'])
                    else:
                        tags_str = updates['tags']
                    set_clauses.append("tags = ?")
                    values.append(tags_str)
                
                if 'privacy_level' in updates:
                    set_clauses.append("privacy_level = ?")
                    values.append(updates['privacy_level'])
                
                if set_clauses:
                    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                    values.append(entry_id)
                    
                    query = f'''
                    UPDATE album_entries
                    SET {', '.join(set_clauses)}
                    WHERE entry_id = ?
                    '''
                    
                    cursor.execute(query, values)
                    conn.commit()
                    
                    # Clear cache
                    self.cache.clear(f"entry_details_{entry_id}")
                    self.cache.clear(f"entries_person_{entry['person_id']}")
                    
                    st.success("Entry updated successfully!")
                    return True
                else:
                    st.warning("No updates provided")
                    return False
                    
        except Exception as e:
            st.error(f"Error updating entry: {str(e)}")
            return False
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete album entry (admin only)"""
        try:
            # Get entry details for cleanup
            entry = self.get_entry_with_details(entry_id)
            if not entry:
                st.error("Entry not found")
                return False
            
            # Check permissions
            if st.session_state['user_role'] != UserRoles.ADMIN.value:
                st.error("Only administrators can delete entries")
                return False
            
            # Confirm deletion
            if not st.session_state.get(f"confirm_delete_{entry_id}"):
                st.warning(f"Are you sure you want to delete entry '{entry.get('caption', 'Unknown')}'?")
                if st.button("Confirm Deletion", key=f"confirm_btn_{entry_id}"):
                    st.session_state[f"confirm_delete_{entry_id}"] = True
                    st.rerun()
                return False
            
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete comments
                cursor.execute('DELETE FROM comments WHERE entry_id = ?', (entry_id,))
                
                # Delete ratings
                cursor.execute('DELETE FROM ratings WHERE entry_id = ?', (entry_id,))
                
                # Delete favorites
                cursor.execute('DELETE FROM user_favorites WHERE entry_id = ?', (entry_id,))
                
                # Delete album entry
                cursor.execute('DELETE FROM album_entries WHERE entry_id = ?', (entry_id,))
                
                conn.commit()
            
            # Clear relevant caches
            self.cache.clear(f"entry_details_{entry_id}")
            self.cache.clear(f"entries_person_{entry['person_id']}")
            self.cache.clear("recent_entries_")
            self.cache.clear("top_rated_entries_")
            
            st.success("Entry deleted successfully!")
            
            # Reset confirmation
            st.session_state[f"confirm_delete_{entry_id}"] = False
            
            return True
            
        except Exception as e:
            st.error(f"Error deleting entry: {str(e)}")
            return False
    
    def export_data(self, format_type: str = 'csv') -> Optional[io.BytesIO]:
        """Export album data in specified format"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                # Get all entries with details
                query = '''
                SELECT 
                    ae.entry_id,
                    ae.caption,
                    ae.description,
                    ae.location,
                    ae.date_taken,
                    ae.tags,
                    ae.privacy_level,
                    ae.created_at,
                    p.display_name as person_name,
                    i.filename,
                    i.file_size,
                    i.format,
                    (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                    (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN images i ON ae.image_id = i.image_id
                ORDER BY ae.created_at DESC
                '''
                
                df = pd.read_sql_query(query, conn)
                
                if format_type == 'csv':
                    output = io.StringIO()
                    df.to_csv(output, index=False)
                    return io.BytesIO(output.getvalue().encode('utf-8'))
                    
                elif format_type == 'json':
                    output = io.StringIO()
                    df.to_json(output, orient='records', indent=2)
                    return io.BytesIO(output.getvalue().encode('utf-8'))
                    
                elif format_type == 'excel':
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Album Data', index=False)
                    output.seek(0)
                    return output
                    
                else:
                    st.error(f"Unsupported format: {format_type}")
                    return None
                    
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return None
    
    def import_data(self, file_content: bytes, format_type: str) -> bool:
        """Import album data from file"""
        try:
            if format_type == 'csv':
                df = pd.read_csv(io.BytesIO(file_content))
            elif format_type == 'json':
                df = pd.read_json(io.BytesIO(file_content))
            elif format_type == 'excel':
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                st.error(f"Unsupported format: {format_type}")
                return False
            
            # Validate required columns
            required_columns = ['caption', 'person_name', 'filename']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Process each row
            success_count = 0
            error_count = 0
            
            for _, row in df.iterrows():
                try:
                    # Find or create person
                    people = self.db.get_all_people()
                    person = None
                    for p in people:
                        if p['display_name'] == row['person_name']:
                            person = p
                            break
                    
                    if not person:
                        # Create new person
                        person_profile = PersonProfile(
                            person_id=str(uuid.uuid4()),
                            folder_name=row['person_name'].lower().replace(' ', '-'),
                            display_name=row['person_name'],
                            bio=f"Photos of {row['person_name']}",
                            birth_date=None,
                            relationship="Family/Friend",
                            contact_info="",
                            social_links={},
                            profile_image=None,
                            created_at=datetime.datetime.now()
                        )
                        self.db.add_person(person_profile)
                        person_id = person_profile.person_id
                    else:
                        person_id = person['person_id']
                    
                    # Check if image file exists
                    image_path = Config.DATA_DIR / person_profile.folder_name / row['filename']
                    if not image_path.exists():
                        st.warning(f"Image file not found: {image_path}")
                        error_count += 1
                        continue
                    
                    # Check if entry already exists
                    with sqlite3.connect(self.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                        SELECT COUNT(*) FROM album_entries ae
                        JOIN images i ON ae.image_id = i.image_id
                        WHERE i.filename = ? AND ae.person_id = ?
                        ''', (row['filename'], person_id))
                        
                        if cursor.fetchone()[0] > 0:
                            st.info(f"Entry already exists for {row['filename']}")
                            success_count += 1
                            continue
                    
                    # Process image and create entry
                    metadata = ImageMetadata.from_image(image_path)
                    thumbnail_path = self.image_processor.create_thumbnail(image_path)
                    thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
                    
                    self.db.add_image(metadata, thumbnail_path_str)
                    
                    # Parse tags
                    tags = []
                    if 'tags' in row and pd.notna(row['tags']):
                        if isinstance(row['tags'], str):
                            tags = [tag.strip() for tag in row['tags'].split(',') if tag.strip()]
                    
                    album_entry = AlbumEntry(
                        entry_id=str(uuid.uuid4()),
                        image_id=metadata.image_id,
                        person_id=person_id,
                        caption=row['caption'],
                        description=row.get('description', ''),
                        location=row.get('location', ''),
                        date_taken=row.get('date_taken'),
                        tags=tags,
                        privacy_level=row.get('privacy_level', 'public'),
                        created_by=st.session_state['username'],
                        created_at=datetime.datetime.now(),
                        updated_at=datetime.datetime.now()
                    )
                    
                    self.db.add_album_entry(album_entry)
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    st.error(f"Error processing row: {str(e)}")
            
            # Clear cache
            self.cache.clear()
            
            st.success(f"Import completed: {success_count} successful, {error_count} failed")
            return success_count > 0
            
        except Exception as e:
            st.error(f"Error importing data: {str(e)}")
            return False

# ============================================================================
# ENHANCED ALBUM MANAGER WITH AI FEATURES
# ============================================================================
if AI_AVAILABLE:
    class EnhancedAlbumManager(AlbumManager):
        """Enhanced album manager with AI capabilities"""
        
        def __init__(self):
            super().__init__()
            self.ai_processor = AIImageProcessor()
            
            # Initialize AI session state
            self._init_ai_session_state()
        
        def _init_ai_session_state(self):
            """Initialize AI-related session state"""
            if 'ai_analysis_results' not in st.session_state:
                st.session_state.ai_analysis_results = {}
            if 'ai_modifications' not in st.session_state:
                st.session_state.ai_modifications = {}
            if 'ai_suggestions' not in st.session_state:
                st.session_state.ai_suggestions = {}
            if 'batch_processing_queue' not in st.session_state:
                st.session_state.batch_processing_queue = []
        
        def analyze_image_with_ai(self, image_path: Path) -> Dict:
            """Analyze image with AI and cache results"""
            try:
                # Check cache
                image_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
                cache_key = f"ai_analysis_{image_hash}"
                
                if cache_key in st.session_state.ai_analysis_results:
                    return st.session_state.ai_analysis_results[cache_key]
                
                # Load and analyze image
                with Image.open(image_path) as img:
                    analysis = self.ai_processor.analyze_image_with_llm(img)
                    
                    # Add file info
                    analysis['file_info'] = {
                        'path': str(image_path),
                        'size': image_path.stat().st_size,
                        'dimensions': img.size
                    }
                    
                    # Cache results
                    st.session_state.ai_analysis_results[cache_key] = analysis
                    
                    return analysis
                    
            except Exception as e:
                st.error(f"Error analyzing image with AI: {str(e)}")
                return {}
        
        def apply_ai_modification(self, image_path: Path, modification_type: str, 
                                **kwargs) -> Optional[Path]:
            """Apply AI modification to image and save result"""
            try:
                # Load image
                with Image.open(image_path) as img:
                    modified = None
                    
                    if modification_type == 'style_transfer':
                        style = kwargs.get('style', 'realistic')
                        strength = kwargs.get('strength', 0.7)
                        modified = self.ai_processor.apply_ai_style(img, style, strength=strength)
                        
                    elif modification_type == 'text_edit':
                        prompt = kwargs.get('prompt', '')
                        if prompt:
                            modified = self.ai_processor.edit_with_prompt(img, prompt)
                            
                    elif modification_type == 'auto_enhance':
                        modified = self.ai_processor.auto_enhance_image(img)
                        
                    elif modification_type == 'filter':
                        filter_type = kwargs.get('filter_type', 'sharpen')
                        intensity = kwargs.get('intensity', 1.0)
                        modified = self.ai_processor.apply_basic_filters(img, filter_type, intensity)
                    
                    if modified:
                        # Save modified image
                        modified_path = image_path.parent / f"{image_path.stem}_modified_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{image_path.suffix}"
                        modified.save(modified_path)
                        
                        # Update cache
                        image_hash = hashlib.md5(image_path.read_bytes()).hexdigest()
                        cache_key = f"modification_{image_hash}_{modification_type}"
                        st.session_state.ai_modifications[cache_key] = str(modified_path)
                        
                        # Add to album
                        self._add_modified_image_to_album(modified_path, image_path)
                        
                        return modified_path
                
                return None
                
            except Exception as e:
                st.error(f"Error applying AI modification: {str(e)}")
                return None
        
        def _add_modified_image_to_album(self, modified_path: Path, original_path: Path):
            """Add modified image to album database"""
            try:
                # Get original entry
                with sqlite3.connect(self.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT ae.*, p.folder_name 
                    FROM album_entries ae
                    JOIN images i ON ae.image_id = i.image_id
                    JOIN people p ON ae.person_id = p.person_id
                    WHERE i.filename LIKE ?
                    ''', (f'%{original_path.name}',))
                    
                    original_entry = cursor.fetchone()
                    
                    if original_entry:
                        # Create metadata for modified image
                        metadata = ImageMetadata.from_image(modified_path)
                        thumbnail_path = self.image_processor.create_thumbnail(modified_path)
                        thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
                        
                        # Add to database
                        self.db.add_image(metadata, thumbnail_path_str)
                        
                        # Create album entry
                        album_entry = AlbumEntry(
                            entry_id=str(uuid.uuid4()),
                            image_id=metadata.image_id,
                            person_id=original_entry['person_id'],
                            caption=f"[AI Modified] {original_entry['caption']}",
                            description=f"AI-modified version of original photo. {original_entry.get('description', '')}",
                            location=original_entry.get('location', ''),
                            date_taken=datetime.datetime.now(),
                            tags=json.loads(original_entry['tags']) if original_entry['tags'] else [] + ['ai-modified'],
                            privacy_level=original_entry.get('privacy_level', 'public'),
                            created_by=st.session_state['username'],
                            created_at=datetime.datetime.now(),
                            updated_at=datetime.datetime.now()
                        )
                        
                        self.db.add_album_entry(album_entry)
                        
            except Exception as e:
                st.warning(f"Could not add modified image to album: {str(e)}")
        
        def batch_analyze_images(self, image_paths: List[Path]) -> Dict:
            """Batch analyze multiple images"""
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, image_path in enumerate(image_paths):
                try:
                    status_text.text(f"Analyzing {image_path.name}...")
                    analysis = self.analyze_image_with_ai(image_path)
                    results[str(image_path)] = analysis
                    
                    # Update progress
                    progress = (i + 1) / len(image_paths)
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.warning(f"Error analyzing {image_path.name}: {str(e)}")
                    results[str(image_path)] = {'error': str(e)}
            
            progress_bar.empty()
            status_text.empty()
            
            return results
        
        def get_ai_suggestions(self, image_path: Path) -> List[Dict]:
            """Get AI suggestions for image improvement"""
            try:
                analysis = self.analyze_image_with_ai(image_path)
                
                suggestions = []
                
                # Composition suggestions
                comp_suggestions = analysis.get('composition', {}).get('suggestions', [])
                for suggestion in comp_suggestions:
                    suggestions.append({
                        'type': 'composition',
                        'suggestion': suggestion,
                        'priority': 'medium'
                    })
                
                # Style transfer suggestions
                caption = analysis.get('caption', '').lower()
                if 'portrait' in caption or 'person' in caption:
                    suggestions.append({
                        'type': 'style',
                        'suggestion': 'Try portrait enhancement or painting style',
                        'priority': 'low'
                    })
                
                if 'landscape' in caption or 'nature' in caption:
                    suggestions.append({
                        'type': 'style',
                        'suggestion': 'Consider fantasy or painting style for landscape',
                        'priority': 'medium'
                    })
                
                # Enhancement suggestions
                suggestions.append({
                    'type': 'enhancement',
                    'suggestion': 'Try auto-enhance for quick improvements',
                    'priority': 'high'
                })
                
                # Tag suggestions
                tags = analysis.get('tags', [])
                if len(tags) < 5:
                    suggestions.append({
                        'type': 'metadata',
                        'suggestion': 'Add more descriptive tags for better searchability',
                        'priority': 'low'
                    })
                
                return suggestions
                
            except Exception as e:
                st.error(f"Error getting AI suggestions: {str(e)}")
                return []

# ============================================================================
# BASIC PHOTO ALBUM APP
# ============================================================================
class PhotoAlbumApp:
    """Main application class with UI components"""
    
    def __init__(self):
        if AI_AVAILABLE:
            self.manager = EnhancedAlbumManager()
        else:
            self.manager = AlbumManager()
        self.setup_page_config()
        self.check_initialization()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=Config.APP_NAME,
            page_icon="📸",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/yourusername/photo-album',
                'Report a bug': 'https://github.com/yourusername/photo-album/issues',
                'About': f"# {Config.APP_NAME} v{Config.VERSION}\nA comprehensive photo album management system"
            }
        )
    
    def check_initialization(self):
        """Check and initialize application directories"""
        try:
            Config.init_directories()
            self.initialized = True
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            self.initialized = False
    
    def render_sidebar(self):
        """Render application sidebar"""
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
                "👥 People": "people",
                "📁 Gallery": "gallery",
                "⭐ Favorites": "favorites",
                "🔍 Search": "search",
                "📊 Statistics": "statistics",
                "⚙️ Settings": "settings",
                "📤 Import/Export": "import_export"
            }
            
            # Add AI navigation if available
            if AI_AVAILABLE:
                nav_options.update({
                    "🤖 AI Analysis": "ai_analysis",
                    "🎨 AI Studio": "ai_studio",
                    "⚡ Quick Enhance": "quick_enhance",
                    "📊 AI Insights": "ai_insights"
                })
            
            for label, key in nav_options.items():
                if st.button(label, use_container_width=True, key=f"nav_{key}"):
                    st.session_state['current_page'] = key
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
            
            # AI status
            if AI_AVAILABLE:
                st.divider()
                st.subheader("AI Status")
                if hasattr(self.manager, 'ai_processor'):
                    loaded_models = len(self.manager.ai_processor.model_manager.model_cache)
                    st.metric("Loaded Models", loaded_models)
                else:
                    st.metric("AI Models", "Not Loaded")
            
            # Theme toggle
            st.divider()
            dark_mode = st.toggle("Dark Mode", value=False)
            if dark_mode:
                st.markdown("""
                <style>
                .stApp {
                    background-color: #1a1a1a;
                    color: white;
                }
                </style>
                """, unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render dashboard page"""
        st.title("📊 Dashboard")
        st.markdown("Welcome to your photo album management system!")
        
        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            people = self.manager.db.get_all_people()
            st.metric("👥 People", len(people))
        
        with col2:
            total_images = sum(self.manager.get_person_stats(p['person_id'])['image_count'] for p in people)
            st.metric("📸 Total Images", total_images)
        
        with col3:
            recent = self.manager.get_recent_entries(limit=1)
            if recent:
                st.metric("🕐 Last Added", recent[0].get('caption', 'N/A'))
            else:
                st.metric("🕐 Last Added", "None")
        
        with col4:
            top_rated = self.manager.get_top_rated_entries(limit=1)
            if top_rated:
                avg_rating = top_rated[0].get('avg_rating', 0)
                st.metric("⭐ Top Rated", f"{avg_rating:.1f}/5")
            else:
                st.metric("⭐ Top Rated", "N/A")
        
        st.divider()
        
        # Recent Activity
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📅 Recent Activity")
            recent_entries = self.manager.get_recent_entries(limit=5)
            
            if recent_entries:
                for entry in recent_entries:
                    with st.container():
                        col_a, col_b = st.columns([1, 4])
                        with col_a:
                            if entry.get('thumbnail_data_url'):
                                st.image(entry['thumbnail_data_url'], width=80)
                            else:
                                st.markdown("🖼️")
                        with col_b:
                            st.markdown(f"**{entry.get('caption', 'Untitled')}**")
                            st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
                            st.caption(f"📅 {entry.get('created_at', '')}")
                        st.divider()
            else:
                st.info("No recent activity")
        
        with col2:
            st.subheader("🏆 Top Rated")
            top_entries = self.manager.get_top_rated_entries(limit=5)
            
            if top_entries:
                for entry in top_entries:
                    rating = entry.get('avg_rating', 0)
                    if rating >= 4:
                        st.markdown(f"⭐ **{entry.get('caption', 'Untitled')}**")
                        st.markdown(f"`{rating:.1f}/5` 👤 {entry.get('display_name', 'Unknown')}")
                        st.divider()
            else:
                st.info("No ratings yet")
        
        # Quick Actions
        st.divider()
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📁 Scan for New Photos", use_container_width=True):
                with st.spinner("Scanning..."):
                    results = self.manager.scan_directory()
                    st.success(f"Found {results['new_images']} new photos!")
                    st.rerun()
        
        with col2:
            if st.button("👥 Manage People", use_container_width=True):
                st.session_state['current_page'] = 'people'
                st.rerun()
        
        with col3:
            if st.button("📤 Export Data", use_container_width=True):
                st.session_state['current_page'] = 'import_export'
                st.rerun()
    
    def render_people_page(self):
        """Render people management page"""
        st.title("👥 People")
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search people...", key="people_search")
        with col2:
            sort_by = st.selectbox("Sort by", ["Name", "Recently Added", "Image Count"], key="people_sort")
        
        # Get all people
        all_people = self.manager.get_all_people_with_stats()
        
        # Filter and sort
        if search_query:
            all_people = [p for p in all_people if search_query.lower() in p['display_name'].lower()]
        
        if sort_by == "Name":
            all_people.sort(key=lambda x: x['display_name'])
        elif sort_by == "Image Count":
            all_people.sort(key=lambda x: x['image_count'], reverse=True)
        elif sort_by == "Recently Added":
            all_people.sort(key=lambda x: x.get('last_activity', ''), reverse=True)
        
        # Display people
        cols = st.columns(3)
        for idx, person in enumerate(all_people):
            with cols[idx % 3]:
                with st.container(border=True):
                    # Profile image or placeholder
                    if person.get('profile_image_data'):
                        st.image(person['profile_image_data'], use_column_width=True)
                    else:
                        # Create colored placeholder
                        colors = ['#667eea', '#764ba2', '#f56565', '#48bb78', '#ed8936']
                        color = colors[hash(person['person_id']) % len(colors)]
                        st.markdown(
                            f'<div style="background: {color}; height: 150px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">'
                            f'<span style="color: white; font-size: 48px;">{person["display_name"][0].upper()}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.subheader(person['display_name'])
                    st.caption(f"📁 {person['folder_name']}")
                    
                    # Stats
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Photos", person.get('image_count', 0))
                    with col_b:
                        st.metric("Comments", person.get('comment_count', 0))
                    with col_c:
                        st.metric("Rating", f"{person.get('avg_rating', 0):.1f}")
                    
                    # Actions
                    if st.button("View Gallery", key=f"view_{person['person_id']}", use_container_width=True):
                        st.session_state['selected_person'] = person['person_id']
                        st.session_state['current_page'] = 'gallery'
                        st.rerun()
        
        # Add new person
        st.divider()
        with st.expander("➕ Add New Person"):
            with st.form("add_person_form"):
                col1, col2 = st.columns(2)
                with col1:
                    folder_name = st.text_input("Folder Name*")
                    display_name = st.text_input("Display Name*")
                    relationship = st.selectbox(
                        "Relationship",
                        ["Family", "Friend", "Colleague", "Relative", "Other"]
                    )
                with col2:
                    birth_date = st.date_input("Birth Date", value=None)
                    contact_info = st.text_input("Contact Info")
                    bio = st.text_area("Bio", height=100)
                
                if st.form_submit_button("Add Person"):
                    if folder_name and display_name:
                        # Create folder
                        person_dir = Config.DATA_DIR / folder_name
                        person_dir.mkdir(exist_ok=True)
                        
                        # Add to database
                        person_profile = PersonProfile(
                            person_id=str(uuid.uuid4()),
                            folder_name=folder_name,
                            display_name=display_name,
                            bio=bio,
                            birth_date=birth_date,
                            relationship=relationship,
                            contact_info=contact_info,
                            social_links={},
                            profile_image=None,
                            created_at=datetime.datetime.now()
                        )
                        
                        try:
                            self.manager.db.add_person(person_profile)
                            st.success(f"Person '{display_name}' added successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding person: {str(e)}")
                    else:
                        st.warning("Please fill in all required fields (*)")
    
    def render_gallery_page(self):
        """Render gallery page"""
        st.title("📁 Gallery")
        
        # Check if we're viewing a specific person
        selected_person_id = st.session_state.get('selected_person')
        if selected_person_id:
            people = self.manager.db.get_all_people()
            selected_person = next((p for p in people if p['person_id'] == selected_person_id), None)
            if selected_person:
                st.subheader(f"Photos of {selected_person['display_name']}")
                if st.button("← Back to All People"):
                    st.session_state['selected_person'] = None
                    st.rerun()
        
        # Search and filter
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                search_query = st.text_input("Search photos...", key="gallery_search")
            with col2:
                view_mode = st.selectbox("View", ["Grid", "List"], key="view_mode")
            with col3:
                sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Rating"], key="gallery_sort")
            with col4:
                items_per_page = st.selectbox("Per Page", [12, 24, 48], key="items_per_page")
        
        # Get entries
        page = st.session_state.get('gallery_page', 1)
        if selected_person_id:
            data = self.manager.get_entries_by_person(selected_person_id, page, search_query)
        else:
            # Get all entries with pagination
            offset = (page - 1) * items_per_page
            with sqlite3.connect(self.manager.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if search_query:
                    search_pattern = f'%{search_query}%'
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                           (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
                    ORDER BY ae.created_at DESC
                    LIMIT ? OFFSET ?
                    ''', (search_pattern, search_pattern, search_pattern, items_per_page, offset))
                else:
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.filename, i.thumbnail_path,
                           (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    ORDER BY ae.created_at DESC
                    LIMIT ? OFFSET ?
                    ''', (items_per_page, offset))
                
                entries = [dict(row) for row in cursor.fetchall()]
                
                # Get total count
                if search_query:
                    search_pattern = f'%{search_query}%'
                    cursor.execute('''
                    SELECT COUNT(*)
                    FROM album_entries ae
                    WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
                    ''', (search_pattern, search_pattern, search_pattern))
                else:
                    cursor.execute('SELECT COUNT(*) FROM album_entries')
                
                total_count = cursor.fetchone()[0]
                total_pages = max(1, math.ceil(total_count / items_per_page))
                
                data = {
                    'entries': entries,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'current_page': page
                }
        
        # Sort entries
        if sort_by == "Oldest":
            data['entries'].sort(key=lambda x: x.get('created_at', ''))
        elif sort_by == "Rating":
            data['entries'].sort(key=lambda x: x.get('avg_rating', 0), reverse=True)
        else:  # Newest (default)
            data['entries'].sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Display entries
        if view_mode == "Grid":
            cols = st.columns(4)
            for idx, entry in enumerate(data['entries']):
                with cols[idx % 4]:
                    self._render_gallery_item(entry)
        else:  # List view
            for entry in data['entries']:
                self._render_gallery_item_list(entry)
        
        # Pagination
        if data['total_pages'] > 1:
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page_numbers = []
                for i in range(1, data['total_pages'] + 1):
                    if i == 1 or i == data['total_pages'] or abs(i - page) <= 2:
                        page_numbers.append(i)
                    elif page_numbers[-1] != "...":
                        page_numbers.append("...")
                
                col_btns = st.columns(len(page_numbers) + 2)
                
                with col_btns[0]:
                    if page > 1 and st.button("◀", key="prev_page"):
                        st.session_state.gallery_page = page - 1
                        st.rerun()
                
                for idx, page_num in enumerate(page_numbers, 1):
                    with col_btns[idx]:
                        if page_num == "...":
                            st.markdown("...")
                        elif page_num == page:
                            st.markdown(f"**{page_num}**")
                        elif st.button(str(page_num), key=f"page_{page_num}"):
                            st.session_state.gallery_page = page_num
                            st.rerun()
                
                with col_btns[-1]:
                    if page < data['total_pages'] and st.button("▶", key="next_page"):
                        st.session_state.gallery_page = page + 1
                        st.rerun()
    
    def _render_gallery_item(self, entry):
        """Render a gallery item in grid view"""
        with st.container(border=True):
            # Image
            if entry.get('thumbnail_data_url'):
                st.image(entry['thumbnail_data_url'], use_column_width=True)
            elif entry.get('thumbnail_path'):
                thumbnail_path = Path(entry['thumbnail_path'])
                if thumbnail_path.exists():
                    data_url = self.manager.image_processor.get_image_data_url(thumbnail_path)
                    st.image(data_url, use_column_width=True)
                else:
                    st.markdown("🖼️")
            else:
                st.markdown("🖼️")
            
            # Caption
            st.markdown(f"**{entry.get('caption', 'Untitled')}**")
            st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
            
            # Rating
            if entry.get('avg_rating'):
                st.markdown(UIComponents.rating_stars(entry['avg_rating'], size=15))
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👁️", key=f"view_{entry['entry_id']}", use_container_width=True):
                    st.session_state['selected_image'] = entry['entry_id']
                    st.session_state['current_page'] = 'image_detail'
                    st.rerun()
            with col2:
                # Check if favorited
                is_favorited = entry['entry_id'] in st.session_state.get('favorites', set())
                if is_favorited:
                    if st.button("⭐", key=f"unfav_{entry['entry_id']}", use_container_width=True):
                        self.manager.remove_from_favorites(entry['entry_id'])
                        st.rerun()
                else:
                    if st.button("☆", key=f"fav_{entry['entry_id']}", use_container_width=True):
                        self.manager.add_to_favorites(entry['entry_id'])
                        st.rerun()
    
    def _render_gallery_item_list(self, entry):
        """Render a gallery item in list view"""
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # Thumbnail
                if entry.get('thumbnail_data_url'):
                    st.image(entry['thumbnail_data_url'], width=100)
                elif entry.get('thumbnail_path'):
                    thumbnail_path = Path(entry['thumbnail_path'])
                    if thumbnail_path.exists():
                        data_url = self.manager.image_processor.get_image_data_url(thumbnail_path)
                        st.image(data_url, width=100)
                    else:
                        st.markdown("🖼️")
                else:
                    st.markdown("🖼️")
            
            with col2:
                st.markdown(f"### {entry.get('caption', 'Untitled')}")
                st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
                
                if entry.get('description'):
                    st.markdown(f"*{entry.get('description', '')[:100]}...*")
                
                if entry.get('tags'):
                    st.markdown(UIComponents.tag_badges(entry['tags'].split(',') if isinstance(entry['tags'], str) else entry['tags'], max_display=3), unsafe_allow_html=True)
            
            with col3:
                # Rating
                if entry.get('avg_rating'):
                    st.markdown(UIComponents.rating_stars(entry['avg_rating']), unsafe_allow_html=True)
                
                # Actions
                if st.button("View Details", key=f"view_det_{entry['entry_id']}", use_container_width=True):
                    st.session_state['selected_image'] = entry['entry_id']
                    st.session_state['current_page'] = 'image_detail'
                    st.rerun()
                
                # Favorite toggle
                is_favorited = entry['entry_id'] in st.session_state.get('favorites', set())
                if is_favorited:
                    if st.button("Remove Favorite", key=f"rem_fav_{entry['entry_id']}", use_container_width=True):
                        self.manager.remove_from_favorites(entry['entry_id'])
                        st.rerun()
                else:
                    if st.button("Add Favorite", key=f"add_fav_{entry['entry_id']}", use_container_width=True):
                        self.manager.add_to_favorites(entry['entry_id'])
                        st.rerun()
    
    def render_image_detail_page(self):
        """Render image detail page"""
        entry_id = st.session_state.get('selected_image')
        if not entry_id:
            st.error("No image selected")
            if st.button("Back to Gallery"):
                st.session_state['current_page'] = 'gallery'
                st.rerun()
            return
        
        entry = self.manager.get_entry_with_details(entry_id)
        if not entry:
            st.error("Image not found")
            if st.button("Back to Gallery"):
                st.session_state['current_page'] = 'gallery'
                st.rerun()
            return
        
        # Back button
        if st.button("← Back"):
            st.session_state['current_page'] = 'gallery'
            st.rerun()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image
            if entry.get('image_data_url'):
                st.image(entry['image_data_url'], use_column_width=True)
            else:
                # Try to load from filepath
                if entry.get('filepath'):
                    image_path = Config.DATA_DIR / entry['filepath']
                    if image_path.exists():
                        data_url = self.manager.image_processor.get_image_data_url(image_path)
                        st.image(data_url, use_column_width=True)
                    else:
                        st.error("Image file not found")
                else:
                    st.error("No image available")
        
        with col2:
            # Image info
            st.title(entry.get('caption', 'Untitled'))
            st.markdown(f"👤 **{entry.get('display_name', 'Unknown')}**")
            
            # Rating
            avg_rating = entry.get('avg_rating', 0)
            rating_count = entry.get('rating_count', 0)
            st.markdown(UIComponents.rating_stars(avg_rating), unsafe_allow_html=True)
            st.caption(f"{rating_count} ratings")
            
            # User rating
            st.subheader("Your Rating")
            user_rating = entry.get('user_rating', 0)
            rating_cols = st.columns(5)
            for i in range(1, 6):
                with rating_cols[i-1]:
                    if st.button(f"{i} ⭐", key=f"rate_{i}", use_container_width=True):
                        self.manager.add_rating_to_entry(entry_id, i)
                        st.rerun()
            
            # Favorite button
            is_favorited = entry.get('is_favorited', False)
            if is_favorited:
                if st.button("⭐ Remove Favorite", use_container_width=True):
                    self.manager.remove_from_favorites(entry_id)
                    st.rerun()
            else:
                if st.button("☆ Add Favorite", use_container_width=True):
                    self.manager.add_to_favorites(entry_id)
                    st.rerun()
            
            # Metadata
            with st.expander("📊 Metadata"):
                if entry.get('description'):
                    st.markdown(f"**Description:** {entry['description']}")
                
                if entry.get('location'):
                    st.markdown(f"**Location:** {entry['location']}")
                
                if entry.get('date_taken'):
                    st.markdown(f"**Date Taken:** {entry['date_taken']}")
                
                if entry.get('tags'):
                    st.markdown("**Tags:**")
                    st.markdown(UIComponents.tag_badges(
                        entry['tags'] if isinstance(entry['tags'], list) else entry['tags'].split(','),
                        max_display=10
                    ), unsafe_allow_html=True)
                
                if entry.get('file_size'):
                    file_size_mb = entry['file_size'] / (1024 * 1024)
                    st.markdown(f"**File Size:** {file_size_mb:.2f} MB")
                
                if entry.get('format'):
                    st.markdown(f"**Format:** {entry['format']}")
                
                if entry.get('dimensions'):
                    st.markdown(f"**Dimensions:** {entry['dimensions']}")
        
        # AI Analysis section if available
        if AI_AVAILABLE and hasattr(self.manager, 'analyze_image_with_ai'):
            st.divider()
            st.subheader("🤖 AI Analysis")
            
            if entry.get('filepath'):
                image_path = Config.DATA_DIR / entry['filepath']
                if image_path.exists():
                    if st.button("Analyze with AI", key="analyze_ai"):
                        with st.spinner("Analyzing image with AI..."):
                            analysis = self.manager.analyze_image_with_ai(image_path)
                            
                            if analysis:
                                EnhancedUIComponents.ai_analysis_panel(analysis)
                                
                                # Show AI modification interface
                                st.divider()
                                EnhancedUIComponents.ai_modification_interface(image_path, self.manager)
        
        # Comments section
        st.divider()
        st.subheader("💬 Comments")
        
        # Add comment form
        with st.form(key="add_comment_form"):
            comment_text = st.text_area("Add a comment...", height=100, max_chars=Config.MAX_COMMENT_LENGTH)
            submitted = st.form_submit_button("Post Comment")
            if submitted and comment_text.strip():
                if self.manager.add_comment_to_entry(entry_id, comment_text.strip()):
                    st.rerun()
        
        # Display comments
        comments = entry.get('comments', [])
        if comments:
            for comment in comments:
                with st.container(border=True):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**{comment.get('username', 'Anonymous')}**")
                        st.caption(comment.get('created_at', ''))
                    with col2:
                        st.markdown(comment.get('content', ''))
        else:
            st.info("No comments yet. Be the first to comment!")
    
    def render_favorites_page(self):
        """Render favorites page"""
        st.title("⭐ Favorites")
        
        favorites = self.manager.get_user_favorites()
        
        if not favorites:
            st.info("You haven't added any favorites yet.")
            st.markdown("Browse the gallery and click the ☆ button to add favorites!")
            return
        
        # Display favorites
        cols = st.columns(4)
        for idx, entry in enumerate(favorites):
            with cols[idx % 4]:
                self._render_gallery_item(entry)
    
    def render_search_page(self):
        """Render search page"""
        st.title("🔍 Search")
        
        # Search options
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search term", key="global_search")
        with col2:
            search_in = st.selectbox(
                "Search in",
                ["All Fields", "Captions", "Descriptions", "Tags", "People"]
            )
        
        if search_query:
            with st.spinner("Searching..."):
                # Perform search
                results = []
                
                if search_in in ["All Fields", "Captions", "Descriptions", "Tags"]:
                    db_results = self.manager.db.search_entries(search_query)
                    results.extend(db_results)
                
                if search_in in ["All Fields", "People"]:
                    people = self.manager.db.get_all_people()
                    for person in people:
                        if search_query.lower() in person['display_name'].lower():
                            # Get person's entries
                            with sqlite3.connect(self.manager.db.db_path) as conn:
                                conn.row_factory = sqlite3.Row
                                cursor = conn.cursor()
                                cursor.execute('''
                                SELECT ae.*, p.display_name, i.filename, i.thumbnail_path
                                FROM album_entries ae
                                JOIN people p ON ae.person_id = p.person_id
                                JOIN images i ON ae.image_id = i.image_id
                                WHERE ae.person_id = ?
                                LIMIT 5
                                ''', (person['person_id'],))
                                person_entries = [dict(row) for row in cursor.fetchall()]
                                results.extend(person_entries)
                
                # Remove duplicates
                seen_ids = set()
                unique_results = []
                for result in results:
                    if result['entry_id'] not in seen_ids:
                        seen_ids.add(result['entry_id'])
                        unique_results.append(result)
                
                # Display results
                st.subheader(f"Found {len(unique_results)} results")
                
                if unique_results:
                    for entry in unique_results:
                        self._render_gallery_item_list(entry)
                else:
                    st.info("No results found")
        else:
            st.info("Enter a search term to find photos")
    
    def render_statistics_page(self):
        """Render statistics page"""
        st.title("📊 Statistics")
        
        # Get all data
        all_people = self.manager.get_all_people_with_stats()
        
        if not all_people:
            st.info("No data available. Scan your directory first.")
            return
        
        # Overall stats
        col1, col2, col3, col4 = st.columns(4)
        
        total_images = sum(p['image_count'] for p in all_people)
        total_comments = sum(p['comment_count'] for p in all_people)
        avg_ratings = [p['avg_rating'] for p in all_people if p['avg_rating'] > 0]
        avg_rating = sum(avg_ratings) / len(avg_ratings) if avg_ratings else 0
        
        with col1:
            st.metric("Total People", len(all_people))
        with col2:
            st.metric("Total Images", total_images)
        with col3:
            st.metric("Total Comments", total_comments)
        with col4:
            st.metric("Average Rating", f"{avg_rating:.1f}")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Images per Person")
            
            # Prepare data for chart
            people_names = [p['display_name'] for p in all_people]
            image_counts = [p['image_count'] for p in all_people]
            
            if image_counts:
                chart_data = pd.DataFrame({
                    'Person': people_names,
                    'Images': image_counts
                })
                st.bar_chart(chart_data.set_index('Person'))
        
        with col2:
            st.subheader("⭐ Ratings Distribution")
            
            # Get all ratings
            with sqlite3.connect(self.manager.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT rating_value, COUNT(*) as count
                FROM ratings
                GROUP BY rating_value
                ORDER BY rating_value
                ''')
                rating_data = cursor.fetchall()
            
            if rating_data:
                ratings = [r[0] for r in rating_data]
                counts = [r[1] for r in rating_data]
                
                chart_data = pd.DataFrame({
                    'Rating': ratings,
                    'Count': counts
                })
                st.bar_chart(chart_data.set_index('Rating'))
        
        # Detailed table
        st.divider()
        st.subheader("👥 People Details")
        
        table_data = []
        for person in all_people:
            table_data.append({
                'Name': person['display_name'],
                'Images': person['image_count'],
                'Comments': person['comment_count'],
                'Avg Rating': f"{person['avg_rating']:.1f}",
                'Last Activity': person.get('last_activity', 'Never')
            })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            
            # Export button
            if st.button("📊 Export Statistics"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="photo_album_stats.csv",
                    mime="text/csv"
                )
    
    def render_settings_page(self):
        """Render settings page"""
        st.title("⚙️ Settings")
        
        tabs = st.tabs(["Application", "User", "Database", "Advanced"])
        
        with tabs[0]:  # Application
            st.subheader("Application Settings")
            
            # Directory settings
            with st.expander("📁 Directory Settings"):
                st.info(f"**Data Directory:** {Config.DATA_DIR}")
                st.info(f"**Thumbnail Directory:** {Config.THUMBNAIL_DIR}")
                st.info(f"**Database Directory:** {Config.DB_DIR}")
                
                if st.button("Create Missing Directories"):
                    Config.init_directories()
                    st.success("Directories created/verified!")
            
            # Image settings
            with st.expander("🖼️ Image Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    new_thumb_size = st.number_input("Thumbnail Width", value=Config.THUMBNAIL_SIZE[0], min_value=100, max_value=800)
                    Config.THUMBNAIL_SIZE = (int(new_thumb_size), int(new_thumb_size))
                with col2:
                    new_preview_size = st.number_input("Preview Width", value=Config.PREVIEW_SIZE[0], min_value=400, max_value=1200)
                    Config.PREVIEW_SIZE = (int(new_preview_size), int(new_preview_size))
                
                if st.button("Apply Image Settings"):
                    st.success("Image settings updated!")
            
            # Gallery settings
            with st.expander("📁 Gallery Settings"):
                items_per_page = st.number_input("Items per Page", value=Config.ITEMS_PER_PAGE, min_value=1, max_value=100)
                grid_columns = st.number_input("Grid Columns", value=Config.GRID_COLUMNS, min_value=1, max_value=8)
                
                Config.ITEMS_PER_PAGE = int(items_per_page)
                Config.GRID_COLUMNS = int(grid_columns)
                
                if st.button("Apply Gallery Settings"):
                    st.success("Gallery settings updated!")
        
        with tabs[1]:  # User
            st.subheader("User Settings")
            
            current_username = st.session_state['username']
            current_role = st.session_state['user_role']
            
            new_username = st.text_input("Username", value=current_username)
            new_role = st.selectbox(
                "Role",
                [role.value for role in UserRoles],
                index=[role.value for role in UserRoles].index(current_role)
            )
            
            if st.button("Update User Settings"):
                st.session_state['username'] = new_username
                st.session_state['user_role'] = new_role
                st.success("User settings updated!")
        
        with tabs[2]:  # Database
            st.subheader("Database Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Rebuild Database", help="Recreate all database tables (data will be preserved)"):
                    try:
                        self.manager.db._init_database()
                        st.success("Database rebuilt successfully!")
                    except Exception as e:
                        st.error(f"Error rebuilding database: {str(e)}")
                
                if st.button("📊 Optimize Database", help="Run database optimization commands"):
                    try:
                        with self.manager.db.get_connection() as conn:
                            conn.execute("VACUUM")
                            conn.execute("ANALYZE")
                        st.success("Database optimized!")
                    except Exception as e:
                        st.error(f"Error optimizing database: {str(e)}")
            
            with col2:
                if st.button("🗑️ Clear All Data", help="WARNING: This will delete all data!"):
                    if st.checkbox("I understand this will delete ALL data"):
                        try:
                            os.remove(self.manager.db.db_path)
                            self.manager.db = DatabaseManager()
                            st.success("All data cleared!")
                        except Exception as e:
                            st.error(f"Error clearing data: {str(e)}")
                
                db_size = self.manager.db.db_path.stat().st_size if self.manager.db.db_path.exists() else 0
                st.info(f"**Database Size:** {db_size / (1024*1024):.2f} MB")
        
        with tabs[3]:  # Advanced
            st.subheader("Advanced Settings")
            
            # Cache settings
            with st.expander("💾 Cache Settings"):
                cache_ttl = st.number_input("Cache TTL (seconds)", value=Config.CACHE_TTL, min_value=60, max_value=86400)
                Config.CACHE_TTL = int(cache_ttl)
                
                cache_size = len(self.manager.cache._cache)
                st.info(f"**Cache Items:** {cache_size}")
                
                if st.button("Clear Cache"):
                    self.manager.cache.clear()
                    st.success("Cache cleared!")
            
            # AI Settings if available
            if AI_AVAILABLE:
                with st.expander("🤖 AI Settings"):
                    ai_enabled = st.checkbox("Enable AI Features", value=Config.AI_ENABLED)
                    Config.AI_ENABLED = ai_enabled
                    
                    if hasattr(self.manager, 'ai_processor'):
                        loaded_models = len(self.manager.ai_processor.model_manager.model_cache)
                        st.info(f"**Loaded AI Models:** {loaded_models}")
                        
                        if st.button("Clear AI Cache"):
                            self.manager.ai_processor.model_manager.clear_all_models()
                            st.success("AI cache cleared!")
                    
                    if st.button("Load AI Models"):
                        if AI_AVAILABLE:
                            st.info("AI models will be loaded on-demand when needed.")
                        else:
                            st.error("AI libraries not installed. Install with: pip install torch transformers diffusers")
            
            # Performance settings
            with st.expander("⚡ Performance"):
                st.checkbox("Enable image lazy loading", value=True)
                st.checkbox("Enable background processing", value=True)
                st.checkbox("Use compression for thumbnails", value=True)
            
            # Debug settings
            with st.expander("🐛 Debug"):
                debug_mode = st.checkbox("Enable debug mode", value=False)
                if debug_mode:
                    st.warning("Debug mode enabled. Additional logging will be shown.")
                
                if st.button("Show Session State"):
                    st.write(st.session_state)
                
                if st.button("Show Configuration"):
                    st.write({
                        'APP_NAME': Config.APP_NAME,
                        'VERSION': Config.VERSION,
                        'DATA_DIR': str(Config.DATA_DIR),
                        'THUMBNAIL_SIZE': Config.THUMBNAIL_SIZE,
                        'ITEMS_PER_PAGE': Config.ITEMS_PER_PAGE,
                        'CACHE_TTL': Config.CACHE_TTL,
                        'AI_ENABLED': Config.AI_ENABLED
                    })
    
    def render_import_export_page(self):
        """Render import/export page"""
        st.title("📤 Import/Export")
        
        tabs = st.tabs(["Export", "Import", "Backup"])
        
        with tabs[0]:  # Export
            st.subheader("Export Data")
            
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "Excel"],
                key="export_format"
            )
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                export_people = st.multiselect(
                    "Select People",
                    [p['display_name'] for p in self.manager.db.get_all_people()],
                    key="export_people"
                )
            with col2:
                date_range = st.date_input(
                    "Date Range",
                    value=(datetime.date.today() - datetime.timedelta(days=365), datetime.date.today()),
                    key="export_dates"
                )
            
            if st.button("Generate Export", type="primary"):
                with st.spinner("Generating export file..."):
                    export_data = self.manager.export_data(export_format.lower())
                    
                    if export_data:
                        file_ext = export_format.lower()
                        if file_ext == "excel":
                            file_ext = "xlsx"
                        
                        st.download_button(
                            label=f"Download {export_format} File",
                            data=export_data,
                            file_name=f"photo_album_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime={
                                "csv": "text/csv",
                                "json": "application/json",
                                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            }[export_format.lower()]
                        )
        
        with tabs[1]:  # Import
            st.subheader("Import Data")
            
            st.info("""
            **Import Format:** CSV, JSON, or Excel files with the following columns:
            - `caption` (required): Photo caption
            - `person_name` (required): Name of the person
            - `filename` (required): Image filename (must exist in person's folder)
            - `description`: Photo description
            - `location`: Location where photo was taken
            - `tags`: Comma-separated tags
            - `privacy_level`: public/private
            """)
            
            import_file = st.file_uploader(
                "Choose import file",
                type=['csv', 'json', 'xlsx', 'xls'],
                key="import_file"
            )
            
            if import_file:
                file_ext = import_file.name.split('.')[-1].lower()
                if file_ext == 'xls':
                    file_ext = 'excel'
                
                if st.button("Import Data", type="primary"):
                    with st.spinner("Importing data..."):
                        success = self.manager.import_data(import_file.getvalue(), file_ext)
                        if success:
                            st.success("Import completed successfully!")
                            st.rerun()
        
        with tabs[2]:  # Backup
            st.subheader("Backup & Restore")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Create Backup")
                backup_name = st.text_input(
                    "Backup Name",
                    value=f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    key="backup_name"
                )
                
                if st.button("🔒 Create Backup", use_container_width=True):
                    # Create backup of database
                    backup_path = Config.EXPORT_DIR / f"{backup_name}.db"
                    import shutil
                    shutil.copy2(self.manager.db.db_path, backup_path)
                    st.success(f"Backup created: {backup_path}")
                    
                    # Offer download
                    with open(backup_path, 'rb') as f:
                        st.download_button(
                            label="Download Backup",
                            data=f,
                            file_name=backup_path.name,
                            mime="application/x-sqlite3"
                        )
            
            with col2:
                st.markdown("### Restore Backup")
                backup_file = st.file_uploader(
                    "Choose backup file",
                    type=['db', 'sqlite', 'sqlite3'],
                    key="restore_file"
                )
                
                if backup_file:
                    if st.button("⚠️ Restore Backup", type="secondary", use_container_width=True):
                        st.warning("This will replace your current database!")
                        if st.checkbox("I understand this will overwrite all current data"):
                            try:
                                # Create backup of current database first
                                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                current_backup = Config.EXPORT_DIR / f"pre_restore_{timestamp}.db"
                                shutil.copy2(self.manager.db.db_path, current_backup)
                                
                                # Write uploaded file
                                with open(self.manager.db.db_path, 'wb') as f:
                                    f.write(backup_file.getvalue())
                                
                                # Reinitialize manager
                                self.manager = AlbumManager()
                                st.success("Backup restored successfully!")
                                st.info(f"Previous database backed up to: {current_backup}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error restoring backup: {str(e)}")
    
    def render_ai_analysis_page(self):
        """Render AI analysis page"""
        if not AI_AVAILABLE:
            st.error("AI features are not available. Please install the required packages.")
            st.code("pip install torch torchvision transformers diffusers")
            return
        
        st.title("🔍 AI Image Analysis")
        
        # Image selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get all people for selection
            people = self.manager.db.get_all_people()
            person_options = {p['display_name']: p['person_id'] for p in people}
            
            selected_person = st.selectbox(
                "Select Person",
                list(person_options.keys()),
                key="ai_analysis_person"
            )
        
        with col2:
            if selected_person:
                person_id = person_options[selected_person]
                
                # Get person's images
                entries = self.manager.get_entries_by_person(person_id, page=1)
                image_options = {}
                for entry in entries.get('entries', []):
                    image_options[entry.get('caption', 'Untitled')] = entry.get('image_id')
                
                selected_image_caption = st.selectbox(
                    "Select Image",
                    list(image_options.keys()),
                    key="ai_analysis_image"
                )
        
        # Display and analyze selected image
        if selected_person and selected_image_caption:
            image_id = image_options[selected_image_caption]
            
            # Get entry details
            entry = self.manager.get_entry_with_details(image_id)
            
            if entry:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display image
                    if entry.get('image_data_url'):
                        st.image(entry['image_data_url'], caption=entry.get('caption'), use_column_width=True)
                    
                    # Image info
                    st.markdown(f"**📏 Dimensions:** {entry.get('dimensions', 'N/A')}")
                    st.markdown(f"**📂 Format:** {entry.get('format', 'N/A')}")
                    st.markdown(f"**🗓️ Date:** {entry.get('created_at', 'N/A')}")
                
                with col2:
                    # Run AI analysis
                    if st.button("🤖 Analyze with AI", type="primary", use_container_width=True):
                        # Get image path
                        if entry.get('filepath'):
                            image_path = Config.DATA_DIR / entry['filepath']
                            
                            if image_path.exists():
                                with st.spinner("Analyzing image with AI..."):
                                    # Perform AI analysis
                                    analysis = self.manager.analyze_image_with_ai(image_path)
                                    
                                    # Display results
                                    EnhancedUIComponents.ai_analysis_panel(analysis)
                                    
                                    # Get suggestions
                                    suggestions = self.manager.get_ai_suggestions(image_path)
                                    
                                    if suggestions:
                                        st.markdown("**💡 AI Suggestions**")
                                        for suggestion in suggestions:
                                            with st.container(border=True):
                                                st.markdown(f"**{suggestion['type'].title()}**")
                                                st.text(suggestion['suggestion'])
                                                st.caption(f"Priority: {suggestion['priority']}")
                            else:
                                st.error("Image file not found")
                        else:
                            st.error("No image path available")
                    else:
                        st.info("Click 'Analyze with AI' to start analysis")
            
            # Show AI modification interface
            st.divider()
            st.subheader("✨ Apply AI Modifications")
            
            if entry and entry.get('filepath'):
                image_path = Config.DATA_DIR / entry['filepath']
                if image_path.exists():
                    EnhancedUIComponents.ai_modification_interface(image_path, self.manager)
    
    def render_ai_studio_page(self):
        """Render AI studio for creative modifications"""
        if not AI_AVAILABLE:
            st.error("AI features are not available. Please install the required packages.")
            st.code("pip install torch torchvision transformers diffusers")
            return
        
        st.title("🎨 AI Studio")
        
        tab1, tab2, tab3 = st.tabs(["Creative Styles", "Text to Image", "Advanced Tools"])
        
        with tab1:
            st.subheader("Creative Style Transfer")
            
            # Image upload or selection
            image_source = st.radio(
                "Image Source",
                ["Select from Album", "Upload New Image"],
                horizontal=True,
                key="studio_image_source"
            )
            
            image = None
            image_path = None
            
            if image_source == "Select from Album":
                # Get recent images
                recent_entries = self.manager.get_recent_entries(limit=10)
                
                if recent_entries:
                    image_options = {e['caption']: e['entry_id'] for e in recent_entries}
                    selected_caption = st.selectbox("Select Image", list(image_options.keys()))
                    
                    if selected_caption:
                        entry_id = image_options[selected_caption]
                        entry = self.manager.get_entry_with_details(entry_id)
                        
                        if entry and entry.get('filepath'):
                            image_path = Config.DATA_DIR / entry['filepath']
                            if image_path.exists():
                                image = Image.open(image_path)
            else:
                uploaded_file = st.file_uploader(
                    "Upload an image",
                    type=['jpg', 'jpeg', 'png', 'webp'],
                    key="studio_upload"
                )
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    image_path = Path(uploaded_file.name)
            
            if image:
                # Display original
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)
                
                with col2:
                    # Style selection
                    style_presets = st.session_state.get('style_presets', {})
                    
                    # Custom style creator
                    with st.expander("🎨 Create Custom Style"):
                        custom_name = st.text_input("Style Name", key="custom_style_name")
                        custom_prompt = st.text_input("Prompt Suffix", key="custom_prompt")
                        custom_negative = st.text_input("Negative Prompt", key="custom_negative")
                        custom_guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5, key="custom_guidance")
                        custom_steps = st.slider("Steps", 10, 100, 30, key="custom_steps")
                        
                        if st.button("Save Custom Style", key="save_custom_style"):
                            if custom_name and custom_prompt:
                                self.manager.ai_processor.diffusion_styler.create_custom_style(
                                    custom_name,
                                    custom_prompt,
                                    custom_negative,
                                    custom_guidance,
                                    custom_steps
                                )
                                st.success(f"Style '{custom_name}' saved!")
                                st.rerun()
                    
                    # Select style
                    all_styles = list(style_presets.keys())
                    selected_style = st.selectbox("Select Style", all_styles, key="studio_style")
                    
                    # Style parameters
                    strength = st.slider("Strength", 0.1, 1.0, 0.7, 0.1, key="studio_strength")
                    num_variations = st.slider("Number of Variations", 1, 4, 1, key="studio_variations")
                    
                    if st.button("Generate Styled Images", type="primary", key="generate_styles"):
                        with st.spinner("Applying style transfer..."):
                            # Generate variations
                            variations = []
                            for i in range(num_variations):
                                with st.spinner(f"Generating variation {i+1}..."):
                                    result = self.manager.ai_processor.diffusion_styler.apply_style_transfer(
                                        image,
                                        selected_style,
                                        strength=strength
                                    )
                                    if result:
                                        variations.append(result)
                            
                            # Display results
                            if variations:
                                st.success(f"Generated {len(variations)} variations!")
                                
                                cols = st.columns(min(4, len(variations)))
                                for i, var in enumerate(variations):
                                    with cols[i % len(cols)]:
                                        st.image(var, caption=f"Variation {i+1}", use_column_width=True)
                                        
                                        # Offer download
                                        buf = io.BytesIO()
                                        var.save(buf, format='JPEG', quality=95)
                                        st.download_button(
                                            label=f"Download Var {i+1}",
                                            data=buf.getvalue(),
                                            file_name=f"styled_{selected_style}_{i+1}.jpg",
                                            mime="image/jpeg",
                                            key=f"download_var_{i}"
                                        )
        
        with tab2:
            st.subheader("Text to Image Generation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                prompt = st.text_area(
                    "Describe the image you want to create",
                    height=150,
                    placeholder="A beautiful sunset over mountains, digital art, vibrant colors, 4k",
                    key="text2img_prompt"
                )
                
                negative_prompt = st.text_input(
                    "What to avoid",
                    placeholder="blurry, distorted, ugly",
                    key="text2img_negative"
                )
                
                # Generation parameters
                num_images = st.slider("Number of images", 1, 4, 1, key="text2img_count")
                guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5, 0.5, key="text2img_guidance")
                num_steps = st.slider("Steps", 10, 100, 30, key="text2img_steps")
                seed = st.number_input("Seed (optional)", value=-1, key="text2img_seed")
            
            with col2:
                if st.button("Generate Images", type="primary", key="generate_text2img"):
                    if prompt:
                        with st.spinner("Generating images..."):
                            try:
                                # Load diffusion model
                                pipe = self.manager.ai_processor.model_manager.load_model("diffusion")
                                
                                if pipe:
                                    # Generate images
                                    generator = None
                                    if seed != -1:
                                        generator = torch.Generator(device=self.manager.ai_processor.model_manager.device).manual_seed(seed)
                                    
                                    images = pipe(
                                        prompt=prompt,
                                        negative_prompt=negative_prompt,
                                        guidance_scale=guidance_scale,
                                        num_inference_steps=num_steps,
                                        num_images_per_prompt=num_images,
                                        generator=generator
                                    ).images
                                    
                                    # Display results
                                    st.success(f"Generated {len(images)} images!")
                                    
                                    cols = st.columns(min(4, len(images)))
                                    for i, img in enumerate(images):
                                        with cols[i % len(cols)]:
                                            st.image(img, caption=f"Generated {i+1}", use_column_width=True)
                                            
                                            # Offer download
                                            buf = io.BytesIO()
                                            img.save(buf, format='JPEG', quality=95)
                                            st.download_button(
                                                label=f"Download {i+1}",
                                                data=buf.getvalue(),
                                                file_name=f"generated_{i+1}.jpg",
                                                mime="image/jpeg",
                                                key=f"download_gen_{i}"
                                            )
                                            
                                            # Option to add to album
                                            if st.button(f"Add to Album", key=f"add_gen_{i}"):
                                                # Save image
                                                save_path = Config.DATA_DIR / "generated" / f"gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                                                save_path.parent.mkdir(exist_ok=True)
                                                img.save(save_path)
                                                
                                                # Add to database
                                                metadata = ImageMetadata.from_image(save_path)
                                                thumbnail_path = self.manager.image_processor.create_thumbnail(save_path)
                                                thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
                                                
                                                self.manager.db.add_image(metadata, thumbnail_path_str)
                                                
                                                # Create album entry
                                                album_entry = AlbumEntry(
                                                    entry_id=str(uuid.uuid4()),
                                                    image_id=metadata.image_id,
                                                    person_id='generated',  # Special person ID for generated images
                                                    caption=f"AI Generated: {prompt[:50]}...",
                                                    description=f"Generated with prompt: {prompt}",
                                                    location="AI Studio",
                                                    date_taken=datetime.datetime.now(),
                                                    tags=['ai-generated', 'text-to-image'],
                                                    privacy_level='public',
                                                    created_by=st.session_state['username'],
                                                    created_at=datetime.datetime.now(),
                                                    updated_at=datetime.datetime.now()
                                                )
                                                
                                                self.manager.db.add_album_entry(album_entry)
                                                st.success("Added to album!")
                                else:
                                    st.error("Could not load diffusion model")
                                    
                            except Exception as e:
                                st.error(f"Error generating images: {str(e)}")
                    else:
                        st.warning("Please enter a prompt")
        
        with tab3:
            st.subheader("Advanced AI Tools")
            
            tool_option = st.selectbox(
                "Select Tool",
                ["Image Inpainting", "Upscaling", "Background Removal", "Colorization"],
                key="advanced_tool"
            )
            
            if tool_option == "Image Inpainting":
                st.markdown("**🖌️ Image Inpainting**")
                st.markdown("Upload an image and draw a mask to remove/replace objects")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    inpaint_image = st.file_uploader(
                        "Upload image for inpainting",
                        type=['jpg', 'jpeg', 'png'],
                        key="inpaint_upload"
                    )
                    
                    if inpaint_image:
                        img = Image.open(inpaint_image)
                        st.image(img, caption="Original", use_column_width=True)
                        
                        # Simple mask drawing
                        st.markdown("**Draw Mask**")
                        st.info("For this demo, we'll use a simple box selection")
                        
                        mask_type = st.radio(
                            "Mask Type",
                            ["Remove Object", "Replace Background"],
                            horizontal=True,
                            key="mask_type"
                        )
                        
                        if mask_type == "Remove Object":
                            prompt = st.text_input(
                                "What should fill the masked area?",
                                placeholder="Leave empty for automatic fill",
                                key="inpaint_prompt"
                            )
                        
                        if st.button("Apply Inpainting", key="apply_inpaint"):
                            st.info("Inpainting requires a proper mask. In production, integrate with a drawing tool.")
                
                with col2:
                    st.markdown("**How it works:**")
                    st.markdown("""
                    1. Upload your image
                    2. Draw a mask over the area you want to modify
                    3. Describe what you want in that area (optional)
                    4. AI will fill the masked area naturally
                    
                    **Use cases:**
                    - Remove unwanted objects
                    - Replace backgrounds
                    - Fix damaged photos
                    - Add new elements
                    """)
            
            elif tool_option == "Upscaling":
                st.markdown("**🔍 Image Upscaling**")
                
                upscale_image = st.file_uploader(
                    "Upload image to upscale",
                    type=['jpg', 'jpeg', 'png'],
                    key="upscale_upload"
                )
                
                if upscale_image:
                    img = Image.open(upscale_image)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(img, caption=f"Original ({img.width}x{img.height})", use_column_width=True)
                    
                    with col2:
                        scale_factor = st.slider("Upscale Factor", 1.5, 4.0, 2.0, 0.5, key="upscale_factor")
                        
                        if st.button("Upscale Image", key="upscale_button"):
                            with st.spinner("Upscaling image..."):
                                # Use diffusion-based upscaling or simple interpolation
                                upscaled = self.manager.ai_processor.diffusion_styler.upscale_image(img, scale_factor)
                                
                                st.image(upscaled, caption=f"Upscaled ({upscaled.width}x{upscaled.height})", use_column_width=True)
                                
                                # Offer download
                                buf = io.BytesIO()
                                upscaled.save(buf, format='JPEG', quality=95)
                                st.download_button(
                                    label="Download Upscaled",
                                    data=buf.getvalue(),
                                    file_name=f"upscaled_{scale_factor}x.jpg",
                                    mime="image/jpeg"
                                )
    
    def render_quick_enhance_page(self):
        """Render quick enhancement page"""
        if not AI_AVAILABLE:
            st.error("AI features are not available. Please install the required packages.")
            st.code("pip install torch torchvision transformers diffusers")
            return
        
        st.title("⚡ Quick Enhance")
        
        # Recent images grid
        st.markdown("### Recent Photos")
        
        recent_entries = self.manager.get_recent_entries(limit=12)
        
        if recent_entries:
            cols = st.columns(4)
            selected_images = []
            
            for idx, entry in enumerate(recent_entries):
                with cols[idx % 4]:
                    with st.container(border=True):
                        # Thumbnail
                        if entry.get('thumbnail_data_url'):
                            st.image(entry['thumbnail_data_url'], use_column_width=True)
                        
                        # Caption
                        st.markdown(f"**{entry.get('caption', 'Untitled')[:20]}...**")
                        
                        # Checkbox for selection
                        if st.checkbox("Select", key=f"select_{entry['entry_id']}"):
                            selected_images.append(entry['entry_id'])
            
            # Batch actions
            if selected_images:
                st.divider()
                st.markdown(f"**{len(selected_images)} images selected**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🤖 Analyze All", use_container_width=True):
                        st.session_state['batch_analysis_ids'] = selected_images
                        st.session_state['current_page'] = 'batch_analysis'
                        st.rerun()
                
                with col2:
                    if st.button("⚡ Auto Enhance All", use_container_width=True):
                        with st.spinner("Enhancing images..."):
                            enhanced_count = 0
                            for entry_id in selected_images:
                                entry = self.manager.get_entry_with_details(entry_id)
                                if entry and entry.get('filepath'):
                                    image_path = Config.DATA_DIR / entry['filepath']
                                    if image_path.exists():
                                        result = self.manager.apply_ai_modification(image_path, 'auto_enhance')
                                        if result:
                                            enhanced_count += 1
                            
                            st.success(f"Enhanced {enhanced_count} images!")
                
                with col3:
                    style = st.selectbox(
                        "Apply Style",
                        list(st.session_state.get('style_presets', {}).keys()),
                        key="batch_style_select"
                    )
                    
                    if st.button(f"🎨 Apply {style}", use_container_width=True):
                        with st.spinner(f"Applying {style} style..."):
                            styled_count = 0
                            for entry_id in selected_images:
                                entry = self.manager.get_entry_with_details(entry_id)
                                if entry and entry.get('filepath'):
                                    image_path = Config.DATA_DIR / entry['filepath']
                                    if image_path.exists():
                                        result = self.manager.apply_ai_modification(
                                            image_path,
                                            'style_transfer',
                                            style=style
                                        )
                                        if result:
                                            styled_count += 1
                            
                            st.success(f"Applied {style} style to {styled_count} images!")
        else:
            st.info("No recent photos found. Scan your directory first.")
    
    def render_ai_insights_page(self):
        """Render AI insights page"""
        if not AI_AVAILABLE:
            st.error("AI features are not available. Please install the required packages.")
            st.code("pip install torch torchvision transformers diffusers")
            return
        
        st.title("📊 AI Insights")
        
        # Get all images for analysis
        all_people = self.manager.get_all_people_with_stats()
        
        if not all_people:
            st.info("No data available. Scan your directory first.")
            return
        
        # Overall AI insights
        st.subheader("🔍 Overall Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Most common tags across all images
            st.markdown("**🏷️ Most Common Tags**")
            
            # Sample analysis
            sample_tags = []
            for person in all_people[:5]:  # Limit for demo
                person_id = person['person_id']
                entries = self.manager.get_entries_by_person(person_id, page=1, limit=5)
                for entry in entries.get('entries', []):
                    if entry.get('tags'):
                        tags = entry['tags'].split(',') if isinstance(entry['tags'], str) else entry['tags']
                        sample_tags.extend(tags)
            
            if sample_tags:
                from collections import Counter
                tag_counts = Counter(sample_tags)
                for tag, count in tag_counts.most_common(10):
                    st.text(f"• {tag}: {count}")
            else:
                st.text("No tags found")
        
        with col2:
            st.markdown("**🎨 Style Recommendations**")
            
            recommendations = [
                "Portrait photos benefit from painting styles",
                "Landscape images work well with fantasy styles",
                "Group photos can be enhanced with vintage filters",
                "Low-light images need brightness adjustment"
            ]
            
            for rec in recommendations:
                st.text(f"• {rec}")
        
        with col3:
            st.markdown("**📈 AI Usage Stats**")
            
            ai_stats = {
                "Style Transfers": len([h for h in st.session_state.get('generation_history', []) 
                                      if h.get('type') == 'style_transfer']),
                "Text Edits": len([h for h in st.session_state.get('generation_history', []) 
                                 if h.get('type') == 'text_edit']),
                "Auto Enhancements": len(st.session_state.get('ai_modifications', {}))
            }
            
            for stat, value in ai_stats.items():
                st.metric(stat, value)
        
        # AI-generated suggestions
        st.divider()
        st.subheader("💡 AI Suggestions for Your Album")
        
        suggestions_tab1, suggestions_tab2, suggestions_tab3 = st.tabs(["Content", "Organization", "Enhancement"])
        
        with suggestions_tab1:
            st.markdown("**Content Suggestions**")
            
            content_suggestions = [
                "Consider adding more landscape photos for variety",
                "Group photos by events or themes",
                "Add descriptive captions to older photos",
                "Consider scanning physical photos to digital"
            ]
            
            for suggestion in content_suggestions:
                with st.container(border=True):
                    st.markdown(suggestion)
                    if st.button("Apply Suggestion", key=f"content_{hash(suggestion)}"):
                        st.info("This would trigger an automated process in production")
        
        with suggestions_tab2:
            st.markdown("**Organization Suggestions**")
            
            org_suggestions = [
                "Create albums by year for better navigation",
                "Add location tags to travel photos",
                "Use facial recognition to group people",
                "Create themed collections (holidays, vacations)"
            ]
            
            for suggestion in org_suggestions:
                with st.container(border=True):
                    st.markdown(suggestion)
        
        with suggestions_tab3:
            st.markdown("**Enhancement Suggestions**")
            
            enhance_suggestions = [
                "Batch process all photos with auto-enhance",
                "Apply consistent filters to event photos",
                "Upscale older low-resolution photos",
                "Colorize black and white photos"
            ]
            
            for suggestion in enhance_suggestions:
                with st.container(border=True):
                    st.markdown(suggestion)
                    if st.button("Run Enhancement", key=f"enhance_{hash(suggestion)}"):
                        st.info("Batch processing would start in production")
        
        # AI model management
        st.divider()
        st.subheader("🤖 AI Model Management")
        
        with st.expander("Model Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Loaded Models**")
                loaded_models = list(self.manager.ai_processor.model_manager.model_cache.keys())
                
                if loaded_models:
                    for model in loaded_models:
                        st.text(f"• {model}")
                else:
                    st.text("No models loaded")
                
                if st.button("Load All Models", key="load_all_models"):
                    with st.spinner("Loading AI models..."):
                        # Load caption model
                        self.manager.ai_processor.model_manager.load_model("caption")
                        # Load diffusion model
                        self.manager.ai_processor.model_manager.load_model("diffusion")
                        st.success("Models loaded!")
                        st.rerun()
            
            with col2:
                st.markdown("**Memory Management**")
                
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    memory_reserved = torch.cuda.memory_reserved() / 1e9
                    
                    st.metric("GPU Memory Used", f"{memory_allocated:.2f} GB")
                    st.metric("GPU Memory Reserved", f"{memory_reserved:.2f} GB")
                
                if st.button("Clear Model Cache", key="clear_model_cache"):
                    self.manager.ai_processor.model_manager.clear_all_models()
                    st.success("Model cache cleared!")
                    st.rerun()
                
                if st.button("Unload Heavy Models", key="unload_heavy"):
                    self.manager.ai_processor.model_manager.unload_model("diffusion")
                    self.manager.ai_processor.model_manager.unload_model("inpaint")
                    st.success("Heavy models unloaded!")
                    st.rerun()
    
    def render_batch_analysis_page(self):
        """Render batch analysis results page"""
        if not AI_AVAILABLE:
            st.error("AI features are not available.")
            return
        
        st.title("🔍 Batch Analysis Results")
        
        if 'batch_analysis_ids' not in st.session_state or not st.session_state.batch_analysis_ids:
            st.warning("No images selected for batch analysis")
            if st.button("Back to Quick Enhance"):
                st.session_state['current_page'] = 'quick_enhance'
                st.rerun()
            return
        
        entry_ids = st.session_state.batch_analysis_ids
        
        # Progress tracking
        if 'batch_analysis_progress' not in st.session_state:
            st.session_state.batch_analysis_progress = 0
            st.session_state.batch_analysis_results = {}
        
        progress_bar = st.progress(st.session_state.batch_analysis_progress)
        status_text = st.empty()
        
        # Analyze each image
        for i, entry_id in enumerate(entry_ids):
            if entry_id not in st.session_state.batch_analysis_results:
                status_text.text(f"Analyzing image {i+1} of {len(entry_ids)}...")
                
                entry = self.manager.get_entry_with_details(entry_id)
                if entry and entry.get('filepath'):
                    image_path = Config.DATA_DIR / entry['filepath']
                    if image_path.exists():
                        analysis = self.manager.analyze_image_with_ai(image_path)
                        st.session_state.batch_analysis_results[entry_id] = {
                            'analysis': analysis,
                            'caption': entry.get('caption'),
                            'image_path': image_path
                        }
                
                # Update progress
                st.session_state.batch_analysis_progress = (i + 1) / len(entry_ids)
                progress_bar.progress(st.session_state.batch_analysis_progress)
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success(f"Analysis complete! Processed {len(st.session_state.batch_analysis_results)} images")
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        all_tags = []
        all_captions = []
        
        for result in st.session_state.batch_analysis_results.values():
            analysis = result.get('analysis', {})
            tags = analysis.get('tags', [])
            all_tags.extend(tags)
            
            caption = result.get('caption', '')
            if caption:
                all_captions.append(caption)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Tag Cloud**")
            if all_tags:
                from collections import Counter
                tag_counts = Counter(all_tags)
                
                # Create tag cloud visualization
                cloud_html = '<div style="line-height: 2.5;">'
                for tag, count in tag_counts.most_common(20):
                    size = min(24, 12 + count * 2)
                    cloud_html += f'<span style="font-size: {size}px; margin: 5px; display: inline-block;">{tag}</span>'
                cloud_html += '</div>'
                
                st.markdown(cloud_html, unsafe_allow_html=True)
            else:
                st.text("No tags generated")
        
        with col2:
            st.markdown("**Common Themes**")
            
            # Simple theme detection
            themes = {}
            for caption in all_captions:
                words = caption.lower().split()
                for word in words:
                    if len(word) > 4 and word not in ['photo', 'image', 'picture']:
                        themes[word] = themes.get(word, 0) + 1
            
            if themes:
                for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True)[:10]:
                    st.text(f"• {theme}: {count} images")
            else:
                st.text("No common themes detected")
        
        # Detailed results
        st.divider()
        st.subheader("📋 Detailed Results")
        
        for entry_id, result in st.session_state.batch_analysis_results.items():
            with st.expander(f"📸 {result.get('caption', 'Unknown')}"):
                analysis = result.get('analysis', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display thumbnail
                    image_path = result.get('image_path')
                    if image_path and image_path.exists():
                        st.image(str(image_path), width=200)
                    
                    # Basic info
                    st.markdown(f"**Caption:** {analysis.get('caption', 'N/A')}")
                    
                    # Tags
                    tags = analysis.get('tags', [])
                    if tags:
                        st.markdown("**Tags:**")
                        st.markdown(UIComponents.tag_badges(tags[:5]), unsafe_allow_html=True)
                
                with col2:
                    # Composition analysis
                    comp = analysis.get('composition', {})
                    if comp:
                        st.markdown("**Composition Analysis:**")
                        
                        metrics = ['brightness', 'contrast', 'sharpness', 'color_balance']
                        for metric in metrics:
                            value = comp.get(metric, 'N/A')
                            st.text(f"• {metric.title()}: {value}")
                        
                        suggestions = comp.get('suggestions', [])
                        if suggestions:
                            st.markdown("**Suggestions:**")
                            for suggestion in suggestions[:3]:
                                st.text(f"• {suggestion}")
        
        # Clear batch analysis
        if st.button("Clear Batch Results", key="clear_batch"):
            del st.session_state.batch_analysis_ids
            del st.session_state.batch_analysis_progress
            del st.session_state.batch_analysis_results
            st.rerun()
    
    def render_main(self):
        """Main application renderer"""
        # Render sidebar
        self.render_sidebar()
        
        # Get current page
        current_page = st.session_state.get('current_page', 'dashboard')
        
        # Render appropriate page
        if current_page == 'dashboard':
            self.render_dashboard()
        elif current_page == 'people':
            self.render_people_page()
        elif current_page == 'gallery':
            self.render_gallery_page()
        elif current_page == 'image_detail':
            self.render_image_detail_page()
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
        elif current_page == 'ai_analysis':
            self.render_ai_analysis_page()
        elif current_page == 'ai_studio':
            self.render_ai_studio_page()
        elif current_page == 'quick_enhance':
            self.render_quick_enhance_page()
        elif current_page == 'ai_insights':
            self.render_ai_insights_page()
        elif current_page == 'batch_analysis':
            self.render_batch_analysis_page()
        else:
            self.render_dashboard()
        
        # Footer
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} v{Config.VERSION}")
            st.caption("A comprehensive photo album management system" + (" with AI enhancements" if AI_AVAILABLE else ""))

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================
def main():
    """Main application entry point"""
    try:
        # Initialize app
        app = PhotoAlbumApp()
        
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

# ============================================================================
# REQUIREMENTS GENERATOR
# ============================================================================
def generate_requirements():
    """Generate requirements.txt content"""
    requirements = """# Core dependencies
streamlit>=1.28.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0

# AI/ML dependencies (optional)
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
diffusers>=0.19.0
accelerate>=0.21.0

# Image processing (optional)
opencv-python>=4.8.0
scipy>=1.11.0
"""
    return requirements

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    # Show requirements in sidebar for easy access
    with st.sidebar.expander("📋 Requirements"):
        st.code(generate_requirements(), language="text")
        
        if not AI_AVAILABLE:
            st.warning("""
            ⚠️ AI libraries not installed. 
            Running in basic mode.
            
            To enable AI features:
            ```bash
            pip install torch torchvision transformers diffusers
            ```
            """)
    
    # Run main application
    main()
