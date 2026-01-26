"""
COMPREHENSIVE WEB PHOTO ALBUM APPLICATION
Version: 2.3.1 - Syntax Fixed
Features: Table of Contents, Image Gallery, Comments, Ratings, Metadata Management, Search, and More
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ExifTags
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

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
class Config:
    """Application configuration constants"""
    APP_NAME = "MemoryVault Pro"
    VERSION = "2.3.1"
    
    # Get absolute paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    
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
    
    @classmethod
    def init_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.THUMBNAIL_DIR,
            cls.METADATA_DIR,
            cls.DB_DIR,
            cls.EXPORT_DIR
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
                    # Create a simple colored image as placeholder
                    from PIL import Image, ImageDraw, ImageFont
                    
                    # Create image with gradient background
                    img = Image.new('RGB', (400, 300), color='#667eea')
                    draw = ImageDraw.Draw(img)
                    
                    # Draw a simple person icon
                    draw.ellipse((150, 50, 250, 150), fill='#ffffff', outline='#4a5568')
                    draw.rectangle((150, 150, 250, 280), fill='#ffffff', outline='#4a5568')
                    
                    # Add text
                    try:
                        font = ImageFont.truetype("arial.ttf", 24)
                    except:
                        font = ImageFont.load_default()
                    
                    draw.text((120, 250), person.replace('-', ' ').title(), fill='#2d3748', font=font)
                    
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
# DATA MODELS - FIXED SYNTAX ERRORS
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
            st.warning(f"Error extracting EXIF data: {str(e)}")
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
    privacy_level: str  # public, private, shared
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
# DATABASE MANAGEMENT - FIXED VERSION
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
        """Initialize database with tables - FIXED VERSION"""
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create tables - FIXED COLUMN NAMES
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
                CREATE TABLE IF NOT EXISTS tags (
                    tag_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS entry_tags (
                    entry_id TEXT,
                    tag_id TEXT,
                    PRIMARY KEY (entry_id, tag_id),
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id),
                    FOREIGN KEY (tag_id) REFERENCES tags (tag_id)
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
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT,
                    view_count INTEGER DEFAULT 0,
                    like_count INTEGER DEFAULT 0,
                    share_count INTEGER DEFAULT 0,
                    last_viewed TIMESTAMP,
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_person ON album_entries(person_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_created ON album_entries(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_entry ON comments(entry_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_entry ON ratings(entry_id)')
                
                conn.commit()
                st.success("Database initialized successfully!")
                
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {str(e)}")
            raise
    
    def add_image(self, metadata, thumbnail_path: str = None):
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
                return (result[0] or 0, result[1] or 0)
        except sqlite3.Error as e:
            st.error(f"Error retrieving ratings: {str(e)}")
            return (0, 0)
    
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
        """Get detailed information for an entry - FIXED VERSION"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Simplified query to avoid column name issues
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
                
                # Convert to RGB if necessary (for GIF, PNG with transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                
                # Create thumbnail
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                
                # Save thumbnail as JPG for consistency
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
    
    @staticmethod
    def extract_exif_metadata(image_path: Path) -> Dict[str, Any]:
        """Extract detailed EXIF metadata"""
        metadata = {}
        try:
            if not image_path.exists():
                return metadata
                
            with Image.open(image_path) as img:
                exif = {}
                if hasattr(img, '_getexif') and img._getexif():
                    raw_exif = img._getexif()
                    for tag_id, value in raw_exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        # Skip large binary data
                        if not isinstance(value, (bytes, np.ndarray)):
                            exif[tag] = str(value)
                return exif
        except Exception as e:
            st.warning(f"Error extracting EXIF from {image_path}: {str(e)}")
            return metadata

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
                # Clean expired cache
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
                'view_mode': 'grid',  # 'grid', 'list', 'detail'
                'sort_by': 'date',
                'sort_order': 'desc',
                'selected_tags': [],
                'user_id': str(uuid.uuid4()),
                'username': 'Guest',
                'user_role': UserRoles.VIEWER.value,
                'favorites': set(),
                'recently_viewed': [],
                'comments_cache': {},
                'ratings_cache': {},
                'analytics_cache': {},
                'toc_page': 1,
                'gallery_page': 1,
                'show_directory_info': True
            })
    
    def scan_directory(self, data_dir: Path = None) -> Dict:
        """Scan directory for images and update database with robust error handling"""
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
            
            # Find all person directories - more flexible matching
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
            
            # First count total files to process
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
                # Handle person name formatting more flexibly
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
                
                # Process images in directory
                image_files = [
                    f for f in person_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS
                ]
                
                for img_path in image_files:
                    try:
                        processed_files += 1
                        progress = processed_files / max(total_files, 1)
                        progress_bar.progress(progress)
                        
                        # Calculate checksum to check for duplicates
                        checksum = ImageMetadata._calculate_checksum(img_path)
                        
                        # Check if image already exists in database
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
            
            # Clear cache after scanning
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
                    
                    # Count images
                    cursor.execute('''
                    SELECT COUNT(*)
                    FROM album_entries
                    WHERE person_id = ?
                    ''', (person_id,))
                    image_count = cursor.fetchone()[0]
                    
                    # Count comments
                    cursor.execute('''
                    SELECT COUNT(DISTINCT c.comment_id)
                    FROM comments c
                    JOIN album_entries ae ON c.entry_id = ae.entry_id
                    WHERE ae.person_id = ?
                    ''', (person_id,))
                    comment_count = cursor.fetchone()[0]
                    
                    # Average rating
                    cursor.execute('''
                    SELECT AVG(r.rating_value)
                    FROM ratings r
                    JOIN album_entries ae ON r.entry_id = ae.entry_id
                    WHERE ae.person_id = ?
                    ''', (person_id,))
                    avg_rating = cursor.fetchone()[0] or 0
                    
                    # Recent activity
                    cursor.execute('''
                    SELECT MAX(ae.created_at)
                    FROM album_entries ae
                    WHERE ae.person_id = ?
                    ''', (person_id,))
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
                    'avg_rating': 0,
                    'last_activity': None
                }
        
        return self.cache.get_or_set(cache_key, generate_stats)
    
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
            
            # Update session state
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
                DELETE FROM user_favorites
                WHERE user_id = ? AND entry_id = ?
                ''', (user_id, entry_id))
                conn.commit()
            
            # Update session state
            if 'favorites' in st.session_state and entry_id in st.session_state.favorites:
                st.session_state.favorites.remove(entry_id)
        except sqlite3.Error as e:
            st.error(f"Error removing from favorites: {str(e)}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PhotoAlbumApp:
    """Main application class with robust error handling"""
    
    def __init__(self):
        self.album_manager = AlbumManager()
        self.setup_page_config()
        self.load_custom_css()
        self.ensure_initialized()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=f"{Config.APP_NAME} v{Config.VERSION}",
            page_icon="🖼️",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/yourusername/photo-album',
                'Report a bug': 'https://github.com/yourusername/photo-album/issues',
                'About': f"""
                # {Config.APP_NAME}
                A comprehensive web album application with advanced features
                for managing, viewing, and sharing photo collections.
                Version: {Config.VERSION}
                """
            }
        )
    
    def load_custom_css(self):
        """Load custom CSS styles"""
        st.markdown("""
        <style>
        /* Main layout */
        .main {
            padding: 1rem 2rem;
            max-width: 1800px;
            margin: 0 auto;
        }
        /* Headers */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 0 0 20px 20px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }
        .main-header h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }
        .main-header p {
            font-size: 1.2rem;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto;
        }
        /* Section styling */
        .section-container {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
            border: 1px solid #eaeaea;
        }
        .section-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        /* Directory Info */
        .directory-info {
            background: #e3f2fd;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
        }
        .directory-info pre {
            background: white;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            margin: 10px 0;
        }
        /* Error message styling */
        .error-message {
            background: #ffebee;
            border-left: 4px solid #f44336;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin: 10px 0;
        }
        /* Image Card */
        .image-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .image-card-img {
            aspect-ratio: 1/1;
            overflow: hidden;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .image-card-img img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease;
        }
        .image-card-img img:hover {
            transform: scale(1.05);
        }
        .image-card-content {
            padding: 16px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }
        .image-card-title {
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 8px;
            color: #333;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .image-card-date {
            color: #666;
            font-size: 12px;
            margin-bottom: 8px;
        }
        .image-card-stats {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
            color: #666;
            font-size: 14px;
        }
        /* Utility Classes */
        .flex-center {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .text-gradient {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin: 2px;
        }
        .badge-primary {
            background: #667eea;
            color: white;
        }
        .badge-success {
            background: #48bb78;
            color: white;
        }
        .badge-warning {
            background: #ed8936;
            color: white;
        }
        .badge-info {
            background: #4299e1;
            color: white;
        }
        .no-photos-message {
            text-align: center;
            padding: 40px;
            color: #666;
            border: 2px dashed #e2e8f0;
            border-radius: 12px;
            background: #f8fafc;
        }
        .sample-image {
            width: 100%;
            height: 200px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            margin: 20px 0;
            font-size: 24px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def ensure_initialized(self):
        """Ensure application is properly initialized"""
        try:
            # Check if database is accessible
            with sqlite3.connect(self.album_manager.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM people')
                count = cursor.fetchone()[0]
                st.success(f"✅ Database initialized with {count} people")
        except Exception as e:
            st.error(f"⚠️ Database initialization issue: {str(e)}")
            st.info("🔄 The application will attempt to recover on the next run")
    
    def render_directory_info(self):
        """Show helpful information about directory structure"""
        if st.session_state.get('show_directory_info', True):
            st.markdown("""
            <div class="directory-info">
                <h3>📁 How to Set Up Your Photo Album</h3>
                <p>To display your photos, organize them in the <code>data/</code> directory with this structure:</p>
                <pre>
                data/
                ├── john-smith/          # Person folder (hyphens, underscores, or spaces)
                │   ├── vacation.jpg    # Your photos
                │   ├── birthday.png
                │   └── ...
                ├── sarah-johnson/      # Another person folder
                │   └── family.jpg
                └── ...
                </pre>
                <p><strong>Key points:</strong></p>
                <ul>
                    <li>Create a folder for each person</li>
                    <li>Folder names can use hyphens, underscores, or spaces</li>
                    <li>Supported image formats: JPG, JPEG, PNG, GIF, BMP, WEBP, TIFF</li>
                    <li>After adding photos, click "📸 Scan for New Photos" in the sidebar</li>
                </ul>
                <p><em>This message will automatically hide once you have photos in your album.</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render main application header"""
        st.markdown(f"""
        <div class="main-header">
            <h1>🖼️ {Config.APP_NAME}</h1>
            <p>Your personal photo gallery with advanced features</p>
            <div style="
                display: flex;
                justify-content: center;
                gap: 10px;
                margin-top: 1rem;
                flex-wrap: wrap;
            ">
                <span class="badge badge-primary">Version {Config.VERSION}</span>
                <span class="badge badge-success">👨‍👩‍👧‍👦 Photo Gallery</span>
                <span class="badge badge-warning">💬 Comments</span>
                <span class="badge badge-info">⭐ Ratings</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render application sidebar with error handling"""
        with st.sidebar:
            st.markdown("## 👆 Navigation")
            
            # User info
            with st.expander("👤 User Profile", expanded=False):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"""
                    <div style="
                        width: 60px;
                        height: 60px;
                        border-radius: 50%;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-size: 24px;
                        margin: 0 auto;
                    ">
                    {st.session_state.username[0].upper()}
                    </div>
                    """, unsafe_allow_html=True)
                
                username = st.text_input("Username", value=st.session_state.username)
                if username and username != st.session_state.username:
                    st.session_state.username = username
                
                role_options = [role.value for role in UserRoles]
                current_role = st.session_state.user_role
                role_index = role_options.index(current_role) if current_role in role_options else 0
                
                new_role = st.selectbox(
                    "Role",
                    options=role_options,
                    index=role_index
                )
                if new_role != st.session_state.user_role:
                    st.session_state.user_role = new_role
            
            # Quick Stats
            st.markdown("---")
            st.markdown("## 📊 Quick Stats")
            
            try:
                with sqlite3.connect(self.album_manager.db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get statistics with error handling
                    stats = {}
                    queries = {
                        'people': 'SELECT COUNT(*) FROM people',
                        'photos': 'SELECT COUNT(*) FROM album_entries',
                        'comments': 'SELECT COUNT(*) FROM comments',
                        'avg_rating': 'SELECT AVG(rating_value) FROM ratings'
                    }
                    
                    for key, query in queries.items():
                        try:
                            cursor.execute(query)
                            result = cursor.fetchone()
                            stats[key] = result[0] if result else 0
                        except sqlite3.Error as e:
                            st.warning(f"Could not get {key} count: {str(e)}")
                            stats[key] = 0
                    
                    # Display stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("👤 People", stats.get('people', 0))
                        st.metric("💬 Comments", stats.get('comments', 0))
                    with col2:
                        st.metric("🖼️ Photos", stats.get('photos', 0))
                        st.metric("⭐ Avg Rating", f"{stats.get('avg_rating', 0):.1f}")
                        
            except Exception as e:
                st.error(f"Error getting stats: {str(e)}")
            
            # Actions - Always show scan button prominently
            st.markdown("---")
            st.markdown("## ⚡ Quick Actions")
            
            if st.button("📸 Scan for New Photos", use_container_width=True, type="primary"):
                with st.spinner("🔍 Scanning directory for photos..."):
                    results = self.album_manager.scan_directory()
                
                # Show detailed results
                results_message = f"""
                ✅ Scan completed successfully!
                • People found: {results['people_found']}
                • New images: {results['new_images']}
                • Updated images: {results['updated_images']}
                • Total images processed: {results['total_images']}
                """
                
                if results['errors']:
                    results_message += f"\n\n⚠️ {len(results['errors'])} errors occurred:"
                    for i, error in enumerate(results['errors'][:3]):  # Show first 3 errors
                        results_message += f"\n• {error}"
                    if len(results['errors']) > 3:
                        results_message += f"\n• ...and {len(results['errors'])-3} more"
                
                st.success(results_message)
                
                # Hide directory info if we found photos
                if results['total_images'] > 0:
                    st.session_state.show_directory_info = False
                
                st.rerun()
            
            if st.button("🗂️ Clear Cache", use_container_width=True):
                self.album_manager.cache.clear()
                st.success("✅ Cache cleared successfully!")
                st.rerun()
    
    def render_dashboard(self):
        """Render main dashboard with error handling"""
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Dashboard Overview</div>', unsafe_allow_html=True)
        
        try:
            # Check if we have any photos
            with sqlite3.connect(self.album_manager.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM album_entries')
                total_photos = cursor.fetchone()[0]
                
                if total_photos == 0:
                    self.render_directory_info()
                    st.markdown('</div>', unsafe_allow_html=True)
                    return
        
            # Get stats with error handling
            stats = {}
            try:
                with sqlite3.connect(self.album_manager.db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    queries = {
                        'people': 'SELECT COUNT(*) FROM people',
                        'photos': 'SELECT COUNT(*) FROM album_entries',
                        'comments': 'SELECT COUNT(*) FROM comments',
                        'avg_rating': 'SELECT AVG(rating_value) FROM ratings'
                    }
                    
                    for key, query in queries.items():
                        cursor.execute(query)
                        result = cursor.fetchone()
                        stats[key] = result[0] if result else 0
            except Exception as e:
                st.error(f"Error getting dashboard stats: {str(e)}")
                stats = {'people': 0, 'photos': 0, 'comments': 0, 'avg_rating': 0}
            
            # Stats cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("👤 People", stats.get('people', 0))
            with col2:
                st.metric("🖼️ Photos", stats.get('photos', 0))
            with col3:
                st.metric("💬 Comments", stats.get('comments', 0))
            with col4:
                st.metric("⭐ Avg Rating", f"{stats.get('avg_rating', 0):.1f}")
            
            # Recent photos
            st.markdown("---")
            st.markdown("### 📸 Recently Added Photos")
            
            try:
                with sqlite3.connect(self.album_manager.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT ae.*, p.display_name, i.thumbnail_path
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN images i ON ae.image_id = i.image_id
                    ORDER BY ae.created_at DESC
                    LIMIT 6
                    ''')
                    recent_photos = [dict(row) for row in cursor.fetchall()]
                
                if recent_photos:
                    cols = st.columns(3)
                    for idx, photo in enumerate(recent_photos):
                        with cols[idx % 3]:
                            thumbnail_path = photo.get('thumbnail_path', '')
                            img_url = ""
                            
                            if thumbnail_path and Path(thumbnail_path).exists():
                                img_url = self.album_manager.image_processor.get_image_data_url(Path(thumbnail_path))
                            else:
                                img_path = Config.DATA_DIR / photo['filepath']
                                if img_path.exists():
                                    img_url = self.album_manager.image_processor.get_image_data_url(img_path)
                            
                            if img_url:
                                st.image(img_url, use_container_width=True)
                            else:
                                st.markdown('<div class="sample-image">Sample Photo</div>', unsafe_allow_html=True)
                            
                            st.caption(f"**{photo['caption']}**")
                            st.caption(f"By: {photo['display_name']}")
                            st.caption(f"Added: {photo['created_at'][:10]}")
                            
                            if st.button("View", key=f"view_recent_{photo['entry_id']}", use_container_width=True):
                                st.session_state.selected_image = photo['entry_id']
                                st.rerun()
                else:
                    st.info("No photos available yet. Add some photos to get started!")
            except Exception as e:
                st.error(f"Error loading recent photos: {str(e)}")
                st.info("No photos available yet. Add some photos to get started!")
        
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
        finally:
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_table_of_contents(self):
        """Render interactive table of contents with error handling"""
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Table of Contents</div>', unsafe_allow_html=True)
        
        try:
            # Get all people
            all_people = self.album_manager.db.get_all_people()
            
            # Show directory info if no people found
            if not all_people:
                self.render_directory_info()
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            # Search and filter
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                search_query = st.text_input("🔍 Search people...", 
                                           placeholder="Type to filter...",
                                           key="toc_search")
            
            with col2:
                sort_option = st.selectbox("Sort by", ["Name", "Photo Count", "Recent Activity"], 
                                         key="toc_sort")
            
            with col3:
                items_per_page = st.slider("Items per page", 5, 50, 10, key="toc_items_per_page")
            
            # Filter based on search
            if search_query:
                all_people = [
                    p for p in all_people 
                    if search_query.lower() in p['display_name'].lower()
                ]
            
            # Sort
            if sort_option == "Name":
                all_people.sort(key=lambda x: x['display_name'])
            elif sort_option == "Photo Count":
                for person in all_people:
                    stats = self.album_manager.get_person_stats(person['person_id'])
                    person['_photo_count'] = stats['image_count']
                all_people.sort(key=lambda x: x.get('_photo_count', 0), reverse=True)
            elif sort_option == "Recent Activity":
                for person in all_people:
                    stats = self.album_manager.get_person_stats(person['person_id'])
                    person['_last_activity'] = stats['last_activity'] or ''
                all_people.sort(key=lambda x: x.get('_last_activity', ''), reverse=True)
            
            # Pagination
            total_pages = max(1, (len(all_people) + items_per_page - 1) // items_per_page)
            page_number = st.session_state.get('toc_page', 1)
            
            # Page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous", disabled=page_number <= 1, key="toc_prev_page"):
                    st.session_state.toc_page = page_number - 1
                    st.rerun()
            with col2:
                st.markdown(f"**Page {page_number} of {total_pages}**")
            with col3:
                if st.button("Next ➡️", disabled=page_number >= total_pages, key="toc_next_page"):
                    st.session_state.toc_page = page_number + 1
                    st.rerun()
            
            # Display current page
            start_idx = (page_number - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(all_people))
            current_people = all_people[start_idx:end_idx]
            
            # Create TOC items
            for person in current_people:
                person_id = person['person_id']
                stats = self.album_manager.get_person_stats(person_id)
                is_active = st.session_state.get('selected_person') == person_id
                
                # Create card for each person
                with st.container():
                    is_clicked = st.button(
                        f"{person['display_name']} ({stats['image_count']} photos)",
                        key=f"person_{person_id}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    )
                    
                    if is_clicked:
                        if st.session_state.get('selected_person') == person_id:
                            st.session_state.selected_person = None
                        else:
                            st.session_state.selected_person = person_id
                        st.rerun()
                
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    padding: 0 1rem;
                    margin-bottom: 1rem;
                ">
                    <span>💬 {stats['comment_count']} comments</span>
                    <span>⭐ {stats['avg_rating']:.1f} avg rating</span>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error in table of contents: {str(e)}")
            self.render_directory_info()
        finally:
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_person_gallery(self, person_id: str):
        """Render gallery for a specific person"""
        person_info = None
        for person in self.album_manager.db.get_all_people():
            if person['person_id'] == person_id:
                person_info = person
                break
        
        if not person_info:
            st.error("Person not found")
            return
        
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        
        # Person header with stats
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"""
            <div style="
                width: 100px;
                height: 100px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 40px;
                font-weight: bold;
                margin: 0 auto;
            ">
            {person_info['display_name'][0].upper()}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            stats = self.album_manager.get_person_stats(person_id)
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <h2 style="margin: 0; color: #2d3748;">{person_info['display_name']}</h2>
                <p style="color: #666; margin: 5px 0;">{person_info.get('bio', 'No bio available')}</p>
            </div>
            <div style="display: flex; gap: 20px; margin-top: 1rem;">
                <div>
                    <div style="font-size: 24px; font-weight: bold; color: #667eea;">{stats['image_count']}</div>
                    <div style="font-size: 12px; color: #666;">PHOTOS</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold; color: #48bb78;">{stats['comment_count']}</div>
                    <div style="font-size: 12px; color: #666;">COMMENTS</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold; color: #ed8936;">{stats['avg_rating']:.1f}</div>
                    <div style="font-size: 12px; color: #666;">AVG RATING</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Gallery controls
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            view_mode = st.selectbox("View", ["Grid", "List"], key="view_mode_select")
        with col2:
            sort_by = st.selectbox("Sort by", ["Date", "Rating", "Comments", "Name"], key="sort_by_select")
        with col3:
            items_per_page = st.slider("Items", 12, 48, 24, key="items_per_page_slider")
        with col4:
            show_private = st.checkbox("Show Private", value=False, key="show_private_check")
        
        # Get person's entries
        try:
            with sqlite3.connect(self.album_manager.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = '''
                SELECT ae.*, i.filename, i.filepath, i.thumbnail_path
                FROM album_entries ae
                JOIN images i ON ae.image_id = i.image_id
                WHERE ae.person_id = ?
                '''
                params = [person_id]
                
                if not show_private:
                    query += " AND ae.privacy_level = 'public'"
                
                # Add sorting
                if sort_by == "Date":
                    query += " ORDER BY ae.created_at DESC"
                elif sort_by == "Rating":
                    query += " ORDER BY (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) DESC NULLS LAST"
                elif sort_by == "Comments":
                    query += " ORDER BY (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) DESC"
                elif sort_by == "Name":
                    query += " ORDER BY ae.caption"
                
                cursor.execute(query, params)
                entries = [dict(row) for row in cursor.fetchall()]
            
            if not entries:
                st.info("No photos found for this person.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            # Display images based on view mode
            cols = st.columns(4)
            for idx, entry in enumerate(entries):
                with cols[idx % 4]:
                    # Get thumbnail path or create data URL
                    img_url = ""
                    thumbnail_path = entry.get('thumbnail_path')
                    img_path = Config.DATA_DIR / entry['filepath']
                    
                    if thumbnail_path and Path(thumbnail_path).exists():
                        img_url = self.album_manager.image_processor.get_image_data_url(Path(thumbnail_path))
                    elif img_path.exists():
                        thumbnail_path = self.album_manager.image_processor.create_thumbnail(img_path)
                        if thumbnail_path:
                            img_url = self.album_manager.image_processor.get_image_data_url(thumbnail_path)
                    
                    # Display image card
                    if img_url:
                        st.image(img_url, use_container_width=True, caption=entry.get('caption', 'Untitled'))
                    else:
                        st.image("https://via.placeholder.com/300x300/667eea/ffffff?text=No+Image", 
                                use_container_width=True, 
                                caption=entry.get('caption', 'Untitled'))
            
        except Exception as e:
            st.error(f"Error loading gallery: {str(e)}")
        finally:
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_image_detail(self, entry_id: str):
        """Render detailed view for a single image with proper error handling"""
        try:
            entry_details = self.album_manager.db.get_entry_details(entry_id)
            if not entry_details:
                st.error("Image not found")
                if st.button("↩️ Back to Gallery"):
                    st.session_state.selected_image = None
                    st.rerun()
                return
            
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            
            # Back button
            if st.button("⬅️ Back to Gallery", key="back_button"):
                st.session_state.selected_image = None
                st.rerun()
            
            # Main content
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display image
                img_path = Config.DATA_DIR / entry_details['filepath']
                if img_path.exists():
                    try:
                        with Image.open(img_path) as img:
                            # Resize for display while maintaining aspect ratio
                            img.thumbnail(Config.PREVIEW_SIZE)
                            st.image(img, use_container_width=True, caption=entry_details.get('caption', ''))
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
                        st.image("https://via.placeholder.com/800x600/667eea/ffffff?text=Image+Error", use_container_width=True)
                else:
                    st.error(f"Image file not found at: {img_path}")
                    st.image("https://via.placeholder.com/800x600/667eea/ffffff?text=File+Not+Found", use_container_width=True)
                
                # Image info
                st.markdown(f"""
                <div style="margin-top: 1rem;">
                    <h3>{entry_details.get('caption', 'Untitled')}</h3>
                    <p style="color: #666;">{entry_details.get('description', '')}</p>
                    <div style="display: flex; gap: 20px; color: #666; font-size: 14px; margin: 10px 0;">
                        <div>
                            <strong>Location:</strong> {entry_details.get('location', 'Unknown')}
                        </div>
                        <div>
                            <strong>Date:</strong> {entry_details.get('date_taken', 'Unknown')[:19] if entry_details.get('date_taken') else 'Unknown'}
                        </div>
                        <div>
                            <strong>Size:</strong> {entry_details.get('file_size', 0) / 1024:.1f} KB
                        </div>
                    </div>
                    <div style="margin: 10px 0;">
                        {UIComponents.tag_badges(entry_details.get('tags', []))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Stats and actions
                avg_rating, rating_count = self.album_manager.db.get_entry_ratings(entry_id)
                
                # Get comments
                comments = self.album_manager.db.get_entry_comments(entry_id)
                
                st.markdown("### 📊 Stats")
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                    <div style="margin-bottom: 10px;">
                        <strong>Rating:</strong><br>
                        {UIComponents.rating_stars(avg_rating)}
                        <div style="color: #666; font-size: 14px;">
                            Based on {rating_count} ratings
                        </div>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>Comments:</strong> {len(comments)}
                    </div>
                    <div>
                        <strong>Uploaded:</strong><br>
                        {entry_details.get('created_at', 'Unknown')[:19]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add rating
                st.markdown("### ⭐ Rate this Photo")
                rating_value = st.slider("Your rating", 1, 5, 3, key=f"rate_{entry_id}")
                if st.button("Submit Rating", use_container_width=True, key="submit_rating"):
                    try:
                        self.album_manager.rate_entry(entry_id, rating_value)
                        st.success("✅ Rating submitted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                
                # Add to favorites
                is_favorite = entry_id in st.session_state.get('favorites', set())
                if st.button(f"{'❤️ Remove from Favorites' if is_favorite else '🤍 Add to Favorites'}", 
                           use_container_width=True, key="favorite_button"):
                    if is_favorite:
                        self.album_manager.remove_from_favorites(entry_id)
                        st.success("✅ Removed from favorites!")
                    else:
                        self.album_manager.add_to_favorites(entry_id)
                        st.success("✅ Added to favorites!")
                    st.rerun()
        
        except sqlite3.Error as e:
            st.error(f"Database error: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            st.markdown('</div>', unsafe_allow_html=True)
    
    def main(self):
        """Main application entry point with robust error handling"""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar
            self.render_sidebar()
            
            # Main content area
            tab1, tab2 = st.tabs([
                "🏠 Dashboard",
                "📋 Table of Contents"
            ])
            
            with tab1:
                self.render_dashboard()
            
            with tab2:
                self.render_table_of_contents()
                # If a person is selected, show their gallery
                if st.session_state.get('selected_person'):
                    self.render_person_gallery(st.session_state.selected_person)
            
            # Handle image selection
            if st.session_state.get('selected_image'):
                self.render_image_detail(st.session_state.selected_image)
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.markdown("""
            <div class="error-message">
                <h3>⚠️ Application Error</h3>
                <p>An unexpected error occurred. Please try the following:</p>
                <ul>
                    <li>Refresh the page</li>
                    <li>Clear the cache using the sidebar button</li>
                    <li>Scan for new photos again</li>
                    <li>If the problem persists, restart the application</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Initialize directories first
    Config.init_directories()
    
    # Create and run the application
    app = PhotoAlbumApp()
    app.main()
