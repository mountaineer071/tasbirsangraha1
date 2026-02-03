"""
COMPREHENSIVE WEB PHOTO & VIDEO ALBUM APPLICATION
Version: 3.0.0 - Enhanced with Video Streaming & Password Protection
Features: Table of Contents, Image/Video Gallery, Comments, Ratings, Metadata Management, Search, Password Auth
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
import mimetypes
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# VIDEO PROCESSING IMPORTS
# ============================================================================
try:
    import cv2
    import moviepy.editor as mp
    from moviepy.editor import VideoFileClip
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False
    st.warning("Video processing libraries not installed. Install with: pip install opencv-python moviepy")

# ============================================================================
# PASSWORD AUTHENTICATION
# ============================================================================
def check_password():
    """Check if user has entered correct password based on website name"""
    # Get the website name from Streamlit Cloud or environment variable
    # For local testing, you can set an environment variable: SITE_NAME=your_site_name
    site_name = os.environ.get("SITE_NAME", "")
    
    # If running locally or SITE_NAME not set, try to extract from URL or use default
    if not site_name and st.secrets.get("site_name"):
        site_name = st.secrets["site_name"]
    
    # If still no site name, use a default for local development
    if not site_name:
        site_name = "localdev"
        st.info("Running in local development mode. Password is 'localdev'")
    
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # If already authenticated, return True
    if st.session_state.authenticated:
        return True
    
    # Show password input
    st.title("🔒 Secure Photo & Video Album")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"### {site_name.upper()}")
        st.markdown("Please enter the password to access the album")
        
        password = st.text_input("Password", type="password", key="password_input")
        
        if st.button("Login", use_container_width=True):
            if password.strip() == site_name:
                st.session_state.authenticated = True
                st.success("Access granted!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        
        # Optional: Show hint
        if st.checkbox("Show hint"):
            st.info(f"The password is exactly: **{site_name}**")
    
    return False

# ============================================================================
# CONFIGURATION AND CONSTANTS - ENHANCED FOR VIDEO
# ============================================================================
class Config:
    """Application configuration constants"""
    APP_NAME = "MemoryVault Pro+"
    VERSION = "3.0.0"
    
    # Get absolute paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    VIDEO_THUMBNAIL_DIR = BASE_DIR / "video_thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    VIDEO_CACHE_DIR = BASE_DIR / "video_cache"
    
    # Files and paths
    METADATA_FILE = METADATA_DIR / "album_metadata.json"
    DB_FILE = DB_DIR / "album.db"
    
    # Image settings
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Video settings
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
    VIDEO_THUMBNAIL_SIZE = (300, 300)
    VIDEO_PREVIEW_SIZE = (800, 450)  # 16:9 aspect ratio
    VIDEO_CACHE_SIZE = 50 * 1024 * 1024  # 50MB max cache
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v']
    
    # Gallery settings
    ITEMS_PER_PAGE = 20
    GRID_COLUMNS = 4
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'} | set(SUPPORTED_VIDEO_FORMATS)
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
            cls.VIDEO_THUMBNAIL_DIR,
            cls.METADATA_DIR,
            cls.DB_DIR,
            cls.EXPORT_DIR,
            cls.VIDEO_CACHE_DIR
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
            readme_file.write_text(f"Photos/Videos of {person.replace('-', ' ').title()}\nAdd your media here!")
            
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

class MediaType(Enum):
    """Media type enumeration"""
    IMAGE = "image"
    VIDEO = "video"

# ============================================================================
# DATA MODELS - ENHANCED FOR VIDEO
# ============================================================================
@dataclass
class MediaMetadata:
    """Metadata for a single media file (image or video)"""
    media_id: str
    filename: str
    filepath: str
    file_size: int
    media_type: str
    dimensions: Tuple[int, int]
    format: str
    duration: Optional[float]  # For videos
    frame_rate: Optional[float]  # For videos
    created_date: datetime.datetime
    modified_date: datetime.datetime
    exif_data: Optional[Dict]  # For images
    checksum: str
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'MediaMetadata':
        """Create metadata from media file"""
        if not file_path.exists():
            raise FileNotFoundError(f"Media file not found: {file_path}")
        
        # Determine media type
        media_type = cls._detect_media_type(file_path)
        
        stats = file_path.stat()
        
        if media_type == MediaType.IMAGE.value:
            return cls._from_image(file_path, stats, media_type)
        elif media_type == MediaType.VIDEO.value:
            return cls._from_video(file_path, stats, media_type)
        else:
            raise ValueError(f"Unsupported media type for file: {file_path}")
    
    @staticmethod
    def _detect_media_type(file_path: Path) -> str:
        """Detect if file is image or video"""
        ext = file_path.suffix.lower()
        if ext in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaType.VIDEO.value
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']:
            return MediaType.IMAGE.value
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    @classmethod
    def _from_image(cls, image_path: Path, stats: os.stat_result, media_type: str) -> 'MediaMetadata':
        """Create metadata from image file"""
        with Image.open(image_path) as img:
            return cls(
                media_id=str(uuid.uuid4()),
                filename=image_path.name,
                filepath=str(image_path.relative_to(Config.DATA_DIR)),
                file_size=stats.st_size,
                media_type=media_type,
                dimensions=img.size,
                format=img.format,
                duration=None,
                frame_rate=None,
                created_date=datetime.datetime.fromtimestamp(stats.st_ctime),
                modified_date=datetime.datetime.fromtimestamp(stats.st_mtime),
                exif_data=cls._extract_exif(img),
                checksum=cls._calculate_checksum(image_path)
            )
    
    @classmethod
    def _from_video(cls, video_path: Path, stats: os.stat_result, media_type: str) -> 'MediaMetadata':
        """Create metadata from video file"""
        dimensions = (0, 0)
        duration = 0.0
        frame_rate = 0.0
        
        if VIDEO_SUPPORT:
            try:
                # Try using moviepy first
                clip = VideoFileClip(str(video_path))
                dimensions = clip.size
                duration = clip.duration
                frame_rate = clip.fps
                clip.close()
            except Exception as e:
                # Fallback to OpenCV
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    dimensions = (width, height)
                    duration = frame_count / fps if fps > 0 else 0
                    frame_rate = fps
                    cap.release()
                except Exception as e2:
                    st.warning(f"Could not extract video metadata for {video_path}: {str(e2)}")
        else:
            st.warning(f"Video processing not available for {video_path}")
        
        return cls(
            media_id=str(uuid.uuid4()),
            filename=video_path.name,
            filepath=str(video_path.relative_to(Config.DATA_DIR)),
            file_size=stats.st_size,
            media_type=media_type,
            dimensions=dimensions,
            format=video_path.suffix[1:].upper(),
            duration=duration,
            frame_rate=frame_rate,
            created_date=datetime.datetime.fromtimestamp(stats.st_ctime),
            modified_date=datetime.datetime.fromtimestamp(stats.st_mtime),
            exif_data=None,
            checksum=cls._calculate_checksum(video_path)
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
    def _calculate_checksum(file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found for checksum: {file_path}")
            
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

@dataclass
class AlbumEntry:
    """Album entry with user content"""
    entry_id: str
    media_id: str  # Changed from image_id to media_id
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
# DATABASE MANAGEMENT - ENHANCED FOR VIDEO
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
                
                # Create media table (replaces images table)
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS media (
                    media_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    file_size INTEGER,
                    media_type TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    duration REAL,
                    frame_rate REAL,
                    created_date TIMESTAMP,
                    modified_date TIMESTAMP,
                    exif_data TEXT,
                    checksum TEXT UNIQUE,
                    thumbnail_path TEXT,
                    video_thumbnail_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create people table
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
                
                # Update album_entries to use media_id
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS album_entries (
                    entry_id TEXT PRIMARY KEY,
                    media_id TEXT NOT NULL,
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
                    FOREIGN KEY (media_id) REFERENCES media (media_id),
                    FOREIGN KEY (person_id) REFERENCES people (person_id)
                )
                ''')
                
                # Create comments table
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
                
                # Create ratings table
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
                
                # Create user_favorites table
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
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_media ON album_entries(media_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_person ON album_entries(person_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_created ON album_entries(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_entry ON comments(entry_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_entry ON ratings(entry_id)')
                
                # Migrate existing data if needed
                self._migrate_existing_data(conn)
                
                conn.commit()
                st.success("✅ Database initialized successfully!")
                
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {str(e)}")
            raise
    
    def _migrate_existing_data(self, conn):
        """Migrate data from old images table to new media table"""
        cursor = conn.cursor()
        
        # Check if old images table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
        if cursor.fetchone():
            # Check if media table is empty
            cursor.execute("SELECT COUNT(*) FROM media")
            media_count = cursor.fetchone()[0]
            
            if media_count == 0:
                # Migrate data from images to media
                cursor.execute('''
                INSERT INTO media (media_id, filename, filepath, file_size, media_type, 
                                  width, height, format, duration, frame_rate,
                                  created_date, modified_date, exif_data, checksum, thumbnail_path)
                SELECT image_id, filename, filepath, file_size, 'image',
                       width, height, format, NULL, NULL,
                       created_date, modified_date, exif_data, checksum, thumbnail_path
                FROM images
                ''')
                
                # Update album_entries to use media_id
                cursor.execute('''
                UPDATE album_entries
                SET media_id = (
                    SELECT media_id FROM media 
                    WHERE media.checksum = (
                        SELECT checksum FROM images 
                        WHERE images.image_id = album_entries.image_id
                    )
                )
                WHERE image_id IS NOT NULL
                ''')
                
                # Drop old images table
                cursor.execute("DROP TABLE images")
                st.info("✅ Migrated existing image data to new media format")
    
    def add_media(self, metadata: MediaMetadata, thumbnail_path: str = None, video_thumbnail_path: str = None):
        """Add media metadata to database"""
        if not isinstance(metadata, MediaMetadata):
            raise TypeError("metadata must be a MediaMetadata instance")
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO media
                (media_id, filename, filepath, file_size, media_type, width, height, format,
                duration, frame_rate, created_date, modified_date, exif_data, checksum,
                thumbnail_path, video_thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.media_id,
                    metadata.filename,
                    metadata.filepath,
                    metadata.file_size,
                    metadata.media_type,
                    metadata.dimensions[0],
                    metadata.dimensions[1],
                    metadata.format,
                    metadata.duration,
                    metadata.frame_rate,
                    metadata.created_date,
                    metadata.modified_date,
                    json.dumps(metadata.exif_data) if metadata.exif_data else None,
                    metadata.checksum,
                    thumbnail_path,
                    video_thumbnail_path
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding media to database: {str(e)}")
            raise
    
    def get_media(self, media_id: str) -> Optional[Dict]:
        """Retrieve media metadata"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM media WHERE media_id = ?', (media_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            st.error(f"Error retrieving media: {str(e)}")
            return None
    
    def get_media_by_type(self, media_type: str, limit: int = 100) -> List[Dict]:
        """Get media by type"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM media 
                WHERE media_type = ? 
                ORDER BY created_date DESC 
                LIMIT ?
                ''', (media_type, limit))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving {media_type}s: {str(e)}")
            return []
    
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
                (entry_id, media_id, person_id, caption, description, location,
                date_taken, tags, privacy_level, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.entry_id,
                    entry.media_id,
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
    
    def search_entries(self, query: str, person_id: str = None, media_type: str = None) -> List[Dict]:
        """Search album entries"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                search_pattern = f'%{query}%'
                
                conditions = []
                params = []
                
                if person_id:
                    conditions.append("ae.person_id = ?")
                    params.append(person_id)
                
                if media_type:
                    conditions.append("m.media_type = ?")
                    params.append(media_type)
                
                conditions.append("(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)")
                params.extend([search_pattern, search_pattern, search_pattern])
                
                where_clause = " AND ".join(conditions)
                
                cursor.execute(f'''
                SELECT ae.*, p.display_name, m.filename, m.media_type, m.thumbnail_path, m.video_thumbnail_path
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
                WHERE {where_clause}
                ORDER BY ae.created_at DESC
                ''', params)
                
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
                    ae.media_id,
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
                    m.filename,
                    m.filepath,
                    m.media_type,
                    m.file_size,
                    m.format,
                    m.duration,
                    m.frame_rate,
                    m.width,
                    m.height,
                    m.created_date,
                    m.exif_data,
                    m.thumbnail_path,
                    m.video_thumbnail_path,
                    (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count,
                    (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                    (SELECT COUNT(*) FROM ratings r WHERE r.entry_id = ae.entry_id) as rating_count
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
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
    
    def get_person_by_folder(self, folder_name: str) -> Optional[Dict]:
        """Get person by folder name"""
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM people WHERE folder_name = ?', (folder_name,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            st.error(f"Error retrieving person by folder: {str(e)}")
            return None

# ============================================================================
# MEDIA PROCESSING AND THUMBNAIL MANAGEMENT
# ============================================================================
class MediaProcessor:
    """Handle media processing operations for both images and videos"""
    
    @staticmethod
    def create_thumbnail(media_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        """Create thumbnail for media file (image or video)"""
        if not media_path.exists():
            st.warning(f"Media file not found: {media_path}")
            return None
        
        # Detect media type
        ext = media_path.suffix.lower()
        is_video = ext in Config.SUPPORTED_VIDEO_FORMATS
        
        if is_video:
            return MediaProcessor._create_video_thumbnail(media_path, thumbnail_dir)
        else:
            return MediaProcessor._create_image_thumbnail(media_path, thumbnail_dir)
    
    @staticmethod
    def _create_image_thumbnail(image_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        """Create thumbnail for image"""
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
    def _create_video_thumbnail(video_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        """Create thumbnail for video"""
        if not VIDEO_SUPPORT:
            st.warning(f"Video processing not available. Cannot create thumbnail for {video_path}")
            return None
        
        thumbnail_dir = thumbnail_dir or Config.VIDEO_THUMBNAIL_DIR
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        thumbnail_path = thumbnail_dir / f"{video_path.stem}_thumb.jpg"
        
        try:
            # Try using OpenCV first
            cap = cv2.VideoCapture(str(video_path))
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Read frame from middle of video
            if total_frames > 0:
                middle_frame = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                img = Image.fromarray(frame_rgb)
                
                # Resize
                img.thumbnail(Config.VIDEO_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                return thumbnail_path
            else:
                st.warning(f"Could not read frame from video: {video_path}")
                return None
                
        except Exception as e:
            st.error(f"Error creating video thumbnail for {video_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_media_data_url(media_path: Path) -> str:
        """Convert media to data URL for embedding"""
        try:
            if not media_path.exists():
                return ""
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(str(media_path))
            if not mime_type:
                # Fallback MIME types
                ext = media_path.suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif ext == '.png':
                    mime_type = 'image/png'
                elif ext == '.gif':
                    mime_type = 'image/gif'
                elif ext in ['.mp4', '.webm']:
                    mime_type = f'video/{ext[1:]}'
                else:
                    mime_type = 'application/octet-stream'
            
            with open(media_path, "rb") as media_file:
                encoded = base64.b64encode(media_file.read()).decode('utf-8')
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            st.error(f"Error encoding media {media_path}: {str(e)}")
            return ""
    
    @staticmethod
    def prepare_video_stream(video_path: Path, max_size_mb: int = 50) -> Optional[bytes]:
        """Prepare video for streaming with size limit"""
        try:
            if not video_path.exists():
                return None
            
            file_size = video_path.stat().st_size
            
            # Check if file is too large
            if file_size > max_size_mb * 1024 * 1024:
                st.warning(f"Video is too large ({file_size/(1024*1024):.1f}MB). Maximum size is {max_size_mb}MB.")
                return None
            
            with open(video_path, "rb") as video_file:
                return video_file.read()
        except Exception as e:
            st.error(f"Error reading video file {video_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_video_info(video_path: Path) -> Dict[str, Any]:
        """Get video information"""
        info = {
            'duration': 0,
            'dimensions': (0, 0),
            'frame_rate': 0,
            'file_size': video_path.stat().st_size if video_path.exists() else 0
        }
        
        if not VIDEO_SUPPORT or not video_path.exists():
            return info
        
        try:
            # Try using moviepy
            clip = VideoFileClip(str(video_path))
            info['duration'] = clip.duration
            info['dimensions'] = clip.size
            info['frame_rate'] = clip.fps
            clip.close()
        except Exception as e:
            # Fallback to OpenCV
            try:
                cap = cv2.VideoCapture(str(video_path))
                info['dimensions'] = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                info['frame_rate'] = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                info['duration'] = frame_count / info['frame_rate'] if info['frame_rate'] > 0 else 0
                cap.release()
            except Exception as e2:
                st.warning(f"Could not get video info for {video_path}: {str(e2)}")
        
        return info

# ============================================================================
# CACHE MANAGEMENT - ENHANCED FOR VIDEO
# ============================================================================
class CacheManager:
    """Manage application caching with video support"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self._video_cache = {}  # Separate cache for video data
    
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
            self._video_cache.clear()
    
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
    
    # Video-specific cache methods
    def get_video(self, video_path: Path) -> Optional[bytes]:
        """Get cached video data"""
        cache_key = f"video_{video_path}"
        if cache_key in self._video_cache:
            return self._video_cache[cache_key]
        return None
    
    def set_video(self, video_path: Path, video_data: bytes):
        """Cache video data"""
        cache_key = f"video_{video_path}"
        self._video_cache[cache_key] = video_data
        
        # Limit cache size
        if len(self._video_cache) > 10:  # Keep only 10 videos in cache
            # Remove oldest (simple FIFO)
            keys = list(self._video_cache.keys())
            if keys:
                del self._video_cache[keys[0]]
    
    def clear_video_cache(self):
        """Clear video cache"""
        self._video_cache.clear()

# ============================================================================
# UI COMPONENTS - ENHANCED FOR VIDEO
# ============================================================================
class UIComponents:
    """Reusable UI components with video support"""
    
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
    def media_type_badge(media_type: str) -> str:
        """Generate HTML badge for media type"""
        if media_type == MediaType.VIDEO.value:
            return '''
            <span style="
                background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 10px;
                font-weight: bold;
                margin-left: 8px;
                display: inline-block;
            ">
            🎬 VIDEO
            </span>
            '''
        else:
            return '''
            <span style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 10px;
                font-weight: bold;
                margin-left: 8px;
                display: inline-block;
            ">
            📸 IMAGE
            </span>
            '''
    
    @staticmethod
    def video_duration_badge(duration: float) -> str:
        """Generate HTML badge for video duration"""
        if not duration or duration <= 0:
            return ""
        
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        return f'''
        <span style="
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: bold;
            position: absolute;
            bottom: 8px;
            right: 8px;
            z-index: 10;
        ">
        {minutes:02d}:{seconds:02d}
        </span>
        '''

# ============================================================================
# ALBUM MANAGER - ENHANCED FOR VIDEO
# ============================================================================
class AlbumManager:
    """Main album management class with video support"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.media_processor = MediaProcessor()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'initialized': True,
                'current_page': 1,
                'selected_person': None,
                'selected_media': None,
                'search_query': '',
                'view_mode': 'grid',
                'sort_by': 'date',
                'sort_order': 'desc',
                'media_filter': 'all',  # 'all', 'image', 'video'
                'selected_tags': [],
                'user_id': str(uuid.uuid4()),
                'username': 'Guest',
                'user_role': UserRoles.VIEWER.value,
                'favorites': set(),
                'recently_viewed': [],
                'toc_page': 1,
                'gallery_page': 1,
                'show_directory_info': True,
                'video_autoplay': False
            })
    
    def scan_directory(self, data_dir: Path = None) -> Dict:
        """Scan directory for media files and update database"""
        data_dir = data_dir or Config.DATA_DIR
        results = {
            'total_media': 0,
            'new_media': 0,
            'updated_media': 0,
            'images_found': 0,
            'videos_found': 0,
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
                          and d.name not in ['thumbnails', 'video_thumbnails', 'video_cache']]
            
            if not person_dirs:
                st.warning("No person folders found. Creating sample structure...")
                Config.create_sample_structure()
                person_dirs = [d for d in data_dir.iterdir() 
                              if d.is_dir() and not d.name.startswith('.') 
                              and d.name not in ['thumbnails', 'video_thumbnails', 'video_cache']]
            
            results['people_found'] = len(person_dirs)
            
            progress_bar = st.progress(0)
            total_files = 0
            processed_files = 0
            
            # Count total files
            for person_dir in person_dirs:
                media_files = [
                    f for f in person_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS
                ]
                total_files += len(media_files)
            
            if total_files == 0:
                st.info("No media files found in any folders. Please add some images/videos to your person folders.")
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
                existing_person = self.db.get_person_by_folder(person_dir.name)
                
                if not existing_person:
                    person_profile = PersonProfile(
                        person_id=str(uuid.uuid4()),
                        folder_name=person_dir.name,
                        display_name=display_name,
                        bio=f"Photos/Videos of {display_name}",
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
                
                # Process media files
                media_files = [
                    f for f in person_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS
                ]
                
                for media_path in media_files:
                    try:
                        processed_files += 1
                        progress = processed_files / max(total_files, 1)
                        progress_bar.progress(progress)
                        
                        # Calculate checksum
                        checksum = MediaMetadata._calculate_checksum(media_path)
                        
                        # Check if media exists
                        with sqlite3.connect(self.db.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                'SELECT media_id FROM media WHERE checksum = ?',
                                (checksum,)
                            )
                            existing = cursor.fetchone()
                        
                        if existing:
                            results['updated_media'] += 1
                            continue
                        
                        # Extract metadata
                        try:
                            metadata = MediaMetadata.from_file(media_path)
                        except Exception as e:
                            results['errors'].append(f"Error processing {media_path.name}: {str(e)}")
                            continue
                        
                        # Create thumbnail based on media type
                        thumbnail_path = None
                        video_thumbnail_path = None
                        
                        if metadata.media_type == MediaType.IMAGE.value:
                            thumbnail_path = self.media_processor.create_thumbnail(media_path)
                            thumbnail_path_str = str(thumbnail_path) if thumbnail_path else None
                            results['images_found'] += 1
                        elif metadata.media_type == MediaType.VIDEO.value:
                            video_thumbnail_path = self.media_processor.create_thumbnail(media_path)
                            video_thumbnail_path_str = str(video_thumbnail_path) if video_thumbnail_path else None
                            results['videos_found'] += 1
                        
                        # Add to database
                        self.db.add_media(metadata, 
                                         str(thumbnail_path) if thumbnail_path else None,
                                         str(video_thumbnail_path) if video_thumbnail_path else None)
                        
                        # Create album entry
                        album_entry = AlbumEntry(
                            entry_id=str(uuid.uuid4()),
                            media_id=metadata.media_id,
                            person_id=person_id,
                            caption=media_path.stem.replace('_', ' ').title(),
                            description=f"Media of {display_name}",
                            location="",
                            date_taken=metadata.created_date,
                            tags=[display_name.lower().replace(' ', '-'), 
                                  metadata.media_type, 
                                  'memory'],
                            privacy_level='public',
                            created_by='system',
                            created_at=datetime.datetime.now(),
                            updated_at=datetime.datetime.now()
                        )
                        self.db.add_album_entry(album_entry)
                        
                        results['new_media'] += 1
                        results['total_media'] += 1
                        
                    except Exception as e:
                        error_msg = f"Error processing {media_path}: {str(e)}"
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
                    media_count = cursor.fetchone()[0]
                    
                    # Count images and videos separately
                    cursor.execute('''
                    SELECT m.media_type, COUNT(*) as count
                    FROM album_entries ae
                    JOIN media m ON ae.media_id = m.media_id
                    WHERE ae.person_id = ?
                    GROUP BY m.media_type
                    ''', (person_id,))
                    
                    image_count = 0
                    video_count = 0
                    for row in cursor.fetchall():
                        if row[0] == 'image':
                            image_count = row[1]
                        elif row[0] == 'video':
                            video_count = row[1]
                    
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
                        'media_count': media_count,
                        'image_count': image_count,
                        'video_count': video_count,
                        'comment_count': comment_count,
                        'avg_rating': float(avg_rating),
                        'last_activity': last_activity
                    }
            except sqlite3.Error as e:
                st.error(f"Error getting person stats: {str(e)}")
                return {
                    'media_count': 0,
                    'image_count': 0,
                    'video_count': 0,
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
                SELECT ae.*, p.display_name, m.filename, m.media_type, m.thumbnail_path, m.video_thumbnail_path
                FROM user_favorites uf
                JOIN album_entries ae ON uf.entry_id = ae.entry_id
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
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
                            person_with_stats['profile_image_data'] = self.media_processor.get_media_data_url(profile_image_path)
                    
                    result.append(person_with_stats)
                
                return result
            except Exception as e:
                st.error(f"Error getting people with stats: {str(e)}")
                return []
        
        return self.cache.get_or_set(cache_key, generate_data)
    
    def get_entries_by_person(self, person_id: str, page: int = 1, search_query: str = None, media_filter: str = 'all') -> Dict:
        """Get entries for a specific person with pagination"""
        cache_key = f"entries_person_{person_id}_page_{page}_query_{search_query}_filter_{media_filter}"
        
        def generate_data():
            try:
                offset = (page - 1) * Config.ITEMS_PER_PAGE
                
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    conditions = ["ae.person_id = ?"]
                    params = [person_id]
                    
                    if search_query:
                        search_pattern = f'%{search_query}%'
                        conditions.append("(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)")
                        params.extend([search_pattern, search_pattern, search_pattern])
                    
                    if media_filter != 'all':
                        conditions.append("m.media_type = ?")
                        params.append(media_filter)
                    
                    where_clause = " AND ".join(conditions)
                    params.extend([Config.ITEMS_PER_PAGE, offset])
                    
                    query = f'''
                    SELECT ae.*, p.display_name, m.filename, m.media_type, m.thumbnail_path, m.video_thumbnail_path,
                           (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                           (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count,
                           m.duration
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN media m ON ae.media_id = m.media_id
                    WHERE {where_clause}
                    ORDER BY ae.created_at DESC
                    LIMIT ? OFFSET ?
                    '''
                    
                    cursor.execute(query, params)
                    entries = [dict(row) for row in cursor.fetchall()]
                    
                    # Get total count
                    count_params = [person_id]
                    if search_query:
                        search_pattern = f'%{search_query}%'
                        conditions_without_limit = ["ae.person_id = ?", "(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)"]
                        count_params.extend([search_pattern, search_pattern, search_pattern])
                        if media_filter != 'all':
                            conditions_without_limit.append("m.media_type = ?")
                            count_params.append(media_filter)
                        count_where = " AND ".join(conditions_without_limit)
                    else:
                        if media_filter != 'all':
                            conditions_without_limit = ["ae.person_id = ?", "m.media_type = ?"]
                            count_params.append(media_filter)
                            count_where = " AND ".join(conditions_without_limit)
                        else:
                            count_where = "ae.person_id = ?"
                    
                    count_query = f'''
                    SELECT COUNT(*)
                    FROM album_entries ae
                    JOIN media m ON ae.media_id = m.media_id
                    WHERE {count_where}
                    '''
                    
                    cursor.execute(count_query, count_params)
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
                    SELECT ae.*, p.display_name, m.filename, m.media_type, m.thumbnail_path, m.video_thumbnail_path,
                           (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN media m ON ae.media_id = m.media_id
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
                    SELECT ae.*, p.display_name, m.filename, m.media_type, m.thumbnail_path, m.video_thumbnail_path,
                           (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                           (SELECT COUNT(*) FROM ratings r WHERE r.entry_id = ae.entry_id) as rating_count
                    FROM album_entries ae
                    JOIN people p ON ae.person_id = p.person_id
                    JOIN media m ON ae.media_id = m.media_id
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
                
                # Get media path
                if entry.get('filepath'):
                    media_path = Config.DATA_DIR / entry['filepath']
                    if media_path.exists():
                        if entry['media_type'] == MediaType.IMAGE.value:
                            entry['media_data_url'] = self.media_processor.get_media_data_url(media_path)
                        elif entry['media_type'] == MediaType.VIDEO.value:
                            # For videos, we'll stream them differently
                            entry['media_path'] = str(media_path)
                
                # Get thumbnail
                if entry.get('thumbnail_path'):
                    thumbnail_path = Path(entry['thumbnail_path'])
                    if thumbnail_path.exists():
                        entry['thumbnail_data_url'] = self.media_processor.get_media_data_url(thumbnail_path)
                elif entry.get('video_thumbnail_path'):
                    video_thumbnail_path = Path(entry['video_thumbnail_path'])
                    if video_thumbnail_path.exists():
                        entry['thumbnail_data_url'] = self.media_processor.get_media_data_url(video_thumbnail_path)
                
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
    
    def stream_video(self, video_path: str):
        """Stream video file"""
        try:
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                st.error("Video file not found")
                return
            
            # Check cache first
            video_data = self.cache.get_video(video_path_obj)
            
            if not video_data:
                # Prepare video for streaming
                video_data = self.media_processor.prepare_video_stream(video_path_obj)
                if video_data:
                    self.cache.set_video(video_path_obj, video_data)
            
            if video_data:
                # Determine MIME type
                mime_type, _ = mimetypes.guess_type(video_path)
                if not mime_type:
                    mime_type = "video/mp4"
                
                # Create video player
                st.video(video_data, format=mime_type, start_time=0)
            else:
                st.error("Could not load video")
                
        except Exception as e:
            st.error(f"Error streaming video: {str(e)}")

# ============================================================================
# UI APPLICATION CLASS - ENHANCED FOR VIDEO
# ============================================================================
class PhotoVideoAlbumApp:
    """Main application class with UI components for both photos and videos"""
    
    def __init__(self):
        self.manager = AlbumManager()
        self.setup_page_config()
        self.check_initialization()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=Config.APP_NAME,
            page_icon="🎬📸",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/yourusername/photo-video-album',
                'Report a bug': 'https://github.com/yourusername/photo-video-album/issues',
                'About': f"# {Config.APP_NAME} v{Config.VERSION}\nA comprehensive photo & video album management system"
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
            st.title(f"🎬📸 {Config.APP_NAME}")
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
                "📁 Media Gallery": "gallery",
                "⭐ Favorites": "favorites",
                "🎬 Video Library": "videos",
                "📸 Photo Library": "photos",
                "🔍 Search": "search",
                "📊 Statistics": "statistics",
                "⚙️ Settings": "settings",
                "📤 Import/Export": "import_export"
            }
            
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
                    if results['new_media'] > 0 or results['updated_media'] > 0:
                        st.success(f"Found {results['new_media']} new media files ({results['images_found']} images, {results['videos_found']} videos)")
                    st.rerun()
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                self.manager.cache.clear()
                st.success("Cache cleared!")
                st.rerun()
            
            # Display info
            st.divider()
            st.subheader("Media Info")
            
            people = self.manager.db.get_all_people()
            total_images = 0
            total_videos = 0
            
            for person in people:
                stats = self.manager.get_person_stats(person['person_id'])
                total_images += stats.get('image_count', 0)
                total_videos += stats.get('video_count', 0)
            
            st.metric("People", len(people))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Images", total_images)
            with col2:
                st.metric("Videos", total_videos)
            
            # Video autoplay toggle
            st.divider()
            autoplay = st.toggle("Video Autoplay", value=st.session_state.get('video_autoplay', False))
            if autoplay != st.session_state.get('video_autoplay', False):
                st.session_state['video_autoplay'] = autoplay
                st.rerun()
    
    def render_dashboard(self):
        """Render dashboard page"""
        st.title("📊 Dashboard")
        st.markdown("Welcome to your photo & video album management system!")
        
        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            people = self.manager.db.get_all_people()
            st.metric("👥 People", len(people))
        
        with col2:
            total_media = 0
            total_images = 0
            total_videos = 0
            for person in people:
                stats = self.manager.get_person_stats(person['person_id'])
                total_media += stats.get('media_count', 0)
                total_images += stats.get('image_count', 0)
                total_videos += stats.get('video_count', 0)
            st.metric("📸🎬 Total Media", total_media)
        
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
        
        # Media type breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Media Distribution")
            if total_media > 0:
                labels = ['Images', 'Videos']
                sizes = [total_images, total_videos]
                colors = ['#667eea', '#FF416C']
                
                fig_data = pd.DataFrame({
                    'Type': labels,
                    'Count': sizes
                })
                
                st.bar_chart(fig_data.set_index('Type'))
            else:
                st.info("No media files found")
        
        with col2:
            st.subheader("⚡ Quick Stats")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Images", total_images)
                st.metric("Videos", total_videos)
            with col_b:
                st.metric("People", len(people))
                if total_media > 0:
                    st.metric("Avg per Person", f"{total_media/len(people):.1f}")
        
        # Recent Activity
        st.divider()
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
                            if entry.get('media_type') == MediaType.VIDEO.value:
                                st.markdown("🎬")
                            else:
                                st.markdown("📸")
                    with col_b:
                        st.markdown(f"**{entry.get('caption', 'Untitled')}**")
                        st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')), unsafe_allow_html=True)
                        st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
                        st.caption(f"📅 {entry.get('created_at', '')}")
                    st.divider()
        else:
            st.info("No recent activity")
        
        # Quick Actions
        st.divider()
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📁 Scan for New Media", use_container_width=True):
                with st.spinner("Scanning..."):
                    results = self.manager.scan_directory()
                    st.success(f"Found {results['new_media']} new media files!")
                    st.rerun()
        
        with col2:
            if st.button("👥 Manage People", use_container_width=True):
                st.session_state['current_page'] = 'people'
                st.rerun()
        
        with col3:
            if st.button("🎬 View Videos", use_container_width=True):
                st.session_state['current_page'] = 'videos'
                st.rerun()
    
    def render_people_page(self):
        """Render people management page"""
        st.title("👥 People")
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search people...", key="people_search")
        with col2:
            sort_by = st.selectbox("Sort by", ["Name", "Recently Added", "Media Count"], key="people_sort")
        
        # Get all people
        all_people = self.manager.get_all_people_with_stats()
        
        # Filter and sort
        if search_query:
            all_people = [p for p in all_people if search_query.lower() in p['display_name'].lower()]
        
        if sort_by == "Name":
            all_people.sort(key=lambda x: x['display_name'])
        elif sort_by == "Media Count":
            all_people.sort(key=lambda x: x['media_count'], reverse=True)
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
                        st.metric("Media", person.get('media_count', 0))
                    with col_b:
                        st.metric("Images", person.get('image_count', 0))
                    with col_c:
                        st.metric("Videos", person.get('video_count', 0))
                    
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
        """Render gallery page for all media"""
        st.title("📁 Media Gallery")
        
        # Check if we're viewing a specific person
        selected_person_id = st.session_state.get('selected_person')
        if selected_person_id:
            people = self.manager.db.get_all_people()
            selected_person = next((p for p in people if p['person_id'] == selected_person_id), None)
            if selected_person:
                st.subheader(f"Media of {selected_person['display_name']}")
                if st.button("← Back to All People"):
                    st.session_state['selected_person'] = None
                    st.rerun()
        
        # Search and filter
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                search_query = st.text_input("Search media...", key="gallery_search")
            with col2:
                media_filter = st.selectbox(
                    "Media Type",
                    ["All", "Image", "Video"],
                    key="media_filter",
                    index=["All", "Image", "Video"].index(st.session_state.get('media_filter', 'All'))
                )
                st.session_state['media_filter'] = media_filter.lower()
            with col3:
                view_mode = st.selectbox("View", ["Grid", "List"], key="view_mode")
            with col4:
                items_per_page = st.selectbox("Per Page", [12, 24, 48], key="items_per_page")
        
        # Get entries
        page = st.session_state.get('gallery_page', 1)
        if selected_person_id:
            data = self.manager.get_entries_by_person(
                selected_person_id, 
                page, 
                search_query,
                st.session_state['media_filter']
            )
        else:
            # Get all entries with pagination
            offset = (page - 1) * items_per_page
            
            with sqlite3.connect(self.manager.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                conditions = []
                params = []
                
                if search_query:
                    search_pattern = f'%{search_query}%'
                    conditions.append("(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)")
                    params.extend([search_pattern, search_pattern, search_pattern])
                
                if st.session_state['media_filter'] != 'all':
                    conditions.append("m.media_type = ?")
                    params.append(st.session_state['media_filter'])
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                params.extend([items_per_page, offset])
                
                cursor.execute(f'''
                SELECT ae.*, p.display_name, m.filename, m.media_type, m.thumbnail_path, m.video_thumbnail_path, m.duration,
                       (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
                WHERE {where_clause}
                ORDER BY ae.created_at DESC
                LIMIT ? OFFSET ?
                ''', params)
                
                entries = [dict(row) for row in cursor.fetchall()]
                
                # Get total count
                count_params = []
                if search_query:
                    search_pattern = f'%{search_query}%'
                    count_conditions = ["(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)"]
                    count_params.extend([search_pattern, search_pattern, search_pattern])
                    if st.session_state['media_filter'] != 'all':
                        count_conditions.append("m.media_type = ?")
                        count_params.append(st.session_state['media_filter'])
                    count_where = " AND ".join(count_conditions)
                else:
                    if st.session_state['media_filter'] != 'all':
                        count_where = "m.media_type = ?"
                        count_params.append(st.session_state['media_filter'])
                    else:
                        count_where = "1=1"
                
                cursor.execute(f'''
                SELECT COUNT(*)
                FROM album_entries ae
                JOIN media m ON ae.media_id = m.media_id
                WHERE {count_where}
                ''', count_params)
                
                total_count = cursor.fetchone()[0]
                total_pages = max(1, math.ceil(total_count / items_per_page))
                
                data = {
                    'entries': entries,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'current_page': page
                }
        
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
            # Media type indicator
            st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')), unsafe_allow_html=True)
            
            # Thumbnail with play icon for videos
            if entry.get('thumbnail_data_url'):
                col1, col2, col3 = st.columns([1, 8, 1])
                with col2:
                    # Create container for thumbnail
                    st.markdown(
                        f'<div style="position: relative; width: 100%;">'
                        f'<img src="{entry["thumbnail_data_url"]}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px;">',
                        unsafe_allow_html=True
                    )
                    
                    # Add play icon overlay for videos
                    if entry.get('media_type') == MediaType.VIDEO.value:
                        st.markdown(
                            f'<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 48px; color: white; opacity: 0.8;">▶</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Add duration badge
                        if entry.get('duration'):
                            st.markdown(UIComponents.video_duration_badge(entry['duration']), unsafe_allow_html=True)
            else:
                if entry.get('media_type') == MediaType.VIDEO.value:
                    st.markdown("🎬 No thumbnail")
                else:
                    st.markdown("📸 No thumbnail")
            
            # Caption
            st.markdown(f"**{entry.get('caption', 'Untitled')}**")
            st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
            
            # Rating
            if entry.get('avg_rating'):
                st.markdown(UIComponents.rating_stars(entry['avg_rating'], size=15), unsafe_allow_html=True)
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👁️", key=f"view_{entry['entry_id']}", use_container_width=True):
                    st.session_state['selected_media'] = entry['entry_id']
                    st.session_state['current_page'] = 'media_detail'
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
                    # Container for thumbnail with play icon for videos
                    if entry.get('media_type') == MediaType.VIDEO.value:
                        st.markdown(
                            f'<div style="position: relative; width: 100px; height: 100px;">'
                            f'<img src="{entry["thumbnail_data_url"]}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px;">'
                            f'<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 24px; color: white; opacity: 0.8;">▶</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.image(entry['thumbnail_data_url'], width=100)
                else:
                    if entry.get('media_type') == MediaType.VIDEO.value:
                        st.markdown("🎬")
                    else:
                        st.markdown("📸")
            
            with col2:
                st.markdown(f"### {entry.get('caption', 'Untitled')}")
                st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')), unsafe_allow_html=True)
                st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
                
                if entry.get('description'):
                    st.markdown(f"*{entry.get('description', '')[:100]}...*")
                
                if entry.get('tags'):
                    st.markdown(UIComponents.tag_badges(
                        entry['tags'].split(',') if isinstance(entry['tags'], str) else entry['tags'], 
                        max_display=3
                    ), unsafe_allow_html=True)
            
            with col3:
                # Rating
                if entry.get('avg_rating'):
                    st.markdown(UIComponents.rating_stars(entry['avg_rating']), unsafe_allow_html=True)
                
                # Video duration
                if entry.get('media_type') == MediaType.VIDEO.value and entry.get('duration'):
                    minutes = int(entry['duration'] // 60)
                    seconds = int(entry['duration'] % 60)
                    st.caption(f"⏱️ {minutes:02d}:{seconds:02d}")
                
                # Actions
                if st.button("View Details", key=f"view_det_{entry['entry_id']}", use_container_width=True):
                    st.session_state['selected_media'] = entry['entry_id']
                    st.session_state['current_page'] = 'media_detail'
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
    
    def render_media_detail_page(self):
        """Render media detail page for both images and videos"""
        entry_id = st.session_state.get('selected_media')
        if not entry_id:
            st.error("No media selected")
            if st.button("Back to Gallery"):
                st.session_state['current_page'] = 'gallery'
                st.rerun()
            return
        
        entry = self.manager.get_entry_with_details(entry_id)
        if not entry:
            st.error("Media not found")
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
            # Display media
            if entry.get('media_type') == MediaType.IMAGE.value:
                # Display image
                if entry.get('media_data_url'):
                    st.image(entry['media_data_url'], use_column_width=True)
                else:
                    # Try to load from filepath
                    if entry.get('filepath'):
                        media_path = Config.DATA_DIR / entry['filepath']
                        if media_path.exists():
                            data_url = self.manager.media_processor.get_media_data_url(media_path)
                            st.image(data_url, use_column_width=True)
                        else:
                            st.error("Image file not found")
                    else:
                        st.error("No image available")
            
            elif entry.get('media_type') == MediaType.VIDEO.value:
                # Display video
                st.subheader("🎬 Video Player")
                
                if entry.get('media_path'):
                    # Stream video
                    self.manager.stream_video(entry['media_path'])
                    
                    # Video info
                    video_info = self.manager.media_processor.get_video_info(Path(entry['media_path']))
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        minutes = int(video_info['duration'] // 60)
                        seconds = int(video_info['duration'] % 60)
                        st.metric("Duration", f"{minutes}:{seconds:02d}")
                    with col_info2:
                        st.metric("Resolution", f"{video_info['dimensions'][0]}x{video_info['dimensions'][1]}")
                    with col_info3:
                        st.metric("Frame Rate", f"{video_info['frame_rate']:.1f} fps")
                else:
                    st.error("Video file not found")
        
        with col2:
            # Media info
            st.title(entry.get('caption', 'Untitled'))
            
            # Media type badge
            st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')), unsafe_allow_html=True)
            
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
                
                if entry.get('width') and entry.get('height'):
                    st.markdown(f"**Dimensions:** {entry['width']}x{entry['height']}")
                
                # Video-specific metadata
                if entry.get('media_type') == MediaType.VIDEO.value:
                    if entry.get('duration'):
                        minutes = int(entry['duration'] // 60)
                        seconds = int(entry['duration'] % 60)
                        st.markdown(f"**Duration:** {minutes:02d}:{seconds:02d}")
                    
                    if entry.get('frame_rate'):
                        st.markdown(f"**Frame Rate:** {entry['frame_rate']:.1f} fps")
        
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
    
    def render_videos_page(self):
        """Render video library page"""
        st.title("🎬 Video Library")
        
        # Get all videos
        videos = self.manager.db.get_media_by_type(MediaType.VIDEO.value)
        
        if not videos:
            st.info("No videos found. Add some videos to your person folders and scan the directory.")
            return
        
        # Video stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Videos", len(videos))
        with col2:
            total_duration = sum(v.get('duration', 0) for v in videos)
            hours = int(total_duration // 3600)
            minutes = int((total_duration % 3600) // 60)
            st.metric("Total Duration", f"{hours}h {minutes}m")
        with col3:
            avg_duration = total_duration / len(videos) if videos else 0
            minutes = int(avg_duration // 60)
            seconds = int(avg_duration % 60)
            st.metric("Avg Duration", f"{minutes}:{seconds:02d}")
        with col4:
            total_size = sum(v.get('file_size', 0) for v in videos) / (1024*1024*1024)
            st.metric("Total Size", f"{total_size:.2f} GB")
        
        st.divider()
        
        # Video grid
        st.subheader("All Videos")
        
        cols = st.columns(4)
        for idx, video in enumerate(videos):
            with cols[idx % 4]:
                with st.container(border=True):
                    # Thumbnail
                    if video.get('video_thumbnail_path'):
                        thumbnail_path = Path(video['video_thumbnail_path'])
                        if thumbnail_path.exists():
                            data_url = self.manager.media_processor.get_media_data_url(thumbnail_path)
                            st.markdown(
                                f'<div style="position: relative; width: 100%;">'
                                f'<img src="{data_url}" style="width: 100%; height: 150px; object-fit: cover; border-radius: 8px;">'
                                f'<div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 32px; color: white; opacity: 0.8;">▶</div>',
                                unsafe_allow_html=True
                            )
                    
                    # Title
                    st.markdown(f"**{video.get('filename', 'Untitled')}**")
                    
                    # Duration
                    if video.get('duration'):
                        minutes = int(video['duration'] // 60)
                        seconds = int(video['duration'] % 60)
                        st.caption(f"⏱️ {minutes:02d}:{seconds:02d}")
                    
                    # Size
                    if video.get('file_size'):
                        size_mb = video['file_size'] / (1024 * 1024)
                        st.caption(f"📦 {size_mb:.1f} MB")
                    
                    # View button
                    # Find entry for this video
                    with sqlite3.connect(self.manager.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                        SELECT entry_id FROM album_entries WHERE media_id = ?
                        ''', (video['media_id'],))
                        entry = cursor.fetchone()
                    
                    if entry and st.button("Play", key=f"play_{video['media_id']}", use_container_width=True):
                        st.session_state['selected_media'] = entry[0]
                        st.session_state['current_page'] = 'media_detail'
                        st.rerun()
    
    def render_photos_page(self):
        """Render photo library page"""
        st.title("📸 Photo Library")
        
        # Get all images
        images = self.manager.db.get_media_by_type(MediaType.IMAGE.value)
        
        if not images:
            st.info("No photos found. Add some images to your person folders and scan the directory.")
            return
        
        # Photo stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Photos", len(images))
        with col2:
            total_size = sum(i.get('file_size', 0) for i in images) / (1024*1024)
            st.metric("Total Size", f"{total_size:.2f} MB")
        with col3:
            avg_size = total_size / len(images) if images else 0
            st.metric("Avg Size", f"{avg_size:.1f} MB")
        
        st.divider()
        
        # Photo grid
        st.subheader("All Photos")
        
        cols = st.columns(4)
        for idx, image in enumerate(images):
            with cols[idx % 4]:
                with st.container(border=True):
                    # Thumbnail
                    if image.get('thumbnail_path'):
                        thumbnail_path = Path(image['thumbnail_path'])
                        if thumbnail_path.exists():
                            data_url = self.manager.media_processor.get_media_data_url(thumbnail_path)
                            st.image(data_url, use_column_width=True)
                    
                    # Title
                    st.markdown(f"**{image.get('filename', 'Untitled')}**")
                    
                    # Dimensions
                    if image.get('width') and image.get('height'):
                        st.caption(f"📐 {image['width']}x{image['height']}")
                    
                    # Size
                    if image.get('file_size'):
                        size_mb = image['file_size'] / (1024 * 1024)
                        st.caption(f"📦 {size_mb:.1f} MB")
                    
                    # View button
                    # Find entry for this image
                    with sqlite3.connect(self.manager.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                        SELECT entry_id FROM album_entries WHERE media_id = ?
                        ''', (image['media_id'],))
                        entry = cursor.fetchone()
                    
                    if entry and st.button("View", key=f"view_{image['media_id']}", use_container_width=True):
                        st.session_state['selected_media'] = entry[0]
                        st.session_state['current_page'] = 'media_detail'
                        st.rerun()
    
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
                ["All Fields", "Captions", "Descriptions", "Tags", "People", "Media Type"]
            )
        
        if search_query:
            with st.spinner("Searching..."):
                # Perform search
                results = []
                
                if search_in in ["All Fields", "Captions", "Descriptions", "Tags", "Media Type"]:
                    media_type_filter = None
                    if search_in == "Media Type":
                        if search_query.lower() in ['image', 'video']:
                            media_type_filter = search_query.lower()
                    
                    db_results = self.manager.db.search_entries(
                        search_query if search_in != "Media Type" else "",
                        media_type=media_type_filter
                    )
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
                                SELECT ae.*, p.display_name, m.filename, m.media_type, m.thumbnail_path, m.video_thumbnail_path
                                FROM album_entries ae
                                JOIN people p ON ae.person_id = p.person_id
                                JOIN media m ON ae.media_id = m.media_id
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
        
        total_media = sum(p['media_count'] for p in all_people)
        total_images = sum(p['image_count'] for p in all_people)
        total_videos = sum(p['video_count'] for p in all_people)
        total_comments = sum(p['comment_count'] for p in all_people)
        avg_ratings = [p['avg_rating'] for p in all_people if p['avg_rating'] > 0]
        avg_rating = sum(avg_ratings) / len(avg_ratings) if avg_ratings else 0
        
        with col1:
            st.metric("Total People", len(all_people))
        with col2:
            st.metric("Total Media", total_media)
        with col3:
            st.metric("Total Comments", total_comments)
        with col4:
            st.metric("Average Rating", f"{avg_rating:.1f}")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Media Distribution")
            
            # Prepare data for chart
            labels = ['Images', 'Videos']
            sizes = [total_images, total_videos]
            
            if total_media > 0:
                chart_data = pd.DataFrame({
                    'Type': labels,
                    'Count': sizes
                })
                st.bar_chart(chart_data.set_index('Type'))
        
        with col2:
            st.subheader("👥 Media per Person")
            
            # Prepare data for chart
            people_names = [p['display_name'] for p in all_people]
            media_counts = [p['media_count'] for p in all_people]
            
            if media_counts:
                chart_data = pd.DataFrame({
                    'Person': people_names,
                    'Media': media_counts
                })
                st.bar_chart(chart_data.set_index('Person'))
        
        # Detailed table
        st.divider()
        st.subheader("👥 People Details")
        
        table_data = []
        for person in all_people:
            table_data.append({
                'Name': person['display_name'],
                'Total Media': person['media_count'],
                'Images': person['image_count'],
                'Videos': person['video_count'],
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
                    file_name="photo_video_album_stats.csv",
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
                st.info(f"**Video Thumbnail Directory:** {Config.VIDEO_THUMBNAIL_DIR}")
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
            
            # Video settings
            with st.expander("🎬 Video Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    new_video_thumb_size = st.number_input("Video Thumbnail Width", value=Config.VIDEO_THUMBNAIL_SIZE[0], min_value=100, max_value=800)
                    Config.VIDEO_THUMBNAIL_SIZE = (int(new_video_thumb_size), int(new_video_thumb_size))
                with col2:
                    max_video_size = st.number_input("Max Video Size (MB)", value=Config.MAX_VIDEO_SIZE // (1024*1024), min_value=10, max_value=500)
                    Config.MAX_VIDEO_SIZE = int(max_video_size) * 1024 * 1024
                
                if st.button("Apply Video Settings"):
                    st.success("Video settings updated!")
            
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
                            import shutil
                            # Remove database file
                            if self.manager.db.db_path.exists():
                                os.remove(self.manager.db.db_path)
                            # Reinitialize database
                            self.manager.db = DatabaseManager()
                            # Clear cache
                            self.manager.cache.clear()
                            st.success("All data cleared!")
                            st.rerun()
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
                video_cache_size = len(self.manager.cache._video_cache)
                st.info(f"**Cache Items:** {cache_size}")
                st.info(f"**Video Cache Items:** {video_cache_size}")
                
                if st.button("Clear All Cache"):
                    self.manager.cache.clear()
                    st.success("Cache cleared!")
                
                if st.button("Clear Video Cache"):
                    self.manager.cache.clear_video_cache()
                    st.success("Video cache cleared!")
            
            # Performance settings
            with st.expander("⚡ Performance"):
                st.checkbox("Enable lazy loading", value=True)
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
                        'VIDEO_THUMBNAIL_SIZE': Config.VIDEO_THUMBNAIL_SIZE,
                        'ITEMS_PER_PAGE': Config.ITEMS_PER_PAGE,
                        'CACHE_TTL': Config.CACHE_TTL,
                        'MAX_VIDEO_SIZE': Config.MAX_VIDEO_SIZE
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
                media_types = st.multiselect(
                    "Media Types",
                    ["Image", "Video"],
                    default=["Image", "Video"],
                    key="export_media_types"
                )
            
            if st.button("Generate Export", type="primary"):
                with st.spinner("Generating export file..."):
                    # Prepare export data
                    try:
                        with sqlite3.connect(self.manager.db.db_path) as conn:
                            # Build query
                            conditions = []
                            params = []
                            
                            if export_people:
                                placeholders = ','.join(['?' for _ in export_people])
                                conditions.append(f"p.display_name IN ({placeholders})")
                                params.extend(export_people)
                            
                            if media_types:
                                media_type_values = [mt.lower() for mt in media_types]
                                placeholders = ','.join(['?' for _ in media_type_values])
                                conditions.append(f"m.media_type IN ({placeholders})")
                                params.extend(media_type_values)
                            
                            where_clause = " AND ".join(conditions) if conditions else "1=1"
                            
                            query = f'''
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
                                m.filename,
                                m.media_type,
                                m.file_size,
                                m.format,
                                m.duration,
                                m.width,
                                m.height,
                                (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                                (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count
                            FROM album_entries ae
                            JOIN people p ON ae.person_id = p.person_id
                            JOIN media m ON ae.media_id = m.media_id
                            WHERE {where_clause}
                            ORDER BY ae.created_at DESC
                            '''
                            
                            df = pd.read_sql_query(query, conn, params=params)
                            
                            if export_format == 'csv':
                                output = io.StringIO()
                                df.to_csv(output, index=False)
                                export_data = io.BytesIO(output.getvalue().encode('utf-8'))
                            elif export_format == 'json':
                                output = io.StringIO()
                                df.to_json(output, orient='records', indent=2)
                                export_data = io.BytesIO(output.getvalue().encode('utf-8'))
                            elif export_format == 'excel':
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    df.to_excel(writer, sheet_name='Album Data', index=False)
                                export_data = output
                            
                            if export_data:
                                file_ext = export_format.lower()
                                if file_ext == "excel":
                                    file_ext = "xlsx"
                                
                                st.download_button(
                                    label=f"Download {export_format} File",
                                    data=export_data,
                                    file_name=f"photo_video_album_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                                    mime={
                                        "csv": "text/csv",
                                        "json": "application/json",
                                        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    }[export_format.lower()]
                                )
                    except Exception as e:
                        st.error(f"Error generating export: {str(e)}")
        
        with tabs[1]:  # Import
            st.subheader("Import Data")
            
            st.info("""
            **Import Format:** CSV, JSON, or Excel files with the following columns:
            - `caption` (required): Media caption
            - `person_name` (required): Name of the person
            - `filename` (required): Media filename (must exist in person's folder)
            - `media_type` (required): 'image' or 'video'
            - `description`: Media description
            - `location`: Location where media was taken
            - `tags`: Comma-separated tags
            - `privacy_level`: public/private
            - `duration`: Video duration in seconds (for videos)
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
                        # Read file
                        try:
                            if file_ext == 'csv':
                                df = pd.read_csv(io.BytesIO(import_file.getvalue()))
                            elif file_ext == 'json':
                                df = pd.read_json(io.BytesIO(import_file.getvalue()))
                            elif file_ext == 'excel':
                                df = pd.read_excel(io.BytesIO(import_file.getvalue()))
                            
                            # Validate required columns
                            required_columns = ['caption', 'person_name', 'filename', 'media_type']
                            missing_columns = [col for col in required_columns if col not in df.columns]
                            
                            if missing_columns:
                                st.error(f"Missing required columns: {missing_columns}")
                                return
                            
                            success_count = 0
                            error_count = 0
                            
                            for _, row in df.iterrows():
                                try:
                                    # Find or create person
                                    existing_person = self.manager.db.get_person_by_folder(
                                        row['person_name'].lower().replace(' ', '-')
                                    )
                                    
                                    if not existing_person:
                                        # Create new person
                                        person_profile = PersonProfile(
                                            person_id=str(uuid.uuid4()),
                                            folder_name=row['person_name'].lower().replace(' ', '-'),
                                            display_name=row['person_name'],
                                            bio=f"Photos/Videos of {row['person_name']}",
                                            birth_date=None,
                                            relationship="Family/Friend",
                                            contact_info="",
                                            social_links={},
                                            profile_image=None,
                                            created_at=datetime.datetime.now()
                                        )
                                        self.manager.db.add_person(person_profile)
                                        person_id = person_profile.person_id
                                    else:
                                        person_id = existing_person['person_id']
                                    
                                    # Check if media file exists
                                    media_path = Config.DATA_DIR / person_profile.folder_name / row['filename']
                                    if not media_path.exists():
                                        st.warning(f"Media file not found: {media_path}")
                                        error_count += 1
                                        continue
                                    
                                    # Process media
                                    metadata = MediaMetadata.from_file(media_path)
                                    
                                    # Create thumbnail
                                    thumbnail_path = None
                                    video_thumbnail_path = None
                                    if metadata.media_type == MediaType.IMAGE.value:
                                        thumbnail_path = self.manager.media_processor.create_thumbnail(media_path)
                                    elif metadata.media_type == MediaType.VIDEO.value:
                                        video_thumbnail_path = self.manager.media_processor.create_thumbnail(media_path)
                                    
                                    # Add to database
                                    self.manager.db.add_media(metadata, 
                                                             str(thumbnail_path) if thumbnail_path else None,
                                                             str(video_thumbnail_path) if video_thumbnail_path else None)
                                    
                                    # Parse tags
                                    tags = []
                                    if 'tags' in row and pd.notna(row['tags']):
                                        if isinstance(row['tags'], str):
                                            tags = [tag.strip() for tag in row['tags'].split(',') if tag.strip()]
                                    
                                    # Create album entry
                                    album_entry = AlbumEntry(
                                        entry_id=str(uuid.uuid4()),
                                        media_id=metadata.media_id,
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
                                    
                                    self.manager.db.add_album_entry(album_entry)
                                    success_count += 1
                                    
                                except Exception as e:
                                    error_count += 1
                                    st.error(f"Error processing row: {str(e)}")
                            
                            # Clear cache
                            self.manager.cache.clear()
                            
                            st.success(f"Import completed: {success_count} successful, {error_count} failed")
                            if success_count > 0:
                                st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error importing data: {str(e)}")
        
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
                    try:
                        import shutil
                        # Create backup of database
                        backup_path = Config.EXPORT_DIR / f"{backup_name}.db"
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
                    except Exception as e:
                        st.error(f"Error creating backup: {str(e)}")
            
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
                                import shutil
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
        elif current_page == 'media_detail':
            self.render_media_detail_page()
        elif current_page == 'videos':
            self.render_videos_page()
        elif current_page == 'photos':
            self.render_photos_page()
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
            st.caption("A comprehensive photo & video album management system")

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================
def main():
    """Main application entry point with password protection"""
    # Check password first
    if not check_password():
        return
    
    try:
        # Initialize app
        app = PhotoVideoAlbumApp()
        
        # Check initialization
        if not app.initialized:
            st.error("Application failed to initialize. Check directory permissions.")
            return
        
        # Check video support
        if not VIDEO_SUPPORT:
            st.warning("""
            ⚠️ Video processing libraries are not installed. 
            Video features will be limited.
            
            Install with:
            ```
            pip install opencv-python moviepy
            ```
            """)
        
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
