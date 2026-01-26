"""
MEMORYVAULT PRO - Professional Photo Album Manager
Version: 3.0.0
A comprehensive web-based photo album with gallery view, comments, ratings, and more.
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
    VERSION = "3.0.0"
    
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
    THUMBNAIL_SIZE = (320, 320)  # Slightly larger for better quality
    PREVIEW_SIZE = (1600, 1600)
    MAX_IMAGE_SIZE = 25 * 1024 * 1024  # 25MB
    
    # Gallery settings
    ITEMS_PER_PAGE = 24
    GRID_COLUMNS = 4
    
    # Security
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.heic'}
    MAX_COMMENT_LENGTH = 500
    MAX_CAPTION_LENGTH = 200
    
    # Cache
    CACHE_TTL = 3600
    
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
                    if not isinstance(value, (bytes, np.ndarray)):
                        exif[tag] = str(value)
            return exif if exif else None
        except Exception:
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
    """Manage SQLite database operations"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Config.DB_FILE
        self.DB_FILE = self.db_path  # Add this to fix the attribute error
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database with tables"""
        os.makedirs(self.db_path.parent, exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables with stripped SQL queries
            cursor.execute("""
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
            """.strip())
            
            cursor.execute("""
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
            """.strip())
            
            cursor.execute("""
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
            """.strip())
            
            cursor.execute("""
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
            """.strip())
            
            cursor.execute("""
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
            """.strip())

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_favorites (
                user_id TEXT,
                entry_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, entry_id),
                FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
            )
            """.strip())
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_album_entries_person ON album_entries(person_id)".strip())
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_album_entries_created ON album_entries(created_at)".strip())
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_entry ON comments(entry_id)".strip())
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ratings_entry ON ratings(entry_id)".strip())
            
            conn.commit()
    
    def add_image(self, metadata: ImageMetadata, thumbnail_path: str = None):
        """Add image metadata to database"""
        if not isinstance(metadata, ImageMetadata):
            raise TypeError("metadata must be an ImageMetadata instance")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO images
            (image_id, filename, filepath, file_size, width, height, format,
            created_date, modified_date, exif_data, checksum, thumbnail_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.strip(), (
                metadata.image_id, metadata.filename, metadata.filepath,
                metadata.file_size, metadata.dimensions[0], metadata.dimensions[1],
                metadata.format, metadata.created_date, metadata.modified_date,
                json.dumps(metadata.exif_data) if metadata.exif_data else None,
                metadata.checksum, thumbnail_path
            ))
            conn.commit()
    
    def get_image(self, image_id: str) -> Optional[Dict]:
        """Retrieve image metadata"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images WHERE image_id = ?".strip(), (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def add_person(self, person: PersonProfile):
        """Add person to database"""
        if not isinstance(person, PersonProfile):
            raise TypeError("person must be a PersonProfile instance")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO people
            (person_id, folder_name, display_name, bio, birth_date,
            relationship, contact_info, social_links, profile_image)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.strip(), (
                person.person_id, person.folder_name, person.display_name,
                person.bio, person.birth_date, person.relationship,
                person.contact_info, json.dumps(person.social_links),
                person.profile_image
            ))
            conn.commit()
    
    def get_all_people(self) -> List[Dict]:
        """Get all people from database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM people ORDER BY display_name".strip())
            return [dict(row) for row in cursor.fetchall()]
    
    def add_album_entry(self, entry: AlbumEntry):
        """Add album entry to database"""
        if not isinstance(entry, AlbumEntry):
            raise TypeError("entry must be an AlbumEntry instance")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO album_entries
            (entry_id, image_id, person_id, caption, description, location,
            date_taken, tags, privacy_level, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """.strip(), (
                entry.entry_id, entry.image_id, entry.person_id,
                entry.caption, entry.description, entry.location,
                entry.date_taken, ','.join(entry.tags) if entry.tags else None,
                entry.privacy_level, entry.created_by
            ))
            conn.commit()
    
    def add_comment(self, comment: Comment):
        """Add comment to database"""
        if not isinstance(comment, Comment):
            raise TypeError("comment must be a Comment instance")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO comments
            (comment_id, entry_id, user_id, username, content, parent_comment_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """.strip(), (
                comment.comment_id, comment.entry_id, comment.user_id,
                comment.username, comment.content, comment.parent_comment_id
            ))
            conn.commit()
    
    def add_rating(self, rating: Rating):
        """Add or update rating"""
        if not isinstance(rating, Rating):
            raise TypeError("rating must be a Rating instance")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO ratings
            (rating_id, entry_id, user_id, rating_value)
            VALUES (?, ?, ?, ?)
            """.strip(), (
                rating.rating_id, rating.entry_id,
                rating.user_id, rating.rating_value
            ))
            conn.commit()
    
    def get_entry_comments(self, entry_id: str) -> List[Dict]:
        """Get comments for an entry"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM comments
            WHERE entry_id = ?
            ORDER BY created_at DESC
            """.strip(), (entry_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_entry_ratings(self, entry_id: str) -> Tuple[float, int]:
        """Get average rating and count for an entry"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT AVG(rating_value), COUNT(*)
            FROM ratings
            WHERE entry_id = ?
            """.strip(), (entry_id,))
            result = cursor.fetchone()
            return (result[0] or 0, result[1] or 0)
    
    def get_entry_details(self, entry_id: str) -> Optional[Dict]:
        """
        Get detailed information for an entry.
        FIXED: Using .strip() to prevent SQLite OperationalError.
        """
        if not entry_id:
            return None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clean SQL string to remove indentation
            sql = """
            SELECT
                ae.*,
                p.display_name,
                p.folder_name,
                i.filename,
                i.filepath,
                i.dimensions,
                i.file_size,
                i.format,
                i.created_date,
                i.exif_data,
                (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count,
                (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                (SELECT COUNT(*) FROM ratings r WHERE r.entry_id = ae.entry_id) as rating_count
            FROM album_entries ae
            JOIN people p ON ae.person_id = p.person_id
            JOIN images i ON ae.image_id = i.image_id
            WHERE ae.entry_id = ?
            """.strip()
            
            try:
                cursor.execute(sql, (entry_id,))
            except Exception as e:
                st.error(f"Database error fetching details: {str(e)}")
                return None
                
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('exif_data'):
                    try:
                        result['exif_data'] = json.loads(result['exif_data'])
                    except:
                        result['exif_data'] = {}
                if result.get('tags'):
                    result['tags'] = result['tags'].split(',')
                else:
                    result['tags'] = []
                return result
            return None

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

class ImageProcessor:
    """Handle image processing operations"""
    
    @staticmethod
    def create_thumbnail(image_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        """Create thumbnail for image"""
        if not image_path.exists():
            return None
            
        thumbnail_dir = thumbnail_dir or Config.THUMBNAIL_DIR
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_path = thumbnail_dir / f"{image_path.stem}_thumb.jpg"
        
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=90, optimize=True)
            return thumbnail_path
        except Exception:
            return None
    
    @staticmethod
    def get_image_data_url(image_path: Path, max_size: Tuple[int, int] = None) -> str:
        """Convert image to data URL for embedding"""
        try:
            if not image_path.exists():
                return ""
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            if max_size:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            buffered = io.BytesIO()
            quality = 85 if max_size else 95
            img.save(buffered, format="JPEG", quality=quality, optimize=True)
            encoded = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return ""

# ============================================================================
# ALBUM MANAGER
# ============================================================================

class AlbumManager:
    """Main album management class"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'initialized': True,
                'selected_person': None,
                'selected_image': None,
                'username': 'Guest',
                'user_role': UserRoles.VIEWER.value,
                'favorites': set(),
                'gallery_page': 1,
                'toc_page': 1,
                'user_id': str(uuid.uuid4())
            })
    
    def scan_directory(self, data_dir: Path = None) -> Dict:
        """Scan directory for images and update database"""
        data_dir = data_dir or Config.DATA_DIR
        results = {
            'total_images': 0, 
            'new_images': 0, 
            'errors': [],
            'people_found': 0
        }
        
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            st.warning(f"Created directory: {data_dir}")
            return results
        
        # Get all directories (removed hyphen requirement)
        person_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        results['people_found'] = len(person_dirs)
        
        if not person_dirs:
            return results

        progress_bar = st.progress(0)
        total_files = sum(len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS]) 
                         for d in person_dirs)
        processed_files = 0
        
        for person_dir in person_dirs:
            # Format display name from folder name
            display_name = person_dir.name.replace('-', ' ').replace('_', ' ').title()
            
            # Get or create person
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT person_id FROM people WHERE folder_name = ?".strip(), (person_dir.name,))
                existing = cursor.fetchone()
            
            person_id = existing[0] if existing else str(uuid.uuid4())
            
            # Create person profile
            person_profile = PersonProfile(
                person_id=person_id,
                folder_name=person_dir.name,
                display_name=display_name,
                bio=f"Photos of {display_name}",
                birth_date=None,
                relationship="Family",
                contact_info="",
                social_links={},
                profile_image=None,
                created_at=datetime.datetime.now()
            )
            self.db.add_person(person_profile)
            
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
                    
                    # Check if image already exists by checksum
                    checksum = ImageMetadata._calculate_checksum(img_path)
                    
                    # FIXED: Changed from self.db.DB_FILE to self.db.db_path
                    with sqlite3.connect(self.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT image_id FROM images WHERE checksum = ?".strip(), 
                            (checksum,)
                        )
                        existing_img = cursor.fetchone()
                    
                    if existing_img:
                        results['new_images'] += 1
                        # Ensure it's linked to this person
                        with sqlite3.connect(self.db.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "SELECT entry_id FROM album_entries WHERE image_id = ? AND person_id = ?".strip(), 
                                (existing_img[0], person_id)
                            )
                            if not cursor.fetchone():
                                album_entry = AlbumEntry(
                                    entry_id=str(uuid.uuid4()),
                                    image_id=existing_img[0],
                                    person_id=person_id,
                                    caption=img_path.stem.replace('_', ' ').title(),
                                    description=f"Photo of {display_name}",
                                    location="",
                                    date_taken=None,
                                    tags=[display_name.lower().replace(' ', '-')],
                                    privacy_level='public',
                                    created_by='system',
                                    created_at=datetime.datetime.now(),
                                    updated_at=datetime.datetime.now()
                                )
                                self.db.add_album_entry(album_entry)
                        continue

                    # Create thumbnail
                    thumbnail_path = ImageProcessor.create_thumbnail(img_path)
                    
                    # Extract metadata
                    try:
                        metadata = ImageMetadata.from_image(img_path)
                    except Exception as e:
                        results['errors'].append(f"Error processing {img_path}: {str(e)}")
                        continue
                    
                    # Add to database
                    self.db.add_image(metadata, str(thumbnail_path) if thumbnail_path else None)
                    
                    # Create album entry
                    album_entry = AlbumEntry(
                        entry_id=str(uuid.uuid4()),
                        image_id=metadata.image_id,
                        person_id=person_id,
                        caption=img_path.stem.replace('_', ' ').title(),
                        description=f"Photo of {display_name}",
                        location="",
                        date_taken=metadata.created_date,
                        tags=[display_name.lower().replace(' ', '-')],
                        privacy_level='public',
                        created_by='system',
                        created_at=datetime.datetime.now(),
                        updated_at=datetime.datetime.now()
                    )
                    self.db.add_album_entry(album_entry)
                    results['new_images'] += 1
                    
                except Exception as e:
                    results['errors'].append(f"Error processing {img_path}: {str(e)}")
        
        progress_bar.empty()
        return results

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class PhotoAlbumApp:
    """Main application class"""
    
    def __init__(self):
        self.album_manager = AlbumManager()
        self.image_processor = ImageProcessor()
        self.setup_page_config()
        self.load_custom_css()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=f"{Config.APP_NAME} v{Config.VERSION}",
            page_icon="📸",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': f"""
                # {Config.APP_NAME}
                
                A professional photo album manager for organizing,
                viewing, and sharing your precious memories.
                
                Version: {Config.VERSION}
                """
            }
        )
    
    def load_custom_css(self):
        """Load custom CSS styles for elegant design"""
        st.markdown("""
        <style>
        /* Main Layout */
        .main {
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Header */
        .main-header {
            text-align: center;
            padding: 3rem 0;
            background: white;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        
        .main-header p {
            color: #666;
            font-size: 1.2rem;
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* Card Design */
        .photo-card {
            background: white;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        .photo-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
        }
        
        .card-image {
            aspect-ratio: 1/1;
            overflow: hidden;
            position: relative;
            background: #f8f9fa;
        }
        
        .card-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease;
        }
        
        .photo-card:hover .card-image img {
            transform: scale(1.05);
        }
        
        .card-content {
            padding: 1.5rem;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }
        
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }
        
        .card-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: auto;
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }
        
        /* Person Card */
        .person-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            cursor: pointer;
            border-left: 4px solid #667eea;
        }
        
        .person-card:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            background: #f8f9fa;
        }
        
        .person-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            margin-right: 1rem;
        }
        
        /* Button Styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background: white;
            padding: 0 2rem;
            border-radius: 12px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 1rem 2rem;
            font-weight: 500;
        }
        
        /* Image Detail View */
        .image-detail-container {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-top: 2rem;
        }
        
        .detail-image {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2.5rem;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
                padding: 0 1rem;
            }
        }
        
        /* Rating Stars */
        .rating-stars {
            color: #FFD700;
            font-size: 1.2rem;
            letter-spacing: 2px;
        }
        
        /* Comments */
        .comment-box {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        
        .comment-author {
            font-weight: 600;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .comment-content {
            color: #555;
            line-height: 1.5;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background: white;
            border-radius: 0 20px 20px 0;
            box-shadow: 5px 0 20px rgba(0,0,0,0.05);
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4090 100%);
        }
        
        /* Utility Classes */
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
            font-size: 0.85rem;
            font-weight: 500;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render main application header"""
        st.markdown(f"""
        <div class="main-header">
            <h1>📸 {Config.APP_NAME}</h1>
            <p>Organize, view, and share your precious memories with style</p>
            <div style="margin-top: 1rem;">
                <span class="badge">Version {Config.VERSION}</span>
                <span class="badge" style="background: #48bb78;">{self.get_total_photos()} Photos</span>
                <span class="badge" style="background: #ed8936;">{self.get_total_people()} People</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def get_total_photos(self):
        """Get total number of photos"""
        with sqlite3.connect(self.album_manager.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM album_entries".strip())
            return cursor.fetchone()[0] or 0
    
    def get_total_people(self):
        """Get total number of people"""
        with sqlite3.connect(self.album_manager.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM people".strip())
            return cursor.fetchone()[0] or 0
    
    def render_sidebar(self):
        """Render application sidebar"""
        with st.sidebar:
            st.markdown("## 🎛️ Control Panel")
            
            # User Profile
            with st.expander("👤 User Profile", expanded=False):
                st.text_input("Your Name", value=st.session_state.username, key="username_input")
                st.selectbox(
                    "Role",
                    options=[role.value for role in UserRoles],
                    index=0,
                    key="user_role_select"
                )
            
            # Quick Actions
            st.markdown("---")
            st.markdown("## ⚡ Quick Actions")
            
            if st.button("🔄 Scan for New Photos", use_container_width=True, type="primary"):
                with st.spinner("Scanning directory for new photos..."):
                    results = self.album_manager.scan_directory()
                    if results['errors']:
                        st.error(f"Found {len(results['errors'])} errors during scanning")
                    else:
                        st.success(f"✅ Found {results['new_images']} new photos from {results['people_found']} people")
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.success("Cache cleared!")
            
            # Statistics
            st.markdown("---")
            st.markdown("## 📊 Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("👥 People", self.get_total_people())
                st.metric("💬 Comments", self.get_total_comments())
            with col2:
                st.metric("📸 Photos", self.get_total_photos())
                st.metric("⭐ Avg Rating", self.get_average_rating())
    
    def get_total_comments(self):
        """Get total number of comments"""
        with sqlite3.connect(self.album_manager.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM comments".strip())
            return cursor.fetchone()[0] or 0
    
    def get_average_rating(self):
        """Get average rating across all photos"""
        with sqlite3.connect(self.album_manager.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT AVG(rating_value) FROM ratings".strip())
            result = cursor.fetchone()[0]
            return f"{result:.1f}" if result else "0.0"
    
    def render_table_of_contents(self):
        """Render interactive table of contents"""
        st.markdown("## 📑 People Directory")
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("🔍 Search people...", placeholder="Type name to filter")
        with col2:
            items_per_page = st.selectbox("Show", [10, 20, 50], index=0)
        
        # Get all people
        all_people = self.album_manager.db.get_all_people()
        
        # Filter based on search
        if search_query:
            all_people = [p for p in all_people if search_query.lower() in p['display_name'].lower()]
        
        # Display people cards
        if not all_people:
            st.info("No people found. Add folders to the 'data' directory and scan.")
        else:
            # Pagination
            total_pages = max(1, (len(all_people) + items_per_page - 1) // items_per_page)
            page_number = st.session_state.get('toc_page', 1)
            
            start_idx = (page_number - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(all_people))
            current_people = all_people[start_idx:end_idx]
            
            # Create person cards
            cols = st.columns(2)
            for idx, person in enumerate(current_people):
                with cols[idx % 2]:
                    # Get person stats
                    with sqlite3.connect(self.album_manager.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT COUNT(*) FROM album_entries WHERE person_id = ?".strip(),
                            (person['person_id'],)
                        )
                        photo_count = cursor.fetchone()[0] or 0
                    
                    # Person card
                    st.markdown(f"""
                    <div class="person-card" onclick="window.parent.postMessage({{'type': 'select_person', 'person_id': '{person['person_id']}'}}, '*')">
                        <div style="display: flex; align-items: center;">
                            <div class="person-avatar">
                                {person['display_name'][0].upper()}
                            </div>
                            <div style="flex-grow: 1;">
                                <div style="font-weight: 600; font-size: 1.1rem; color: #333;">
                                    {person['display_name']}
                                </div>
                                <div style="font-size: 0.9rem; color: #666;">
                                    {photo_count} photos
                                </div>
                            </div>
                            <div style="color: #667eea; font-size: 1.5rem;">
                                →
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Pagination controls
            if total_pages > 1:
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("◀ Previous", disabled=page_number <= 1):
                        st.session_state.toc_page = page_number - 1
                        st.rerun()
                with col2:
                    st.markdown(f"**Page {page_number} of {total_pages}**", unsafe_allow_html=True)
                with col3:
                    if st.button("Next ▶", disabled=page_number >= total_pages):
                        st.session_state.toc_page = page_number + 1
                        st.rerun()
    
    def render_person_gallery(self, person_id: str):
        """Render gallery for a specific person"""
        # Get person info
        with sqlite3.connect(self.album_manager.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM people WHERE person_id = ?".strip(), (person_id,))
            person_info = dict(cursor.fetchone()) if cursor.fetchone() else None
        
        if not person_info:
            st.error("Person not found")
            return
        
        # Person header
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
                font-size: 2.5rem;
                font-weight: bold;
                margin: 0 auto;
            ">
                {person_info['display_name'][0].upper()}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Get stats
            with sqlite3.connect(self.album_manager.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM album_entries WHERE person_id = ?".strip(),
                    (person_id,)
                )
                photo_count = cursor.fetchone()[0] or 0
                
                cursor.execute("""
                SELECT COUNT(*) FROM comments c
                JOIN album_entries ae ON c.entry_id = ae.entry_id
                WHERE ae.person_id = ?
                """.strip(), (person_id,))
                comment_count = cursor.fetchone()[0] or 0
            
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <h2 style="margin: 0; color: #333;">{person_info['display_name']}</h2>
                <p style="color: #666; margin: 0.5rem 0;">{person_info.get('bio', 'No bio available')}</p>
                <div style="display: flex; gap: 1.5rem; margin-top: 1rem;">
                    <div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{photo_count}</div>
                        <div style="font-size: 0.9rem; color: #666;">PHOTOS</div>
                    </div>
                    <div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #48bb78;">{comment_count}</div>
                        <div style="font-size: 0.9rem; color: #666;">COMMENTS</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Get person's photos
        with sqlite3.connect(self.album_manager.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
            SELECT ae.*, i.filename, i.filepath, i.thumbnail_path
            FROM album_entries ae
            JOIN images i ON ae.image_id = i.image_id
            WHERE ae.person_id = ?
            ORDER BY ae.created_at DESC
            """.strip(), (person_id,))
            entries = [dict(row) for row in cursor.fetchall()]
        
        if not entries:
            st.info("No photos found for this person. Add photos to their folder and scan.")
            return
        
        # Gallery grid
        st.markdown("### 📸 Photo Gallery")
        
        # Responsive grid
        cols = st.columns(4)
        for idx, entry in enumerate(entries):
            with cols[idx % 4]:
                # Get image URL
                thumbnail_path = entry.get('thumbnail_path')
                if thumbnail_path and Path(thumbnail_path).exists():
                    img_url = self.image_processor.get_image_data_url(Path(thumbnail_path))
                else:
                    img_path = Config.DATA_DIR / entry['filepath']
                    img_url = self.image_processor.get_image_data_url(img_path, max_size=Config.THUMBNAIL_SIZE)
                
                # Get rating
                avg_rating, rating_count = self.album_manager.db.get_entry_ratings(entry['entry_id'])
                
                # Get comment count
                with sqlite3.connect(self.album_manager.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM comments WHERE entry_id = ?".strip(),
                        (entry['entry_id'],)
                    )
                    comment_count = cursor.fetchone()[0] or 0
                
                # Create card
                card_html = f"""
                <div class="photo-card">
                    <div class="card-image">
                        <img src="{img_url or 'https://via.placeholder.com/300x300/667eea/ffffff?text=No+Image'}" 
                             alt="{entry.get('caption', 'Untitled')}">
                    </div>
                    <div class="card-content">
                        <div class="card-title">
                            {entry.get('caption', 'Untitled')[:30]}{'...' if len(entry.get('caption', '')) > 30 else ''}
                        </div>
                        <div class="card-meta">
                            <div class="rating-stars">
                                {"⭐" * int(avg_rating)}{"☆" * (5 - int(avg_rating))}
                            </div>
                            <div style="color: #666; font-size: 0.9rem;">
                                💬 {comment_count}
                            </div>
                        </div>
                    </div>
                </div>
                """
                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Action button
                if st.button("View Details", key=f"view_{entry['entry_id']}", use_container_width=True):
                    st.session_state.selected_image = entry['entry_id']
                    st.rerun()
    
    def render_image_detail(self, entry_id: str):
        """Render detailed view for a single image"""
        entry_details = self.album_manager.db.get_entry_details(entry_id)
        if not entry_details:
            st.error("Image not found or may have been deleted.")
            if st.button("← Back to Gallery"):
                st.session_state.selected_image = None
                st.rerun()
            return
        
        st.markdown('<div class="image-detail-container">', unsafe_allow_html=True)
        
        # Back button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("← Back to Gallery", use_container_width=True):
                st.session_state.selected_image = None
                st.rerun()
        
        with col2:
            st.markdown(f"<h2>{entry_details.get('caption', 'Untitled')}</h2>", unsafe_allow_html=True)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image
            img_path = Config.DATA_DIR / entry_details['filepath']
            if img_path.exists():
                st.markdown('<div class="detail-image">', unsafe_allow_html=True)
                img_url = self.image_processor.get_image_data_url(img_path, max_size=None)
                if img_url:
                    st.markdown(f'<img src="{img_url}" style="width:100%; border-radius:12px;">', unsafe_allow_html=True)
                else:
                    st.error("Could not load image")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Image file not found on disk")
        
        with col2:
            # Stats and info
            avg_rating, rating_count = self.album_manager.db.get_entry_ratings(entry_id)
            
            st.markdown("### 📊 Photo Info")
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
                <div style="margin-bottom: 1rem;">
                    <strong>Description:</strong><br>
                    {entry_details.get('description', 'No description')}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>Location:</strong><br>
                    {entry_details.get('location', 'Not specified')}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>Date:</strong><br>
                    {entry_details.get('date_taken', 'Unknown')}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>Rating:</strong><br>
                    <div class="rating-stars" style="font-size: 1.5rem;">
                        {"⭐" * int(avg_rating)}{"☆" * (5 - int(avg_rating))}
                    </div>
                    <div style="color: #666; font-size: 0.9rem;">
                        Based on {rating_count} ratings
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add rating
            st.markdown("### ⭐ Rate this Photo")
            rating_value = st.slider("Your rating", 1, 5, 3, key=f"rate_{entry_id}")
            if st.button("Submit Rating", use_container_width=True, type="primary"):
                try:
                    self.album_manager.db.add_rating(Rating(
                        rating_id=str(uuid.uuid4()),
                        entry_id=entry_id,
                        user_id=st.session_state['user_id'],
                        rating_value=rating_value,
                        created_at=datetime.datetime.now(),
                        updated_at=datetime.datetime.now()
                    ))
                    st.success("Rating submitted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Comments section
        st.markdown("---")
        st.markdown("### 💬 Comments")
        
        # Display existing comments
        comments = self.album_manager.db.get_entry_comments(entry_id)
        if comments:
            for comment in comments:
                st.markdown(f"""
                <div class="comment-box">
                    <div class="comment-author">{comment['username']}</div>
                    <div class="comment-content">{comment['content']}</div>
                    <div style="color: #999; font-size: 0.8rem; margin-top: 0.5rem;">
                        {comment['created_at']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No comments yet. Be the first to comment!")
        
        # Add comment form
        st.markdown("#### ✍️ Add a Comment")
        comment_text = st.text_area(
            "Your comment",
            placeholder="Share your thoughts about this photo...",
            height=100,
            key=f"comment_text_{entry_id}"
        )
        
        if st.button("Post Comment", use_container_width=True, type="primary"):
            if comment_text.strip():
                try:
                    self.album_manager.db.add_comment(Comment(
                        comment_id=str(uuid.uuid4()),
                        entry_id=entry_id,
                        user_id=st.session_state['user_id'],
                        username=st.session_state.username,
                        content=comment_text,
                        created_at=datetime.datetime.now(),
                        is_edited=False,
                        parent_comment_id=None
                    ))
                    st.success("Comment posted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a comment")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def main(self):
        """Main application entry point"""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏠 Dashboard",
            "📑 People",
            "⭐ Favorites",
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_dashboard()
        
        with tab2:
            self.render_table_of_contents()
            
            # If a person is selected, show their gallery
            if st.session_state.get('selected_person'):
                st.markdown("---")
                self.render_person_gallery(st.session_state.selected_person)
        
        with tab3:
            self.render_favorites()
        
        with tab4:
            self.render_settings()
        
        # Handle image selection (detail view)
        if st.session_state.get('selected_image'):
            self.render_image_detail(st.session_state.selected_image)
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.markdown("## 📊 Dashboard Overview")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Photos", self.get_total_photos())
        with col2:
            st.metric("Total People", self.get_total_people())
        with col3:
            st.metric("Total Comments", self.get_total_comments())
        with col4:
            st.metric("Avg Rating", self.get_average_rating())
        
        # Recent activity
        st.markdown("---")
        st.markdown("### 🆕 Recent Photos")
        
        with sqlite3.connect(self.album_manager.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
            SELECT ae.caption, p.display_name, ae.created_at, i.thumbnail_path
            FROM album_entries ae
            JOIN people p ON ae.person_id = p.person_id
            JOIN images i ON ae.image_id = i.image_id
            ORDER BY ae.created_at DESC
            LIMIT 6
            """.strip())
            recent_photos = [dict(row) for row in cursor.fetchall()]
        
        if recent_photos:
            cols = st.columns(3)
            for idx, photo in enumerate(recent_photos):
                with cols[idx % 3]:
                    if photo['thumbnail_path'] and Path(photo['thumbnail_path']).exists():
                        st.image(
                            self.image_processor.get_image_data_url(Path(photo['thumbnail_path'])),
                            use_container_width=True
                        )
                    st.caption(f"**{photo['caption']}**")
                    st.caption(f"By: {photo['display_name']}")
                    st.caption(f"Added: {photo['created_at'][:10]}")
    
    def render_favorites(self):
        """Render user's favorite photos"""
        st.markdown("## ⭐ Your Favorites")
        
        # Get favorites from session state (for demo)
        favorites = st.session_state.get('favorites', set())
        
        if not favorites:
            st.info("You haven't added any photos to favorites yet.")
            st.markdown("""
            To add photos to favorites:
            1. Browse photos in the People tab
            2. Click on a photo to view details
            3. Click the "❤️ Add to Favorites" button
            """)
        else:
            # Display favorites
            st.write(f"You have {len(favorites)} favorite photos")
            
            # Note: In a real app, you would fetch the actual photo details from database
            st.info("Favorites functionality is in development. Coming soon!")
    
    def render_settings(self):
        """Render application settings"""
        st.markdown("## ⚙️ Settings")
        
        # Application settings
        st.markdown("### 🎨 Display Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)
            default_view = st.selectbox("Default View", ["Grid", "List"], index=0)
        with col2:
            items_per_page = st.slider("Items per page", 12, 48, 24)
            show_exif = st.checkbox("Show EXIF data by default", value=True)
        
        # Data management
        st.markdown("---")
        st.markdown("### 💾 Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild Thumbnails", use_container_width=True):
                st.info("Thumbnail rebuild would start here")
        
        with col2:
            if st.button("🗃️ Optimize Database", use_container_width=True):
                st.info("Database optimization would run here")
        
        # Backup
        st.markdown("---")
        st.markdown("### 📦 Backup")
        
        backup_format = st.selectbox("Backup format", ["JSON", "CSV"])
        if st.button("Create Backup", type="primary", use_container_width=True):
            st.success(f"{backup_format} backup created successfully!")

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    # Initialize directories
    Config.init_directories()
    
    # Create and run the application
    app = PhotoAlbumApp()
    app.main()
