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
    VERSION = "2.3.0"  # Stability Upgrade
    
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
    PREVIEW_SIZE = (1600, 1600) # Increased for better "capable" viewing
    MAX_IMAGE_SIZE = 25 * 1024 * 1024  # 25MB
    
    # Gallery settings
    ITEMS_PER_PAGE = 20
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
        if not image_path.exists():
            raise FileNotFoundError(f"File not found for checksum: {image_path}")
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

@dataclass
class AlbumEntry:
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
        data = asdict(self)
        data['date_taken'] = self.date_taken.isoformat() if self.date_taken else None
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class Comment:
    comment_id: str
    entry_id: str
    user_id: str
    username: str
    content: str
    created_at: datetime.datetime
    is_edited: bool
    parent_comment_id: Optional[str]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class Rating:
    rating_id: str
    entry_id: str
    user_id: str
    rating_value: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

@dataclass
class PersonProfile:
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
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        os.makedirs(self.db_path.parent, exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create tables - using .strip() to fix OperationalError from indentation
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
            
            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_album_entries_person ON album_entries(person_id)".strip())
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_album_entries_created ON album_entries(created_at)".strip())
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_entry ON comments(entry_id)".strip())
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ratings_entry ON ratings(entry_id)".strip())
            
            conn.commit()
    
    def add_image(self, metadata: ImageMetadata, thumbnail_path: str = None):
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
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images WHERE image_id = ?".strip(), (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def add_person(self, person: PersonProfile):
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
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM people ORDER BY display_name".strip())
            return [dict(row) for row in cursor.fetchall()]
    
    def add_album_entry(self, entry: AlbumEntry):
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
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM comments
            WHERE entry_id = ?
            ORDER BY created_at DESC
            """.strip(), (entry_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_entry_ratings(self, entry_id: str) -> Tuple[float, int]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT AVG(rating_value), COUNT(*)
            FROM ratings
            WHERE entry_id = ?
            """.strip(), (entry_id,))
            result = cursor.fetchone()
            return (result[0] or 0, result[1] or 0)
    
    def search_entries(self, query: str, person_id: str = None) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            search_pattern = f'%{query}%'
            
            if person_id:
                sql = """
                SELECT ae.*, p.display_name, i.filename
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN images i ON ae.image_id = i.image_id
                WHERE ae.person_id = ? AND
                (ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)
                ORDER BY ae.created_at DESC
                """.strip()
                cursor.execute(sql, (person_id, search_pattern, search_pattern, search_pattern))
            else:
                sql = """
                SELECT ae.*, p.display_name, i.filename
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN images i ON ae.image_id = i.image_id
                WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?
                ORDER BY ae.created_at DESC
                """.strip()
                cursor.execute(sql, (search_pattern, search_pattern, search_pattern))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_entry_details(self, entry_id: str) -> Optional[Dict]:
        """
        Get detailed information for an entry.
        FIXED: Using .strip() to prevent SQLite OperationalError.
        """
        if not entry_id:
            return None

        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
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
# CACHE MANAGEMENT
# ============================================================================
class CacheManager:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str, default=None):
        if key in self._cache:
            if time.time() - self._timestamps[key] < Config.CACHE_TTL:
                return self._cache[key]
            else:
                del self._cache[key]
                del self._timestamps[key]
        return default
    
    def set(self, key: str, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self, key: str = None):
        if key:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
        else:
            self._cache.clear()
            self._timestamps.clear()

# ============================================================================
# UI COMPONENTS
# ============================================================================
class UIComponents:
    @staticmethod
    def rating_stars(rating: float, size: int = 20) -> str:
        if rating is None or rating <= 0:
            rating = 0
        stars_html = []
        full_stars = int(rating)
        has_half_star = rating - full_stars >= 0.5
        for i in range(5):
            if i < full_stars:
                stars_html.append('⭐')
            elif i == full_stars and has_half_star:
                stars_html.append('⭐')
            else:
                stars_html.append('☆')
        return f"""
        <div style="color: #FFD700; font-size: {size}px; letter-spacing: 1px; display: inline-block;">
        {''.join(stars_html)}
        <span style="color: #666; font-size: 14px; margin-left: 8px;">
        {rating:.1f}/5.0
        </span>
        </div>
        """
    
    @staticmethod
    def image_card(image_url: str, caption: str, comments_count: int, rating: float, entry_id: str) -> str:
        stars = UIComponents.rating_stars(rating)
        if not image_url:
            image_url = "https://via.placeholder.com/300x300/667eea/ffffff?text=No+Image"
        
        return f'''
        <div style="
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #eee;
            height: 100%;
            transition: transform 0.2s;
            display: flex;
            flex-direction: column;
        ">
            <div style="
                aspect-ratio: 1/1;
                overflow: hidden;
                background: #f8f9fa;
                position: relative;
                cursor: pointer;
            ">
                <img src="{image_url}" 
                     alt="{caption}"
                     loading="lazy"
                     style="
                        width: 100%; 
                        height: 100%; 
                        object-fit: cover;
                        transition: opacity 0.3s;
                     ">
            </div>
            <div style="padding: 12px; flex-grow: 1; display: flex; flex-direction: column;">
                <div style="font-weight: 600; font-size: 14px; color: #333; margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                {caption}
                </div>
                <div style="margin-top: auto; display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: #666; border-top: 1px solid #f0f0f0; padding-top: 8px;">
                    <div>{stars}</div>
                    <div style="display: flex; gap: 8px;">
                        <span>💬 {comments_count}</span>
                    </div>
                </div>
            </div>
        </div>
        '''

# ============================================================================
# ALBUM MANAGER
# ============================================================================
class AlbumManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.image_processor = ImageProcessor()
        self._init_session_state()
    
    def _init_session_state(self):
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'initialized': True, 'selected_person': None, 'selected_image': None,
                'username': 'Guest', 'user_role': UserRoles.VIEWER.value,
                'favorites': set(), 'gallery_page': 1, 'toc_page': 1
            })
    
    def scan_directory(self, data_dir: Path = None) -> Dict:
        data_dir = data_dir or Config.DATA_DIR
        results = {'total_images': 0, 'new_images': 0, 'errors': []}
        
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            st.warning(f"Created directory: {data_dir}")
            return results
        
        # Fix: Removed hyphen requirement
        person_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not person_dirs:
            return results

        progress_bar = st.progress(0)
        total_files = sum(len(list(d.iterdir())) for d in person_dirs)
        processed_files = 0
        
        for person_dir in person_dirs:
            display_name = person_dir.name.replace('-', ' ').replace('_', ' ').title()
            
            # Get or create person
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT person_id FROM people WHERE folder_name = ?".strip(), (person_dir.name,))
                existing = cursor.fetchone()
            
            person_id = existing[0] if existing else str(uuid.uuid4())
            
            person_profile = PersonProfile(
                person_id=person_id, folder_name=person_dir.name, display_name=display_name,
                bio=f"Photos of {display_name}", birth_date=None,
                relationship="Family", contact_info="", social_links={},
                profile_image=None, created_at=datetime.datetime.now()
            )
            self.db.add_person(person_profile)
            
            # Process Images
            image_files = [f for f in person_dir.iterdir() if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS]
            
            for img_path in image_files:
                try:
                    processed_files += 1
                    progress = processed_files / max(total_files, 1)
                    progress_bar.progress(progress)
                    
                    checksum = ImageMetadata._calculate_checksum(img_path)
                    
                    with sqlite3.connect(self.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT image_id FROM images WHERE checksum = ?".strip(), (checksum,))
                        existing_img = cursor.fetchone()
                    
                    if existing_img:
                        results['new_images'] += 1 # Counted as found
                        # Ensure it's linked to this person
                        with sqlite3.connect(self.db.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT entry_id FROM album_entries WHERE image_id = ? AND person_id = ?".strip(), (existing_img[0], person_id))
                            if not cursor.fetchone():
                                ae = AlbumEntry(
                                    entry_id=str(uuid.uuid4()), image_id=existing_img[0], person_id=person_id,
                                    caption=img_path.stem.replace('_', ' ').title(),
                                    description=f"Photo of {display_name}", location="",
                                    date_taken=ImageMetadata.from_image(img_path).created_date,
                                    tags=[display_name.lower().replace(' ', '-')], privacy_level='public',
                                    created_by='system', created_at=datetime.datetime.now(),
                                    updated_at=datetime.datetime.now()
                                )
                                self.db.add_album_entry(ae)
                        continue

                    thumbnail_path = self.image_processor.create_thumbnail(img_path)
                    try:
                        metadata = ImageMetadata.from_image(img_path)
                    except Exception:
                        continue
                    
                    self.db.add_image(metadata, str(thumbnail_path) if thumbnail_path else None)
                    
                    ae = AlbumEntry(
                        entry_id=str(uuid.uuid4()), image_id=metadata.image_id, person_id=person_id,
                        caption=img_path.stem.replace('_', ' ').title(),
                        description=f"Photo of {display_name}", location="",
                        date_taken=metadata.created_date,
                        tags=[display_name.lower().replace(' ', '-')], privacy_level='public',
                        created_by='system', created_at=datetime.datetime.now(), updated_at=datetime.datetime.now()
                    )
                    self.db.add_album_entry(ae)
                    results['new_images'] += 1
                    
                except Exception as e:
                    results['errors'].append(str(e))
        
        progress_bar.empty()
        return results
    
    def add_comment_to_entry(self, entry_id: str, content: str, username: str = None):
        if not username:
            username = st.session_state.get('username', 'Anonymous')
        comment = Comment(
            comment_id=str(uuid.uuid4()), entry_id=entry_id, user_id=st.session_state['user_id'],
            username=username, content=content, created_at=datetime.datetime.now(),
            is_edited=False, parent_comment_id=None
        )
        self.db.add_comment(comment)
        return comment
    
    def rate_entry(self, entry_id: str, rating_value: int):
        if not 1 <= rating_value <= 5:
            raise ValueError("Rating must be between 1 and 5")
        rating = Rating(
            rating_id=str(uuid.uuid4()), entry_id=entry_id, user_id=st.session_state['user_id'],
            rating_value=rating_value, created_at=datetime.datetime.now(), updated_at=datetime.datetime.now()
        )
        self.db.add_rating(rating)
        return rating
    
    def add_to_favorites(self, entry_id: str):
        user_id = st.session_state['user_id']
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO user_favorites (user_id, entry_id) VALUES (?, ?)".strip(), (user_id, entry_id))
            conn.commit()
        st.session_state.favorites.add(entry_id)

    def remove_from_favorites(self, entry_id: str):
        user_id = st.session_state['user_id']
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_favorites WHERE user_id = ? AND entry_id = ?".strip(), (user_id, entry_id))
            conn.commit()
        if entry_id in st.session_state.favorites:
            st.session_state.favorites.remove(entry_id)
    
    def get_user_favorites(self) -> List[Dict]:
        user_id = st.session_state['user_id']
        with sqlite3.connect(self.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
            SELECT ae.*, p.display_name, i.thumbnail_path
            FROM user_favorites uf
            JOIN album_entries ae ON uf.entry_id = ae.entry_id
            JOIN people p ON ae.person_id = p.person_id
            JOIN images i ON ae.image_id = i.image_id
            WHERE uf.user_id = ?
            ORDER BY uf.created_at DESC
            """.strip(), (user_id,))
            return [dict(row) for row in cursor.fetchall()]

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PhotoAlbumApp:
    def __init__(self):
        self.album_manager = AlbumManager()
        self.setup_page_config()
        self.load_custom_css()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title=f"{Config.APP_NAME} v{Config.VERSION}",
            page_icon="👨‍👩‍👧‍👦",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_custom_css(self):
        st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stButton>button { width: 100%; }
        .img-container { text-align: center; margin-bottom: 20px; }
        .img-fluid { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #1a202c; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        .card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #e2e8f0; }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        with st.sidebar:
            st.title("👨‍👩‍👧‍👦 MemoryVault")
            st.markdown("---")
            if st.button("📸 Scan for New Photos", use_container_width=True):
                with st.spinner("Scanning..."):
                    res = self.album_manager.scan_directory()
                st.success(f"Processed {res['new_images']} new images.")
            
            st.markdown("### Profile")
            st.text_input("Name", value=st.session_state.username, key="username")
            
            st.markdown("---")
            st.markdown("### Quick Stats")
            with sqlite3.connect(self.album_manager.db.db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM people".strip())
                people = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM album_entries".strip())
                photos = c.fetchone()[0]
                st.metric("People", people)
                st.metric("Photos", photos)
    
    def render_image_detail(self, entry_id: str):
        # Added safety check
        if not entry_id:
            st.warning("No image selected.")
            return

        entry_details = self.album_manager.db.get_entry_details(entry_id)
        
        if not entry_details:
            st.error("Image not found or may have been deleted from the database.")
            st.button("Back to Gallery", on_click=lambda: st.session_state.update(selected_image=None))
            return

        # Back button
        if st.button("⬅️ Back to Gallery"):
            st.session_state.selected_image = None
            st.rerun()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            img_path = Config.DATA_DIR / entry_details['filepath']
            if img_path.exists():
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                img_url = self.album_manager.image_processor.get_image_data_url(img_path, max_size=None)
                if img_url:
                    st.markdown(f'<img src="{img_url}" class="img-fluid" alt="Photo">', unsafe_allow_html=True)
                else:
                    st.error("Could not load image file.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(f"<h2>{entry_details.get('caption', 'Untitled')}</h2>", unsafe_allow_html=True)
                st.write(entry_details.get('description', ''))
            else:
                st.error("Original image file is missing from disk.")
        
        with col2:
            avg, count = self.album_manager.db.get_entry_ratings(entry_id)
            st.markdown(f"**Rating:** {UIComponents.rating_stars(avg)} ({count} votes)")
            
            st.markdown("### Rate Photo")
            r = st.slider("Your rating", 1, 5, key=f"rate_{entry_id}")
            if st.button("Submit", key="sub_rate"):
                self.album_manager.rate_entry(entry_id, r)
                st.rerun()
            
            st.markdown("### Comments")
            comments = self.album_manager.db.get_entry_comments(entry_id)
            for c in comments[:5]:
                st.markdown(f"<small><b>{c['username']}</b>: {c['content']}</small>", unsafe_allow_html=True)
                st.markdown("---")
            
            txt = st.text_area("Add Comment", key="comm_txt")
            if st.button("Post Comment"):
                if txt:
                    self.album_manager.add_comment_to_entry(entry_id, txt)
                    st.rerun()

    def render_gallery(self, person_id: str):
        with sqlite3.connect(self.album_manager.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            # Clean SQL
            sql = """
            SELECT ae.*, p.display_name, i.thumbnail_path, i.filepath
            FROM album_entries ae
            JOIN people p ON ae.person_id = p.person_id
            JOIN images i ON ae.image_id = i.image_id
            WHERE ae.person_id = ?
            ORDER BY ae.created_at DESC
            """.strip()
            
            c.execute(sql, (person_id,))
            entries = [dict(row) for row in c.fetchall()]
        
        if not entries:
            st.info("No photos found. Scan the directory to add photos.")
            return
        
        # Responsive Grid (4 columns)
        cols = st.columns(4)
        for idx, entry in enumerate(entries):
            with cols[idx % 4]:
                # Handle thumbnail logic
                thumb_path = entry.get('thumbnail_path')
                if thumb_path and Path(thumb_path).exists():
                    img_url = self.album_manager.image_processor.get_image_data_url(Path(thumb_path))
                else:
                    # Fallback to full image resized
                    img_url = self.album_manager.image_processor.get_image_data_url(Config.DATA_DIR / entry['filepath'], max_size=Config.THUMBNAIL_SIZE)
                
                # Get stats
                avg, count = self.album_manager.db.get_entry_ratings(entry['entry_id'])
                
                # Card HTML
                card = UIComponents.image_card(
                    img_url, entry.get('caption', ''), count, avg, entry['entry_id']
                )
                st.markdown(card, unsafe_allow_html=True)
                
                # Action
                if st.button("View", key=f"view_{entry['entry_id']}"):
                    st.session_state.selected_image = entry['entry_id']
                    st.rerun()

    def main(self):
        # Main Tabs
        tab1, tab2, tab3 = st.tabs(["📋 Table of Contents", "👤 Profile / Favorites", "⚙️ Settings"])
        
        with tab1:
            people = self.album_manager.db.get_all_people()
            if not people:
                st.info("No people found. Add folders to the 'data' directory and scan.")
            
            # TOC List
            for person in people:
                if st.button(f"📂 {person['display_name']}", key=f"toc_{person['person_id']}"):
                    st.session_state.selected_person = person['person_id']
                    st.rerun()
            
            # Render Gallery if person selected
            if st.session_state.get('selected_person'):
                st.markdown("---")
                self.render_gallery(st.session_state.selected_person)
            else:
                st.image("https://via.placeholder.com/800x300/667eea/ffffff?text=Select+a+Person+to+Start", use_container_width=True)
        
        with tab2:
            st.subheader("Your Favorites")
            favs = self.album_manager.get_user_favorites()
            if not favs:
                st.info("No favorites yet.")
            else:
                cols = st.columns(3)
                for idx, f in enumerate(favs):
                    with cols[idx % 3]:
                        st.caption(f"**{f.get('caption', 'No Title')}**")
                        st.button("View", key=f"fav_view_{f['entry_id']}")
        
        with tab3:
            st.write("Settings coming soon.")
        
        # Image Detail Overlay
        if st.session_state.get('selected_image'):
            st.markdown("---")
            self.render_image_detail(st.session_state.selected_image)

if __name__ == "__main__":
    Config.init_directories()
    app = PhotoAlbumApp()
    app.render_sidebar()
    app.main()
