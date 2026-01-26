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
    VERSION = "2.2.0"  # Upgraded Version
    
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
    THUMBNAIL_SIZE = (300, 300) # Good for grid
    PREVIEW_SIZE = (1200, 1200) # Larger for capable viewing
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB Limit
    
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
# DATABASE MANAGEMENT
# ============================================================================
class DatabaseManager:
    """Manage SQLite database operations"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Config.DB_FILE
        self._init_database()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database with tables"""
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
    
    def add_image(self, metadata: ImageMetadata, thumbnail_path: str = None):
        """Add image metadata to database"""
        if not isinstance(metadata, ImageMetadata):
            raise TypeError("metadata must be an ImageMetadata instance")
            
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
    
    def get_image(self, image_id: str) -> Optional[Dict]:
        """Retrieve image metadata"""
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM images WHERE image_id = ?', (image_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def add_person(self, person: PersonProfile):
        """Add person to database"""
        if not isinstance(person, PersonProfile):
            raise TypeError("person must be a PersonProfile instance")
            
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
    
    def get_all_people(self) -> List[Dict]:
        """Get all people from database"""
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM people ORDER BY display_name')
            return [dict(row) for row in cursor.fetchall()]
    
    def add_album_entry(self, entry: AlbumEntry):
        """Add album entry to database"""
        if not isinstance(entry, AlbumEntry):
            raise TypeError("entry must be an AlbumEntry instance")
            
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
    
    def add_comment(self, comment: Comment):
        """Add comment to database"""
        if not isinstance(comment, Comment):
            raise TypeError("comment must be a Comment instance")
            
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
    
    def add_rating(self, rating: Rating):
        """Add or update rating"""
        if not isinstance(rating, Rating):
            raise TypeError("rating must be a Rating instance")
            
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
    
    def get_entry_comments(self, entry_id: str) -> List[Dict]:
        """Get comments for an entry"""
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
            SELECT * FROM comments
            WHERE entry_id = ?
            ORDER BY created_at DESC
            ''', (entry_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_entry_ratings(self, entry_id: str) -> Tuple[float, int]:
        """Get average rating and count for an entry"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT AVG(rating_value), COUNT(*)
            FROM ratings
            WHERE entry_id = ?
            ''', (entry_id,))
            result = cursor.fetchone()
            return (result[0] or 0, result[1] or 0)
    
    def search_entries(self, query: str, person_id: str = None) -> List[Dict]:
        """Search album entries"""
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
    
    def get_entry_details(self, entry_id: str) -> Optional[Dict]:
        """Get detailed information for an entry"""
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
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
                    result['tags'] = result['tags'].split(',')
                else:
                    result['tags'] = []
                return result
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
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Create thumbnail with high-quality resampling
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                
                # Save thumbnail as JPG for consistency
                img.save(thumbnail_path, 'JPEG', quality=90, optimize=True)
                
            return thumbnail_path
        except Exception as e:
            st.error(f"Error creating thumbnail for {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_image_data_url(image_path: Path, max_size: Tuple[int, int] = None) -> str:
        """Convert image to data URL for embedding with optional resizing"""
        try:
            if not image_path.exists():
                return ""
                
            # Optionally resize before encoding for performance
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            
            if max_size:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Handle format conversion for embedding
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            buffered = io.BytesIO()
            # Use high quality for detail view
            quality = 85 if max_size else 95 
            img.save(buffered, format="JPEG", quality=quality, optimize=True)
            
            encoded = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{encoded}"
            
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
    
    @staticmethod
    def image_card(image_url: str, caption: str, tags: List[str],
                  rating: float, comments_count: int,
                  entry_id: str, metadata: Dict = None) -> str:
        """Generate HTML for image card with enhanced styling"""
        tags_html = UIComponents.tag_badges(tags[:3])
        stars_html = UIComponents.rating_stars(rating)
        
        # Format date if available
        date_str = ""
        if metadata and 'date_taken' in metadata and metadata['date_taken']:
            date_obj = metadata['date_taken']
            if isinstance(date_obj, str):
                try:
                    date_obj = datetime.datetime.fromisoformat(date_obj)
                except:
                    date_obj = None
            if date_obj:
                date_str = f"""
                <div style="
                    color: #666;
                    font-size: 12px;
                    margin-top: 4px;
                ">
                📅  {date_obj.strftime('%Y-%m-%d')}
                </div>
                """
        
        # Fallback image if URL is empty
        if not image_url:
            image_url = "https://via.placeholder.com/300x300/667eea/ffffff?text=No+Image"
        
        return f'''
        <div style="
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            border: 1px solid #f0f0f0;
        ">
            <div style="
                aspect-ratio: 1/1;
                overflow: hidden;
                background: #f8f9fa;
                position: relative;
            ">
                <img src="{image_url}"
                style="
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    transition: transform 0.5s ease;
                "
                onmouseover="this.style.transform='scale(1.05)'"
                onmouseout="this.style.transform='scale(1)'"
                alt="{caption}"
                loading="lazy"
                >
            </div>
            <div style="padding: 16px; flex-grow: 1; display: flex; flex-direction: column;">
                <div style="
                    font-weight: 600;
                    font-size: 15px;
                    margin-bottom: 6px;
                    color: #333;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                ">
                {caption}
                </div>
                {date_str}
                <div style="margin: 8px 0; flex-grow: 1;">
                {stars_html}
                </div>
                <div style="margin: 8px 0; min-height: 36px;">
                {tags_html}
                </div>
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-top: 12px;
                    color: #888;
                    font-size: 13px;
                    border-top: 1px solid #f0f0f0;
                    padding-top: 10px;
                ">
                    <span style="display:flex; align-items:center; gap:4px;">💬 {comments_count}</span>
                    <span style="display:flex; align-items:center; gap:4px;">👁️ {metadata.get('view_count', 0) if metadata else 0}</span>
                </div>
            </div>
        </div>
        '''

# ============================================================================
# ALBUM MANAGER
# ============================================================================
class AlbumManager:
    """Main album management class"""
    
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
                'recent_searches': []
            })
    
    def scan_directory(self, data_dir: Path = None) -> Dict:
        """
        Scan directory for images and update database.
        FIXED: Now accepts folder names with or without hyphens.
        """
        data_dir = data_dir or Config.DATA_DIR
        results = {
            'total_images': 0,
            'new_images': 0,
            'updated_images': 0,
            'people_found': 0,
            'errors': []
        }
        
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            st.warning(f"Created directory: {data_dir}")
            return results
        
        # Find all person directories
        # FIXED: Removed strict requirement for '-' in folder name
        person_dirs = [d for d in data_dir.iterdir() 
                      if d.is_dir() and not d.name.startswith('.')]
        
        results['people_found'] = len(person_dirs)
        
        if not person_dirs:
            st.info("No folders found in the data directory. Please create a folder (e.g., 'My Family') and add images inside it.")
            return results

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
            st.warning("Found folders, but they contain no supported image files.")
            return results

        for person_dir in person_dirs:
            # FIXED: Use the folder name directly as display name if no hyphen exists
            display_name = person_dir.name.replace('-', ' ').replace('_', ' ').title()
            
            # Check if person already exists to avoid changing UUIDs constantly (which breaks links)
            # This logic ensures we keep the same person_id if they exist
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT person_id FROM people WHERE folder_name = ?', (person_dir.name,))
                existing_person = cursor.fetchone()
            
            if existing_person:
                person_id = existing_person[0]
            else:
                person_id = str(uuid.uuid4())
            
            # Create or update person profile
            person_profile = PersonProfile(
                person_id=person_id, # Use the ID found or generated
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
                        # Check if an album entry exists for this person/image combo
                        # If not, create one (in case of moving images)
                        with sqlite3.connect(self.db.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                'SELECT entry_id FROM album_entries WHERE image_id = ? AND person_id = ?',
                                (existing[0], person_id)
                            )
                            if not cursor.fetchone():
                                # Re-link existing image to this person
                                album_entry = AlbumEntry(
                                    entry_id=str(uuid.uuid4()),
                                    image_id=existing[0],
                                    person_id=person_id,
                                    caption=img_path.stem.replace('_', ' ').title(),
                                    description=f"Photo of {display_name}",
                                    location="",
                                    date_taken=ImageMetadata.from_image(img_path).created_date,
                                    tags=[display_name.lower().replace(' ', '-'), 'photo', 'memory'],
                                    privacy_level='public',
                                    created_by='system',
                                    created_at=datetime.datetime.now(),
                                    updated_at=datetime.datetime.now()
                                )
                                self.db.add_album_entry(album_entry)
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
                    # Only print to console to avoid spamming UI, but log to results
                    print(error_msg)
        
        progress_bar.empty()
        
        # Clear cache after scanning
        self.cache.clear()
        
        return results
    
    def _format_name(self, folder_name: str) -> str:
        """Format folder name to display name"""
        parts = folder_name.split('-')
        return ' '.join(part.capitalize() for part in parts)
    
    def get_person_stats(self, person_id: str) -> Dict:
        """Get statistics for a person"""
        cache_key = f"person_stats_{person_id}"
        
        def generate_stats():
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
        
        return self.cache.get_or_set(cache_key, generate_stats)
    
    def get_all_tags(self) -> List[str]:
        """Get all unique tags from database"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT DISTINCT tag
            FROM (
                SELECT TRIM(value) as tag
                FROM album_entries, json_each('["' || REPLACE(tags, ',', '","') || '"]')
                WHERE tags IS NOT NULL AND tags != ''
            )
            WHERE tag != ''
            ORDER BY tag
            ''')
            return [row[0] for row in cursor.fetchall()]
    
    def add_comment_to_entry(self, entry_id: str, content: str,
                           username: str = None, parent_id: str = None):
        """Add comment to an entry"""
        if not username:
            username = st.session_state.get('username', 'Anonymous')
            
        if len(content) > Config.MAX_COMMENT_LENGTH:
            raise ValueError(f"Comment exceeds {Config.MAX_COMMENT_LENGTH} characters")
        
        comment = Comment(
            comment_id=str(uuid.uuid4()),
            entry_id=entry_id,
            user_id=st.session_state['user_id'],
            username=username,
            content=content,
            created_at=datetime.datetime.now(),
            is_edited=False,
            parent_comment_id=parent_id
        )
        self.db.add_comment(comment)
        
        # Clear comments cache for this entry
        cache_key = f"comments_{entry_id}"
        self.cache.clear(cache_key)
        
        return comment
    
    def rate_entry(self, entry_id: str, rating_value: int):
        """Rate an entry"""
        if not 1 <= rating_value <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        rating = Rating(
            rating_id=str(uuid.uuid4()),
            entry_id=entry_id,
            user_id=st.session_state['user_id'],
            rating_value=rating_value,
            created_at=datetime.datetime.now(),
            updated_at=datetime.datetime.now()
        )
        self.db.add_rating(rating)
        
        # Clear ratings cache for this entry
        cache_key = f"ratings_{entry_id}"
        self.cache.clear(cache_key)
        
        return rating
    
    def add_to_favorites(self, entry_id: str):
        """Add entry to user favorites"""
        user_id = st.session_state['user_id']
        
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
    
    def remove_from_favorites(self, entry_id: str):
        """Remove entry from user favorites"""
        user_id = st.session_state['user_id']
        
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
    
    def get_user_favorites(self) -> List[Dict]:
        """Get user's favorite entries"""
        user_id = st.session_state['user_id']
        
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
    
    def export_album_data(self, format_type: str = 'json') -> io.BytesIO:
        """Export album data in specified format"""
        # Get all data
        with sqlite3.connect(self.db.db_path) as conn:
            # Get people
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
            SELECT
                p.*,
                COUNT(DISTINCT ae.entry_id) as photo_count,
                COUNT(DISTINCT c.comment_id) as comment_count
            FROM people p
            LEFT JOIN album_entries ae ON p.person_id = ae.person_id
            LEFT JOIN comments c ON ae.entry_id = c.entry_id
            GROUP BY p.person_id
            ''')
            people_data = [dict(row) for row in cursor.fetchall()]
            
            # Get album entries
            cursor.execute('''
            SELECT ae.*, p.display_name, i.filename
            FROM album_entries ae
            JOIN people p ON ae.person_id = p.person_id
            JOIN images i ON ae.image_id = i.image_id
            ''')
            entries_data = [dict(row) for row in cursor.fetchall()]
        
        if format_type == 'json':
            export_data = {
                'people': people_data,
                'entries': entries_data,
                'export_date': datetime.datetime.now().isoformat(),
                'total_people': len(people_data),
                'total_entries': len(entries_data)
            }
            output = io.BytesIO()
            output.write(json.dumps(export_data, indent=2, default=str).encode('utf-8'))
            output.seek(0)
        elif format_type == 'csv':
            output = io.BytesIO()
            
            # Write people CSV
            if people_data:
                people_df = pd.DataFrame(people_data)
                people_csv = people_df.to_csv(index=False)
                output.write(b"PEOPLE DATA\n")
                output.write(people_csv.encode('utf-8'))
                output.write(b"\n\n")
            
            # Write entries CSV
            if entries_data:
                entries_df = pd.DataFrame(entries_data)
                entries_csv = entries_df.to_csv(index=False)
                output.write(b"ALBUM ENTRIES\n")
                output.write(entries_csv.encode('utf-8'))
            
            output.seek(0)
        
        return output

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PhotoAlbumApp:
    """Main application class"""
    
    def __init__(self):
        self.album_manager = AlbumManager()
        self.setup_page_config()
        self.load_custom_css()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=f"{Config.APP_NAME} v{Config.VERSION}",
            page_icon="👨‍👩‍👧‍👦",
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
        /* Table of Contents */
        .toc-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        .toc-item {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            margin: 8px 0;
            background: white;
            border-radius: 10px;
            transition: all 0.3s ease;
            cursor: pointer;
            border-left: 4px solid transparent;
        }
        .toc-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left-color: #667eea;
        }
        .toc-item.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .toc-person-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 12px;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .toc-person-name {
            font-weight: 600;
            font-size: 16px;
            flex-grow: 1;
        }
        .toc-person-stats {
            display: flex;
            gap: 10px;
            font-size: 12px;
            color: #666;
        }
        .toc-person-stats span {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        /* Image Grid */
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }
        /* Comment Section */
        .comment-container {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        .comment-item {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        .comment-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .comment-author {
            font-weight: 600;
            color: #2d3748;
        }
        .comment-date {
            font-size: 12px;
            color: #666;
        }
        .comment-content {
            color: #4a5568;
            line-height: 1.5;
        }
        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid #eaeaea;
            transition: transform 0.3s ease;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #667eea;
            margin: 10px 0;
        }
        .stat-label {
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        /* Search Box */
        .search-container {
            background: white;
            border-radius: 50px;
            padding: 10px 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
            display: flex;
            align-items: center;
        }
        .search-icon {
            color: #667eea;
            font-size: 20px;
            margin-right: 10px;
        }
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            .image-grid {
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 1rem;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 480px) {
            .image-grid {
                grid-template-columns: 1fr;
            }
            .main {
                padding: 1rem;
            }
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
            background: #667eea;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #764ba2;
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
        /* Image Detail Container */
        .detail-image-container {
            width: 100%;
            text-align: center;
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render main application header"""
        st.markdown(f"""
        <div class="main-header">
            <h1>👨‍👩‍👧‍👦 {Config.APP_NAME}</h1>
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
        """Render application sidebar"""
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
                
                new_username = st.text_input("Username", value=st.session_state.username)
                if new_username != st.session_state.username:
                    st.session_state.username = new_username
                
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
            with sqlite3.connect(self.album_manager.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM people')
                people_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM album_entries')
                entries_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM comments')
                comments_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT AVG(rating_value) FROM ratings')
                avg_rating = cursor.fetchone()[0] or 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("👤 People", people_count)
                st.metric("💬 Comments", comments_count)
            with col2:
                st.metric("🖼️ Photos", entries_count)
                st.metric("⭐ Avg Rating", f"{avg_rating:.1f}")
            
            # Actions
            st.markdown("---")
            st.markdown("## ⚡ Quick Actions")
            if st.button("📸 Scan for New Photos", use_container_width=True, key="scan_button"):
                with st.spinner("Scanning directory..."):
                    results = self.album_manager.scan_directory()
                
                success_message = f"""
                Scan complete!
                - New images: {results['new_images']}
                - Updated: {results['updated_images']}
                - Total: {results['total_images']}
                - People: {results['people_found']}
                """
                
                if results['errors']:
                    success_message += f"\n\n⚠️ {len(results['errors'])} errors occurred:"
                    for error in results['errors'][:3]:  # Show first 3 errors
                        success_message += f"\n- {error}"
                    if len(results['errors']) > 3:
                        success_message += f"\n- and {len(results['errors'])-3} more..."
                
                st.success(success_message)
            
            if st.button("🗂️ Clear Cache", use_container_width=True, key="clear_cache_button"):
                self.album_manager.cache.clear()
                st.success("Cache cleared!")
            
            # Export options
            st.markdown("---")
            st.markdown("## 📤 Export")
            export_format = st.selectbox("Format", ["json", "csv"], key="export_format")
            if st.button("💾 Export Album Data", use_container_width=True, key="export_button"):
                with st.spinner("Preparing export..."):
                    try:
                        export_data = self.album_manager.export_album_data(export_format)
                        export_name = f"album_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
                        
                        st.download_button(
                            label=f"Download {export_format.upper()}",
                            data=export_data,
                            file_name=export_name,
                            mime="application/json" if export_format == "json" else "text/csv",
                            use_container_width=True,
                            key="download_export"
                        )
                    except Exception as e:
                        st.error(f"Error during export: {str(e)}")
    
    def render_table_of_contents(self):
        """Render interactive table of contents"""
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Table of Contents</div>', unsafe_allow_html=True)
        
        # Search and filter
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            search_query = st.text_input("🔍 Search people...", 
                                       placeholder="Type to filter...",
                                       key="toc_search")
            if search_query and search_query not in st.session_state.recent_searches:
                st.session_state.recent_searches.append(search_query)
                if len(st.session_state.recent_searches) > 5:
                    st.session_state.recent_searches.pop(0)
        
        with col2:
            sort_option = st.selectbox("Sort by", ["Name", "Photo Count", "Recent Activity"], 
                                     key="toc_sort")
        
        with col3:
            items_per_page = st.slider("Items per page", 5, 50, 10, key="toc_items_per_page")
        
        # Get all people
        all_people = self.album_manager.db.get_all_people()
        
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
        
        # Pagination
        total_pages = max(1, (len(entries) + items_per_page - 1) // items_per_page)
        page_number = st.session_state.get('gallery_page', 1)
        start_idx = (page_number - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(entries))
        current_entries = entries[start_idx:end_idx]
        
        # Display images based on view mode
        if view_mode == "Grid":
            self._render_grid_view(current_entries, person_info)
        else:  # List view
            self._render_list_view(current_entries, person_info)
        
        # Pagination controls
        if total_pages > 1:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⬅️ Previous Page", key="prev_page_btn", disabled=page_number <= 1):
                    st.session_state.gallery_page = page_number - 1
                    st.rerun()
            with col2:
                st.markdown(f"**Page {page_number} of {total_pages}**")
            with col3:
                if st.button("Next Page ➡️", key="next_page_btn", disabled=page_number >= total_pages):
                    st.session_state.gallery_page = page_number + 1
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_grid_view(self, entries: List[Dict], person_info: Dict):
        """Render images in grid view using HTML components for better styling"""
        if not entries:
            st.info("No photos found for this person.")
            return

        # Calculate columns based on screen width logic (Streamlit handles this, but we limit to 4 for consistency)
        cols = st.columns(4)
        
        for idx, entry in enumerate(entries):
            with cols[idx % 4]:
                # Get thumbnail path or create data URL
                thumbnail_path = entry.get('thumbnail_path')
                img_url = ""
                
                if thumbnail_path and Path(thumbnail_path).exists():
                    # Use thumbnail for grid
                    img_url = self.album_manager.image_processor.get_image_data_url(Path(thumbnail_path))
                else:
                    # Fallback to actual image
                    img_path = Config.DATA_DIR / entry['filepath']
                    if img_path.exists():
                        # Resize on fly for grid if no thumb exists
                        img_url = self.album_manager.image_processor.get_image_data_url(img_path, max_size=Config.THUMBNAIL_SIZE)
                
                # Get rating and comment count
                avg_rating, rating_count = self.album_manager.db.get_entry_ratings(entry['entry_id'])
                
                with sqlite3.connect(self.album_manager.db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'SELECT COUNT(*) FROM comments WHERE entry_id = ?',
                        (entry['entry_id'],)
                    )
                    comment_count = cursor.fetchone()[0] or 0
                
                # Parse tags
                tags = entry.get('tags', '').split(',') if entry.get('tags') else []
                
                # Generate HTML Card
                card_html = UIComponents.image_card(
                    image_url=img_url,
                    caption=entry.get('caption', 'Untitled'),
                    tags=tags,
                    rating=avg_rating,
                    comments_count=comment_count,
                    entry_id=entry['entry_id'],
                    metadata={'date_taken': entry.get('date_taken'), 'view_count': 0}
                )
                
                # Render HTML Card
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Render Action Buttons separately to ensure they work
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("👁️", key=f"view_{entry['entry_id']}", use_container_width=True, help="View Details"):
                        st.session_state.selected_image = entry['entry_id']
                        st.rerun()
                with col_btn2:
                    is_favorite = entry['entry_id'] in st.session_state.get('favorites', set())
                    fav_icon = "❤️" if is_favorite else "🤍"
                    if st.button(fav_icon, key=f"favorite_{entry['entry_id']}", use_container_width=True, help="Toggle Favorite"):
                        if is_favorite:
                            self.album_manager.remove_from_favorites(entry['entry_id'])
                        else:
                            self.album_manager.add_to_favorites(entry['entry_id'])
                        st.rerun()
    
    def _render_list_view(self, entries: List[Dict], person_info: Dict):
        """Render images in list view"""
        for entry in entries:
            with st.container():
                col_img, col_info = st.columns([1, 2])
                
                with col_img:
                    # Display thumbnail
                    thumbnail_path = entry.get('thumbnail_path')
                    img_url = ""
                    
                    if thumbnail_path and Path(thumbnail_path).exists():
                        img_url = self.album_manager.image_processor.get_image_data_url(Path(thumbnail_path))
                    else:
                        img_path = Config.DATA_DIR / entry['filepath']
                        img_url = self.album_manager.image_processor.get_image_data_url(img_path, max_size=Config.THUMBNAIL_SIZE) if img_path.exists() else ""
                    
                    if img_url:
                        st.markdown(f'<img src="{img_url}" style="width:100%; border-radius:8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
                    else:
                        st.markdown("No image available")
                
                with col_info:
                    # Entry details
                    avg_rating, rating_count = self.album_manager.db.get_entry_ratings(entry['entry_id'])
                    
                    with sqlite3.connect(self.album_manager.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            'SELECT COUNT(*) FROM comments WHERE entry_id = ?',
                            (entry['entry_id'],)
                        )
                        comment_count = cursor.fetchone()[0] or 0
                    
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <h4 style="margin: 0; color: #2d3748;">{entry.get('caption', 'Untitled')}</h4>
                        <div style="color: #666; font-size: 14px; margin: 5px 0;">
                            {entry.get('description', 'No description')}
                        </div>
                        <div style="margin: 10px 0;">
                            {UIComponents.rating_stars(avg_rating)}
                        </div>
                        <div style="display: flex; gap: 15px; color: #666; font-size: 12px;">
                            <span>💬 {comment_count} comments</span>
                            <span>⭐ {rating_count} ratings</span>
                        </div>
                        <div style="margin-top: 10px;">
                            {UIComponents.tag_badges(entry.get('tags', '').split(',') if entry.get('tags') else [])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Action buttons
                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
                with col_btn1:
                    if st.button("👁️ View", key=f"view_list_{entry['entry_id']}", use_container_width=True):
                        st.session_state.selected_image = entry['entry_id']
                        st.rerun()
                with col_btn2:
                    if st.button("💬 Comment", key=f"comment_list_{entry['entry_id']}", use_container_width=True):
                        st.session_state.comment_for_entry = entry['entry_id']
                        st.rerun()
                with col_btn3:
                    if st.button("⭐ Rate", key=f"rate_list_{entry['entry_id']}", use_container_width=True):
                        st.session_state.rate_for_entry = entry['entry_id']
                        st.rerun()
                with col_btn4:
                    is_favorite = entry['entry_id'] in st.session_state.get('favorites', set())
                    if st.button(
                        f"{'❤️' if is_favorite else '🤍'}", 
                        key=f"favorite_list_{entry['entry_id']}",
                        use_container_width=True
                    ):
                        if is_favorite:
                            self.album_manager.remove_from_favorites(entry['entry_id'])
                        else:
                            self.album_manager.add_to_favorites(entry['entry_id'])
                        st.rerun()
                
                st.markdown("---")
    
    def render_image_detail(self, entry_id: str):
        """Render detailed view for a single image with high-res capability"""
        entry_details = self.album_manager.db.get_entry_details(entry_id)
        if not entry_details:
            st.error("Image not found")
            return
        
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        
        # Back button
        if st.button("⬅️ Back to Gallery", key="back_button"):
            st.session_state.selected_image = None
            st.rerun()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image using HTML for responsive, high-quality rendering
            img_path = Config.DATA_DIR / entry_details['filepath']
            
            if img_path.exists():
                # Use the image processor to get a base64 string, but do NOT resize it Python-side
                # This allows the browser to handle the full resolution, enabling zooming capability
                st.markdown('<div class="detail-image-container">', unsafe_allow_html=True)
                
                try:
                    full_img_url = self.album_manager.image_processor.get_image_data_url(img_path, max_size=None)
                    
                    # HTML Image with responsive width and auto height
                    st.markdown(f'''
                    <img src="{full_img_url}" 
                         alt="{entry_details.get('caption', 'Image')}" 
                         style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <p style="text-align: center; color: #888; font-size: 12px; margin-top: 5px;">
                        Click image to open in new tab for full resolution
                    </p>
                    ''', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                    st.image("https://via.placeholder.com/800x600/667eea/ffffff?text=Image+Error")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Image info
            st.markdown(f"""
            <div style="margin-top: 1.5rem;">
                <h2>{entry_details.get('caption', 'Untitled')}</h2>
                <p style="color: #666; font-size: 16px; line-height: 1.6;">{entry_details.get('description', '')}</p>
                <div style="display: flex; gap: 20px; color: #555; font-size: 14px; margin: 15px 0; flex-wrap: wrap;">
                    <div><strong>📍 Location:</strong> {entry_details.get('location', 'Unknown')}</div>
                    <div><strong>📅 Date:</strong> {entry_details.get('date_taken', 'Unknown')[:10] if entry_details.get('date_taken') else 'Unknown'}</div>
                    <div><strong>💾 Size:</strong> {entry_details.get('file_size', 0) / 1024:.1f} KB</div>
                    <div><strong>📐 Dimensions:</strong> {entry_details.get('dimensions', 'Unknown')}</div>
                </div>
                <div style="margin: 15px 0;">
                    <div style="font-weight: 600; margin-bottom: 5px;">Tags:</div>
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
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid #eaeaea;">
                <div style="margin-bottom: 12px;">
                    <strong>Rating:</strong><br>
                    <div style="margin: 5px 0;">{UIComponents.rating_stars(avg_rating, size=24)}</div>
                    <div style="color: #666; font-size: 14px;">
                        Based on {rating_count} ratings
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
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
                    st.success("Rating submitted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            # Add to favorites
            is_favorite = entry_id in st.session_state.get('favorites', set())
            favorite_button = st.button(
                f"{'❤️ Remove from Favorites' if is_favorite else '🤍 Add to Favorites'}",
                use_container_width=True,
                key="favorite_button"
            )
            if favorite_button:
                if is_favorite:
                    self.album_manager.remove_from_favorites(entry_id)
                    st.success("Removed from favorites!")
                else:
                    self.album_manager.add_to_favorites(entry_id)
                    st.success("Added to favorites!")
                st.rerun()
        
        # Comments section
        st.markdown("---")
        st.markdown("### 💬 Comments")
        
        # Display existing comments
        if comments:
            for comment in comments[:10]:  # Limit to 10 comments initially
                with st.container():
                    st.markdown(f"""
                    <div class="comment-item">
                        <div class="comment-header">
                            <span class="comment-author">{comment['username']}</span>
                            <span class="comment-date">{comment.get('created_at', '')[:19]}</span>
                        </div>
                        <div class="comment-content">{comment['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No comments yet. Be the first to comment!")
        
        # Add comment form
        st.markdown("### ✏️ Add a Comment")
        comment_text = st.text_area(
            "Your comment",
            placeholder="Share your thoughts...",
            height=100,
            max_chars=Config.MAX_COMMENT_LENGTH,
            key=f"comment_text_{entry_id}"
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Post Comment", use_container_width=True, type="primary", key="post_comment"):
                if comment_text.strip():
                    try:
                        self.album_manager.add_comment_to_entry(
                            entry_id,
                            comment_text,
                            st.session_state.username
                        )
                        st.success("Comment posted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a comment")
        with col2:
            if st.button("Cancel", use_container_width=True, key="cancel_comment"):
                st.rerun()
        
        # EXIF Data (collapsible)
        with st.expander("📸 EXIF Metadata", expanded=False):
            if entry_details.get('exif_data'):
                exif_data = entry_details['exif_data']
                if isinstance(exif_data, str):
                    try:
                        exif_data = json.loads(exif_data)
                    except:
                        exif_data = {}
                
                if exif_data:
                    for key, value in exif_data.items():
                        if value and str(value).strip():  # Skip None/empty values
                            st.text(f"{key}: {value}")
                else:
                    st.info("No EXIF data available")
            else:
                st.info("No EXIF data available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_search(self):
        """Render global search functionality"""
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Search Album</div>', unsafe_allow_html=True)
        
        # Search box
        col1, col2 = st.columns([4, 1])
        with col1:
            search_query = st.text_input(
                "Search for photos, people, or tags...",
                placeholder="Enter keywords...",
                key="global_search"
            )
        with col2:
            search_button = st.button("Search", use_container_width=True, type="primary", key="search_button")
        
        if search_query:
            if search_query not in st.session_state.recent_searches:
                st.session_state.recent_searches.append(search_query)
                if len(st.session_state.recent_searches) > 5:
                    st.session_state.recent_searches.pop(0)
        
        if search_button and search_query:
            with st.spinner("Searching..."):
                results = self.album_manager.db.search_entries(search_query)
            
            if results:
                st.success(f"Found {len(results)} results")
                
                # Display results
                for result in results:
                    with st.expander(f"{result['caption']} - {result['display_name']}"):
                        col_img, col_info = st.columns([1, 2])
                        with col_img:
                            # Try to display thumbnail
                            img_path = Config.DATA_DIR / result.get('filepath', '')
                            if img_path.exists():
                                st.image(
                                    self.album_manager.image_processor.get_image_data_url(img_path, max_size=Config.THUMBNAIL_SIZE),
                                    width=150
                                )
                        with col_info:
                            st.write(f"**Caption:** {result['caption']}")
                            st.write(f"**Description:** {result.get('description', 'No description')}")
                            st.write(f"**Person:** {result['display_name']}")
                            st.write(f"**Date:** {result.get('date_taken', 'Unknown')[:10] if result.get('date_taken') else 'Unknown'}")
                        
                        if st.button("View Details", key=f"view_search_{result['entry_id']}"):
                            st.session_state.selected_image = result['entry_id']
                            st.rerun()
            else:
                st.warning("No results found")
        
        # Recent searches
        if st.session_state.recent_searches:
            st.markdown("---")
            st.markdown("### 🔍 Recent Searches")
            cols = st.columns(3)
            for idx, search in enumerate(st.session_state.recent_searches[-6:]):
                with cols[idx % 3]:
                    if st.button(search, use_container_width=True, key=f"recent_search_{idx}"):
                        st.session_state.global_search = search
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render main dashboard with overview"""
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Dashboard Overview</div>', unsafe_allow_html=True)
        
        # Quick stats
        with sqlite3.connect(self.album_manager.db.db_path) as conn:
            cursor = conn.cursor()
            # Get various statistics
            cursor.execute('SELECT COUNT(*) FROM people')
            total_people = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM album_entries')
            total_photos = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM comments')
            total_comments = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(rating_value) FROM ratings')
            avg_rating = cursor.fetchone()[0] or 0
            
            cursor.execute('''
            SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count
            FROM album_entries
            GROUP BY month
            ORDER BY month DESC
            LIMIT 6
            ''')
            monthly_stats = cursor.fetchall()
        
        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👤 People", total_people)
        with col2:
            st.metric("🖼️ Photos", total_photos)
        with col3:
            st.metric("💬 Comments", total_comments)
        with col4:
            st.metric("⭐ Avg Rating", f"{avg_rating:.1f}")
        
        # Charts and visualizations
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Photos Per Person")
            # Get photos per person
            with sqlite3.connect(self.album_manager.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT p.display_name, COUNT(ae.entry_id) as photo_count
                FROM people p
                LEFT JOIN album_entries ae ON p.person_id = ae.person_id
                GROUP BY p.person_id
                ORDER BY photo_count DESC
                LIMIT 10
                ''')
                people_stats = cursor.fetchall()
            
            if people_stats:
                df = pd.DataFrame(people_stats, columns=['Person', 'Photos'])
                st.bar_chart(df.set_index('Person'))
        
        with col2:
            st.markdown("### 📅 Recent Activity")
            if monthly_stats:
                chart_data = []
                for month, count in monthly_stats:
                    # Ensure month is in correct format
                    if month:
                        chart_data.append({'Month': month, 'Photos': count})
                
                if chart_data:
                    df = pd.DataFrame(chart_data)
                    df = df.sort_values('Month')
                    st.line_chart(df.set_index('Month'))
                else:
                    st.info("No activity data available")
            else:
                st.info("No activity data available")
        
        # Recent photos
        st.markdown("---")
        st.markdown("### 📸 Recently Added Photos")
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
                    thumbnail_path = photo.get('thumbnail_path')
                    if thumbnail_path and Path(thumbnail_path).exists():
                        img_url = self.album_manager.image_processor.get_image_data_url(Path(thumbnail_path))
                        st.image(img_url, use_container_width=True)
                    
                    st.caption(f"**{photo['caption']}**")
                    st.caption(f"By: {photo['display_name']}")
                    st.caption(f"Added: {photo['created_at'][:10]}")
                    
                    if st.button("View", key=f"view_recent_{photo['entry_id']}", use_container_width=True):
                        st.session_state.selected_image = photo['entry_id']
                        st.rerun()
        else:
            st.info("No photos available yet. Add some photos to get started!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_favorites(self):
        """Render user's favorite photos"""
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">❤️ Your Favorites</div>', unsafe_allow_html=True)
        
        favorites = self.album_manager.get_user_favorites()
        
        if not favorites:
            st.info("You haven't added any photos to favorites yet.")
            st.markdown("""
            To add photos to favorites:
            1. Browse photos in the Table of Contents tab
            2. Click on a photo to view details
            3. Click the "🤍 Add to Favorites" button
            """)
        else:
            st.write(f"You have {len(favorites)} favorite photos")
            
            # Display favorites in a grid
            cols = st.columns(3)
            for idx, entry in enumerate(favorites):
                with cols[idx % 3]:
                    thumbnail_path = entry.get('thumbnail_path')
                    img_url = ""
                    
                    if thumbnail_path and Path(thumbnail_path).exists():
                        img_url = self.album_manager.image_processor.get_image_data_url(Path(thumbnail_path))
                    else:
                        img_path = Config.DATA_DIR / entry['filepath']
                        if img_path.exists():
                            img_url = self.album_manager.image_processor.get_image_data_url(img_path, max_size=Config.THUMBNAIL_SIZE)
                    
                    if img_url:
                        st.image(img_url, use_container_width=True)
                    
                    st.caption(entry.get('caption', 'Untitled'))
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("View", key=f"view_fav_{entry['entry_id']}", use_container_width=True):
                            st.session_state.selected_image = entry['entry_id']
                            st.rerun()
                    with col_btn2:
                        if st.button("Remove", key=f"remove_fav_{entry['entry_id']}", use_container_width=True):
                            self.album_manager.remove_from_favorites(entry['entry_id'])
                            st.success("Removed from favorites!")
                            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_settings(self):
        """Render application settings"""
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ Settings</div>', unsafe_allow_html=True)
        
        # Application settings
        st.markdown("### 🎨 Display Settings")
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], 
                               index=0, key="theme_select")
            default_view = st.selectbox("Default View", ["Grid", "List"], 
                                      index=0, key="default_view_select")
        with col2:
            items_per_page = st.slider("Items per page", 12, 48, 24, key="settings_items_per_page")
            show_exif = st.checkbox("Show EXIF data by default", value=True, key="show_exif_check")
        
        # Data management
        st.markdown("---")
        st.markdown("### 💾 Data Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild Thumbnails", use_container_width=True, key="rebuild_thumbnails"):
                with st.spinner("Rebuilding thumbnails..."):
                    # This would rebuild all thumbnails
                    st.info("Thumbnail rebuild functionality would be implemented here")
        with col2:
            if st.button("⚡ Optimize Database", use_container_width=True, key="optimize_db"):
                with st.spinner("Optimizing database..."):
                    # This would optimize the SQLite database
                    st.info("Database optimization functionality would be implemented here")
        
        # Backup and restore
        st.markdown("---")
        st.markdown("### 💾 Backup & Restore")
        tab1, tab2 = st.tabs(["Backup", "Restore"])
        
        with tab1:
            st.info("Create a backup of your album data")
            backup_format = st.selectbox("Backup format", ["JSON", "CSV"], key="backup_format")
            if st.button("Create Backup", type="primary", key="create_backup"):
                with st.spinner("Creating backup..."):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_data = self.album_manager.export_album_data(
                        "json" if backup_format == "JSON" else "csv"
                    )
                    st.download_button(
                        label=f"Download {backup_format} Backup",
                        data=backup_data,
                        file_name=f"album_backup_{timestamp}.{backup_format.lower()}",
                        mime="application/octet-stream",
                        key="download_backup"
                    )
        
        with tab2:
            st.warning("⚠️ Restoring will replace current data")
            uploaded_file = st.file_uploader("Choose a backup file", 
                                            type=['json', 'csv'], 
                                            key="backup_uploader")
            if uploaded_file and st.button("Restore from Backup", type="secondary", key="restore_backup"):
                st.error("Restore functionality is not implemented in this demo version")
        
        # Danger zone
        st.markdown("---")
        st.markdown("### ⚠️ Danger Zone")
        with st.expander("Advanced Operations", expanded=False):
            st.warning("These operations cannot be undone!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear All Data", type="secondary", use_container_width=True, key="clear_all_data"):
                    if st.checkbox("I understand this will delete all data", key="confirm_delete"):
                        if st.button("Confirm Delete All Data", type="primary", key="confirm_delete_button"):
                            st.error("Data deletion is not implemented in this demo version")
            with col2:
                if st.button("Reset Application", type="secondary", use_container_width=True, key="reset_app"):
                    if st.checkbox("Reset all settings and cache", key="confirm_reset"):
                        if st.button("Confirm Reset", type="primary", key="confirm_reset_button"):
                            for key in list(st.session_state.keys()):
                                del st.session_state[key]
                            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def main(self):
        """Main application entry point"""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Dashboard",
            "📋 Table of Contents",
            "🔍 Search",
            "❤️ Favorites",
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_dashboard()
        
        with tab2:
            self.render_table_of_contents()
            # If a person is selected, show their gallery
            if st.session_state.get('selected_person'):
                self.render_person_gallery(st.session_state.selected_person)
        
        with tab3:
            self.render_search()
        
        with tab4:
            self.render_favorites()
        
        with tab5:
            self.render_settings()
        
        # Handle image selection
        if st.session_state.get('selected_image'):
            self.render_image_detail(st.session_state.selected_image)

# ============================================================================
# APPLICATION INITIALIZATION AND ENTRY POINT
# ============================================================================
def initialize_application():
    """Initialize the application with default data if needed"""
    # Create necessary directories
    Config.init_directories()
    
    return True

def create_sample_data():
    """Create sample data for demonstration purposes"""
    # Create sample person directories
    sample_people = ["john-smith", "sarah-johnson", "michael-brown"]
    
    for person in sample_people:
        person_dir = Config.DATA_DIR / person
        person_dir.mkdir(exist_ok=True)
        
        # Create a sample README file
        readme_file = person_dir / "README.txt"
        readme_file.write_text(f"Photos of {person.replace('-', ' ').title()}\nAdd your photos here!")
    
    st.success(f"Created sample data structure with {len(sample_people)} person folders")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Set up Streamlit page
    st.set_page_config(
        page_title="MemoryVault Pro",
        page_icon="👨‍👩‍👧‍👦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize application
    Config.init_directories()
    
    # Check if we need to create sample data
    if not any(Config.DATA_DIR.iterdir()):
        st.info("No data found. Let's set up your album!")
        if st.button("Create Sample Data Structure"):
            create_sample_data()
            st.rerun()
    
    # Create and run the application
    app = PhotoAlbumApp()
    app.main()
