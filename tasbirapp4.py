"""
COMPREHENSIVE WEB PHOTO & VIDEO ALBUM APPLICATION
Version: 7.1.0 - Fixed Database Index Creation, All Features Included
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ExifTags, ImageDraw
import base64
import json
import datetime
import uuid
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import io
import time
import math
from dataclasses import dataclass, asdict
from enum import Enum
import os
from contextlib import contextmanager
import mimetypes
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# VIDEO SUPPORT (optional)
# ============================================================================
try:
    import cv2
    import moviepy.editor as mp
    from moviepy.editor import VideoFileClip
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False

# ============================================================================
# NUMERIC PASSWORD (8 digits only)
# ============================================================================
_REAL_PASSWORD = "19870505"

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.markdown("""
    <style>
    .login-bg{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
              padding:60px 30px;border-radius:24px;text-align:center;margin:40px 0;}
    .login-title{font-size:2.6em;font-weight:800;
                 background:linear-gradient(90deg,#f9d423,#ff4e50);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    </style>
    <div class="login-bg">
        <div class="login-title">MemoryVault Pro+</div>
        <div>Secure Photo & Video Album</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        pwd = st.text_input("Access Key", type="password", placeholder="8‑digit numeric code")
        if st.button("Unlock", use_container_width=True, type="primary"):
            if pwd.strip().isdigit() and pwd.strip() == _REAL_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid key")
    return False


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    APP_NAME = "MemoryVault Pro+"
    VERSION = "7.1.0"
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    VIDEO_THUMBNAIL_DIR = BASE_DIR / "video_thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    VIDEO_CACHE_DIR = BASE_DIR / "video_cache"
    DB_FILE = DB_DIR / "album.db"
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    HD_SIZE = (1920, 1080)
    THUMB_STRIP_SIZE = (120, 90)
    MAX_VIDEO_SIZE = 100 * 1024 * 1024
    SUPPORTED_VIDEO_FORMATS = ['.mp4','.mov','.avi','.mkv','.webm','.wmv','.flv','.m4v']
    IMAGE_EXTENSIONS = {'.jpg','.jpeg','.png','.gif','.bmp','.webp','.tiff'}
    ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | set(SUPPORTED_VIDEO_FORMATS)
    MAX_COMMENT_LENGTH = 500
    CACHE_TTL = 3600
    FRAME_STYLES = ["Elegant Gold","Polaroid","Modern Shadow","Dark Museum","Vintage","Gallery White"]
    DEFAULT_FRAME = "Elegant Gold"
    ITEMS_PER_PAGE = 20
    GRID_COLUMNS = 4

    @classmethod
    def init_directories(cls):
        for d in [cls.DATA_DIR, cls.THUMBNAIL_DIR, cls.VIDEO_THUMBNAIL_DIR,
                  cls.METADATA_DIR, cls.DB_DIR, cls.EXPORT_DIR, cls.VIDEO_CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_sample_structure()

    @classmethod
    def create_sample_structure(cls):
        sample_people = ["john-smith", "sarah-johnson", "michael-brown"]
        for person in sample_people:
            person_dir = cls.DATA_DIR / person
            person_dir.mkdir(exist_ok=True)
            readme = person_dir / "README.txt"
            readme.write_text(f"Photos/Videos of {person.replace('-', ' ').title()}")
            for i in range(1,4):
                img_path = person_dir / f"sample_{i}.jpg"
                if not img_path.exists():
                    try:
                        img = Image.new('RGB', (600,400), color=['#667eea','#f56565','#48bb78'][i-1])
                        draw = ImageDraw.Draw(img)
                        draw.text((200,180), f"{person.split('-')[0].title()} {i}", fill='#fff')
                        img.save(img_path, 'JPEG')
                    except:
                        pass


# ============================================================================
# FRAME RENDERER
# ============================================================================
class FrameRenderer:
    @staticmethod
    def wrap_detail(src: str, style: str, expanded: bool = False) -> str:
        max_h = "85vh" if expanded else "65vh"
        styles = {
            "Elegant Gold": ('background:linear-gradient(135deg,#b8860b,#daa520,#ffd700);padding:14px;border-radius:10px;',
                             'background:#fffff5;padding:20px;border-radius:6px;'),
            "Polaroid": ('background:#fff;padding:20px 20px 70px 20px;border-radius:3px;',''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:16px;box-shadow:0 14px 48px rgba(0,0,0,.2);',''),
            "Dark Museum": ('background:linear-gradient(160deg,#0d0d1a,#1a1a30);padding:28px;border-radius:16px;',
                            'background:#fffff8;padding:18px;border-radius:6px;'),
            "Vintage": ('background:linear-gradient(135deg,#d4b896,#e8d5b7);padding:16px;border-radius:6px;border:2px solid #a08050;',
                        'background:#faf5ee;padding:14px;border-radius:4px;'),
            "Gallery White": ('background:#fff;padding:24px;border-radius:4px;border:1px solid #e0e0e0;','')
        }
        outer, inner = styles.get(style, styles["Elegant Gold"])
        return f'''<div style="{outer}"><div style="{inner}">
            <img src="{src}" style="width:100%;max-height:{max_h};object-fit:contain;display:block;margin:0 auto;border-radius:4px;">
        </div></div>'''

    @staticmethod
    def wrap_thumb_strip_item(src: str, active: bool = False, is_video: bool = False) -> str:
        active_cls = "active" if active else ""
        play = '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:20px;color:#fff;opacity:.8;">▶</div>' if is_video else ''
        return f'''
        <div class="thumb-item {active_cls}">
            <img src="{src}">{play}
        </div>
        '''

    @staticmethod
    def inject_css():
        st.markdown("""
        <style>
        .nav-container { display: flex; align-items: center; justify-content: center; height: 100%; min-height: 60vh; }
        .nav-btn { background: #667eea; color: white; border: none; border-radius: 50%; width: 56px; height: 56px; font-size: 28px; cursor: pointer; transition: 0.2s; margin: 0 10px; }
        .nav-btn:hover { background: #764ba2; transform: scale(1.05); }
        .nav-btn.disabled { opacity: 0.3; cursor: not-allowed; pointer-events: none; }
        .thumb-strip-wrapper { overflow-x: auto; white-space: nowrap; padding: 10px 0; }
        .thumb-strip { display: flex; gap: 8px; }
        .thumb-item { min-width: 90px; height: 68px; border-radius: 6px; overflow: hidden; cursor: pointer; opacity: 0.5; transition: 0.2s; position: relative; display: inline-block; }
        .thumb-item.active { opacity: 1; transform: scale(1.05); box-shadow: 0 0 0 2px #667eea; }
        .thumb-item img { width: 100%; height: 100%; object-fit: cover; }
        .sidebar-folder { padding: 8px 12px; border-radius: 8px; cursor: pointer; margin-bottom: 2px; transition: background 0.1s; }
        .sidebar-folder:hover { background: rgba(102,126,234,0.2); }
        .sidebar-folder.active { background: rgba(102,126,234,0.3); font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)


# ============================================================================
# DATA MODELS (full original)
# ============================================================================
class UserRoles(Enum):
    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    EDITOR = "editor"
    ADMIN = "admin"

class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"

@dataclass
class MediaMetadata:
    media_id: str
    filename: str
    filepath: str
    file_size: int
    media_type: str
    dimensions: Tuple[int,int]
    format: str
    duration: Optional[float]
    frame_rate: Optional[float]
    created_date: datetime.datetime
    modified_date: datetime.datetime
    exif_data: Optional[Dict]
    checksum: str

    @classmethod
    def from_file(cls, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        media_type = cls._detect_media_type(file_path)
        stats = file_path.stat()
        if media_type == MediaType.IMAGE.value:
            return cls._from_image(file_path, stats, media_type)
        else:
            return cls._from_video(file_path, stats, media_type)

    @staticmethod
    def _detect_media_type(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaType.VIDEO.value
        elif ext in Config.IMAGE_EXTENSIONS:
            return MediaType.IMAGE.value
        raise ValueError(f"Unsupported format: {ext}")

    @classmethod
    def _from_image(cls, image_path: Path, stats, media_type: str):
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
    def _from_video(cls, video_path: Path, stats, media_type: str):
        dimensions, duration, frame_rate = (0,0), 0.0, 0.0
        if VIDEO_SUPPORT:
            try:
                clip = VideoFileClip(str(video_path))
                dimensions, duration, frame_rate = clip.size, clip.duration, clip.fps
                clip.close()
            except:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = fc / fps if fps > 0 else 0
                    frame_rate = fps
                    cap.release()
                except:
                    pass
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
    def _extract_exif(img):
        try:
            exif = {}
            if hasattr(img, '_getexif') and img._getexif():
                raw = img._getexif()
                for tag_id, val in raw.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if not isinstance(val, (bytes, np.ndarray)):
                        exif[tag] = str(val)
            return exif or None
        except:
            return None

    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()


@dataclass
class AlbumEntry:
    entry_id: str
    media_id: str
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

    def to_dict(self):
        d = asdict(self)
        d['date_taken'] = self.date_taken.isoformat() if self.date_taken else None
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d

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

    def to_dict(self):
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        return d

@dataclass
class Rating:
    rating_id: str
    entry_id: str
    user_id: str
    rating_value: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    def to_dict(self):
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d

@dataclass
class PersonProfile:
    person_id: str
    folder_name: str
    display_name: str
    bio: str
    birth_date: Optional[datetime.date]
    relationship: str
    contact_info: str
    social_links: Dict[str,str]
    profile_image: Optional[str]
    created_at: datetime.datetime

    def to_dict(self):
        d = asdict(self)
        d['birth_date'] = self.birth_date.isoformat() if self.birth_date else None
        d['created_at'] = self.created_at.isoformat()
        return d


# ============================================================================
# DATABASE MANAGER (fixed index creation)
# ============================================================================
class DatabaseManager:
    def __init__(self):
        self.db_path = Config.DB_FILE
        self._init_database()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        Config.DB_DIR.mkdir(parents=True, exist_ok=True)
        with self.get_connection() as conn:
            cur = conn.cursor()
            # Create tables
            cur.execute('''CREATE TABLE IF NOT EXISTS media (
                media_id TEXT PRIMARY KEY, filename TEXT NOT NULL, filepath TEXT NOT NULL,
                file_size INTEGER, media_type TEXT NOT NULL, width INTEGER, height INTEGER,
                format TEXT, duration REAL, frame_rate REAL,
                created_date TIMESTAMP, modified_date TIMESTAMP, exif_data TEXT,
                checksum TEXT UNIQUE, thumbnail_path TEXT, video_thumbnail_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            cur.execute('''CREATE TABLE IF NOT EXISTS people (
                person_id TEXT PRIMARY KEY, folder_name TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL, bio TEXT, birth_date DATE,
                relationship TEXT, contact_info TEXT, social_links TEXT,
                profile_image TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            cur.execute('''CREATE TABLE IF NOT EXISTS album_entries (
                entry_id TEXT PRIMARY KEY, media_id TEXT NOT NULL, person_id TEXT NOT NULL,
                caption TEXT, description TEXT, location TEXT, date_taken TIMESTAMP,
                tags TEXT, privacy_level TEXT DEFAULT 'public', created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(media_id) REFERENCES media(media_id),
                FOREIGN KEY(person_id) REFERENCES people(person_id)
            )''')
            cur.execute('''CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY, entry_id TEXT NOT NULL,
                user_id TEXT NOT NULL, username TEXT NOT NULL, content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_edited BOOLEAN DEFAULT 0, parent_comment_id TEXT,
                FOREIGN KEY(entry_id) REFERENCES album_entries(entry_id)
            )''')
            cur.execute('''CREATE TABLE IF NOT EXISTS ratings (
                rating_id TEXT PRIMARY KEY, entry_id TEXT NOT NULL, user_id TEXT NOT NULL,
                rating_value INTEGER CHECK(rating_value BETWEEN 1 AND 5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entry_id, user_id),
                FOREIGN KEY(entry_id) REFERENCES album_entries(entry_id)
            )''')
            cur.execute('''CREATE TABLE IF NOT EXISTS user_favorites (
                user_id TEXT, entry_id TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, entry_id),
                FOREIGN KEY(entry_id) REFERENCES album_entries(entry_id)
            )''')
            # Create indexes (fixed)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_album_entries_media ON album_entries(media_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_album_entries_person ON album_entries(person_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_album_entries_created ON album_entries(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_comments_entry ON comments(entry_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ratings_entry ON ratings(entry_id)")
            conn.commit()

    def add_media(self, metadata: MediaMetadata, thumb_path=None, vid_thumb_path=None):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute('''INSERT OR REPLACE INTO media
                (media_id, filename, filepath, file_size, media_type, width, height, format,
                 duration, frame_rate, created_date, modified_date, exif_data, checksum,
                 thumbnail_path, video_thumbnail_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (metadata.media_id, metadata.filename, metadata.filepath, metadata.file_size,
                 metadata.media_type, metadata.dimensions[0], metadata.dimensions[1], metadata.format,
                 metadata.duration, metadata.frame_rate,
                 metadata.created_date, metadata.modified_date,
                 json.dumps(metadata.exif_data) if metadata.exif_data else None,
                 metadata.checksum, thumb_path, vid_thumb_path))
            conn.commit()

    def get_media_by_type(self, media_type: str, limit=100):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM media WHERE media_type = ? ORDER BY created_date DESC LIMIT ?",
                        (media_type, limit))
            return [dict(row) for row in cur.fetchall()]

    def add_person(self, person: PersonProfile):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute('''INSERT OR REPLACE INTO people
                (person_id, folder_name, display_name, bio, birth_date, relationship,
                 contact_info, social_links, profile_image)
                VALUES (?,?,?,?,?,?,?,?,?)''',
                (person.person_id, person.folder_name, person.display_name, person.bio,
                 person.birth_date.isoformat() if person.birth_date else None,
                 person.relationship, person.contact_info,
                 json.dumps(person.social_links), person.profile_image))
            conn.commit()

    def get_all_people(self):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM people ORDER BY display_name")
            return [dict(row) for row in cur.fetchall()]

    def get_person_by_folder(self, folder):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM people WHERE folder_name = ?", (folder,))
            row = cur.fetchone()
            return dict(row) if row else None

    def add_album_entry(self, entry: AlbumEntry):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute('''INSERT OR REPLACE INTO album_entries
                (entry_id, media_id, person_id, caption, description, location,
                 date_taken, tags, privacy_level, created_by, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                (entry.entry_id, entry.media_id, entry.person_id, entry.caption,
                 entry.description, entry.location, entry.date_taken,
                 ','.join(entry.tags) if entry.tags else None,
                 entry.privacy_level, entry.created_by, entry.created_at, entry.updated_at))
            conn.commit()

    def add_comment(self, comment: Comment):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute('''INSERT INTO comments
                (comment_id, entry_id, user_id, username, content, parent_comment_id)
                VALUES (?,?,?,?,?,?)''',
                (comment.comment_id, comment.entry_id, comment.user_id,
                 comment.username, comment.content, comment.parent_comment_id))
            conn.commit()

    def add_rating(self, rating: Rating):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute('''INSERT OR REPLACE INTO ratings
                (rating_id, entry_id, user_id, rating_value)
                VALUES (?,?,?,?)''',
                (rating.rating_id, rating.entry_id, rating.user_id, rating.rating_value))
            conn.commit()

    def get_entry_comments(self, entry_id):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM comments WHERE entry_id = ? ORDER BY created_at DESC", (entry_id,))
            return [dict(row) for row in cur.fetchall()]

    def get_entry_ratings(self, entry_id):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT AVG(rating_value), COUNT(*) FROM ratings WHERE entry_id = ?", (entry_id,))
            avg, cnt = cur.fetchone()
            return (float(avg) if avg else 0.0, cnt or 0)

    def search_entries(self, query, person_id=None, media_type=None):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            q = f'%{query}%'
            conditions = ["(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)"]
            params = [q,q,q]
            if person_id:
                conditions.append("ae.person_id = ?")
                params.append(person_id)
            if media_type and media_type != 'all':
                conditions.append("m.media_type = ?")
                params.append(media_type)
            where = " AND ".join(conditions)
            cur.execute(f'''
                SELECT ae.*, p.display_name, m.filename, m.media_type,
                       m.thumbnail_path, m.video_thumbnail_path
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
                WHERE {where} ORDER BY ae.created_at DESC
            ''', params)
            return [dict(row) for row in cur.fetchall()]

    def get_entry_details(self, entry_id):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute('''
                SELECT ae.*, p.display_name, p.folder_name,
                       m.filename, m.filepath, m.media_type, m.file_size,
                       m.format, m.duration, m.frame_rate, m.width, m.height,
                       m.created_date, m.exif_data, m.thumbnail_path, m.video_thumbnail_path,
                       (SELECT COUNT(*) FROM comments c WHERE c.entry_id = ae.entry_id) as comment_count,
                       (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id = ae.entry_id) as avg_rating,
                       (SELECT COUNT(*) FROM ratings r WHERE r.entry_id = ae.entry_id) as rating_count
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
                WHERE ae.entry_id = ?
            ''', (entry_id,))
            row = cur.fetchone()
            if not row:
                return None
            result = dict(row)
            if result.get('exif_data'):
                try:
                    result['exif_data'] = json.loads(result['exif_data'])
                except:
                    result['exif_data'] = {}
            result['tags'] = [t.strip() for t in result['tags'].split(',') if t.strip()] if result.get('tags') else []
            return result

    def get_recent_entries(self, limit=10):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute('''
                SELECT ae.*, p.display_name, m.filename, m.media_type,
                       m.thumbnail_path, m.video_thumbnail_path
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
                ORDER BY ae.created_at DESC LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cur.fetchall()]


# ============================================================================
# MEDIA PROCESSOR
# ============================================================================
class MediaProcessor:
    @staticmethod
    def create_thumbnail(media_path: Path):
        if media_path.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaProcessor._create_video_thumbnail(media_path)
        return MediaProcessor._create_image_thumbnail(media_path)

    @staticmethod
    def _create_image_thumbnail(image_path: Path):
        thumb_dir = Config.THUMBNAIL_DIR
        thumb_dir.mkdir(exist_ok=True)
        thumb_path = thumb_dir / f"{image_path.stem}_thumb.jpg"
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA','LA','P'):
                    bg = Image.new('RGB', img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA','LA') else None)
                    img = bg
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(thumb_path, 'JPEG', quality=85)
            return thumb_path
        except:
            return None

    @staticmethod
    def _create_video_thumbnail(video_path: Path):
        if not VIDEO_SUPPORT:
            return None
        thumb_dir = Config.VIDEO_THUMBNAIL_DIR
        thumb_dir.mkdir(exist_ok=True)
        thumb_path = thumb_dir / f"{video_path.stem}_thumb.jpg"
        try:
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(thumb_path, 'JPEG', quality=85)
                return thumb_path
        except:
            pass
        return None

    @staticmethod
    def get_hd_url(file_path: Path) -> str:
        try:
            with Image.open(file_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA','LA','P'):
                    bg = Image.new('RGB', img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA','LA') else None)
                    img = bg
                img.thumbnail(Config.HD_SIZE, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except:
            return ""

    @staticmethod
    def get_thumb_strip_url(file_path: Path, is_video=False) -> str:
        try:
            if is_video:
                vt = Config.VIDEO_THUMBNAIL_DIR / f"{file_path.stem}_thumb.jpg"
                if vt.exists():
                    with open(vt, "rb") as f:
                        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
                return ""
            with Image.open(file_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA','LA','P'):
                    bg = Image.new('RGB', img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA','LA') else None)
                    img = bg
                img.thumbnail(Config.THUMB_STRIP_SIZE, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=80)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except:
            return ""

    @staticmethod
    def get_media_data_url(file_path: Path) -> str:
        try:
            mt, _ = mimetypes.guess_type(str(file_path))
            if not mt:
                ext = file_path.suffix.lower()
                mm = {'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png','.gif':'image/gif',
                      '.mp4':'video/mp4','.webm':'video/webm'}
                mt = mm.get(ext, 'application/octet-stream')
            with open(file_path, "rb") as f:
                return f"data:{mt};base64,{base64.b64encode(f.read()).decode()}"
        except:
            return ""


# ============================================================================
# CACHE MANAGER
# ============================================================================
class CacheManager:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}

    def get(self, key, default=None):
        if key in self._cache:
            if time.time() - self._timestamps[key] < Config.CACHE_TTL:
                return self._cache[key]
            del self._cache[key]
            del self._timestamps[key]
        return default

    def set(self, key, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self):
        self._cache.clear()
        self._timestamps.clear()


# ============================================================================
# UI COMPONENTS
# ============================================================================
class UIComponents:
    @staticmethod
    def rating_stars(rating: float, size=20):
        stars = "⭐" * int(round(rating)) + "☆" * (5 - int(round(rating)))
        return f'<span style="font-size:{size}px;color:#FFD700;">{stars}</span> <span style="font-size:14px;">{rating:.1f}/5</span>'

    @staticmethod
    def tag_badges(tags: List[str], max_display=5):
        if not tags:
            return ""
        html = " ".join(f'<span style="background:#667eea;color:white;padding:2px 8px;border-radius:12px;font-size:12px;">{t.replace("-"," ")}</span>' for t in tags[:max_display])
        if len(tags) > max_display:
            html += f' <span style="background:#ccc;padding:2px 8px;border-radius:12px;">+{len(tags)-max_display}</span>'
        return html


# ============================================================================
# ALBUM MANAGER
# ============================================================================
class AlbumManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.media_proc = MediaProcessor()
        self._init_session()

    def _init_session(self):
        if "initialized" not in st.session_state:
            st.session_state.update({
                'initialized': True,
                'current_page': 'dashboard',
                'selected_person': None,
                'selected_media': None,
                'search_query': '',
                'view_mode': 'grid',
                'media_filter': 'all',
                'user_id': str(uuid.uuid4()),
                'username': 'Guest',
                'user_role': UserRoles.VIEWER.value,
                'favorites': set(),
                'frame_style': Config.DEFAULT_FRAME,
                'expanded_view': False,
                'thumb_offset': 0,
                'thumb_limit': 30,
            })

    def scan_directory(self):
        Config.init_directories()
        results = {'new':0, 'errors':[]}
        for person_dir in Config.DATA_DIR.iterdir():
            if not person_dir.is_dir() or person_dir.name.startswith('.') or person_dir.name in ['thumbnails','video_thumbnails','database','exports']:
                continue
            display = person_dir.name.replace('-',' ').replace('_',' ').title()
            existing = self.db.get_person_by_folder(person_dir.name)
            if not existing:
                pp = PersonProfile(person_id=str(uuid.uuid4()), folder_name=person_dir.name,
                                   display_name=display, bio="", birth_date=None,
                                   relationship="", contact_info="", social_links={},
                                   profile_image=None, created_at=datetime.datetime.now())
                self.db.add_person(pp)
                person_id = pp.person_id
            else:
                person_id = existing['person_id']
            for media_file in person_dir.iterdir():
                if not media_file.is_file() or media_file.suffix.lower() not in Config.ALLOWED_EXTENSIONS:
                    continue
                checksum = MediaMetadata._calculate_checksum(media_file)
                # check existing
                with self.db.get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT media_id FROM media WHERE checksum = ?", (checksum,))
                    if cur.fetchone():
                        continue
                metadata = MediaMetadata.from_file(media_file)
                thumb = self.media_proc.create_thumbnail(media_file) if metadata.media_type == MediaType.IMAGE.value else None
                vid_thumb = self.media_proc.create_thumbnail(media_file) if metadata.media_type == MediaType.VIDEO.value else None
                self.db.add_media(metadata, str(thumb) if thumb else None, str(vid_thumb) if vid_thumb else None)
                entry = AlbumEntry(
                    entry_id=str(uuid.uuid4()), media_id=metadata.media_id, person_id=person_id,
                    caption=media_file.stem.replace('_',' ').title(), description="",
                    location="", date_taken=metadata.created_date,
                    tags=[display.lower().replace(' ','-'), metadata.media_type],
                    privacy_level="public", created_by="system",
                    created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
                self.db.add_album_entry(entry)
                results['new'] += 1
        self.cache.clear()
        return results

    def get_all_entries_for_person(self, person_id, media_filter='all'):
        cache_key = f"all_entries_{person_id}_{media_filter}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        with self.db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            conds, params = ["ae.person_id = ?"], [person_id]
            if media_filter != 'all':
                conds.append("m.media_type = ?")
                params.append(media_filter)
            where = " AND ".join(conds)
            cur.execute(f'''
                SELECT ae.*, p.display_name, m.filename, m.media_type,
                       m.thumbnail_path, m.video_thumbnail_path, m.duration,
                       m.filepath, m.width, m.height, m.file_size, m.format,
                       (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id=ae.entry_id) as avg_rating
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
                WHERE {where} ORDER BY ae.created_at DESC
            ''', params)
            entries = [dict(row) for row in cur.fetchall()]
            self.cache.set(cache_key, entries)
            return entries

    def get_all_folders_with_files(self):
        tree = {}
        skip = {'thumbnails','video_thumbnails','database','exports'}
        for folder in sorted(Config.DATA_DIR.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.') or folder.name in skip:
                continue
            files = []
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS:
                    files.append({
                        'path': str(f),
                        'name': f.name,
                        'stem': f.stem,
                        'type': 'video' if f.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS else 'image',
                        'size': f.stat().st_size,
                    })
            if files:
                display = ' '.join(p.capitalize() for p in folder.name.replace('-',' ').replace('_',' ').split())
                tree[folder.name] = {
                    'display_name': display,
                    'files': files,
                    'image_count': sum(1 for f in files if f['type']=='image'),
                    'video_count': sum(1 for f in files if f['type']=='video')
                }
        return tree


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PhotoVideoAlbumApp:
    def __init__(self):
        st.set_page_config(page_title=Config.APP_NAME, layout="wide")
        Config.init_directories()
        self.manager = AlbumManager()
        self.tree = self.manager.get_all_folders_with_files()
        self._auto_select()
        self._init_navigation_state()

    def _auto_select(self):
        if not st.session_state.get('selected_folder') and self.tree:
            st.session_state.selected_folder = list(self.tree.keys())[0]
            st.session_state.selected_index = 0
            st.session_state.thumb_offset = 0

    def _init_navigation_state(self):
        if 'selected_folder' not in st.session_state:
            st.session_state.selected_folder = None
            st.session_state.selected_index = 0
            st.session_state.frame_style = Config.DEFAULT_FRAME
            st.session_state.expanded_view = False
            st.session_state.thumb_offset = 0
            st.session_state.thumb_limit = 30

    def _current_files(self):
        folder = st.session_state.selected_folder
        if folder and folder in self.tree:
            return self.tree[folder]['files']
        return []

    def _current_file(self):
        files = self._current_files()
        idx = st.session_state.selected_index
        if 0 <= idx < len(files):
            return files[idx]
        return None

    def render_sidebar(self):
        with st.sidebar:
            st.title("📁 MemoryVault")
            st.caption(f"v{Config.VERSION}")

            fs = st.selectbox("Frame Style", Config.FRAME_STYLES,
                              index=Config.FRAME_STYLES.index(st.session_state.frame_style))
            if fs != st.session_state.frame_style:
                st.session_state.frame_style = fs

            st.divider()
            st.subheader("Directories")
            if st.button("🔄 Refresh", use_container_width=True):
                self.tree = self.manager.get_all_folders_with_files()
                self._auto_select()
                st.rerun()

            for folder_name, info in self.tree.items():
                active = (st.session_state.selected_folder == folder_name)
                col1, col2 = st.columns([4,1])
                with col1:
                    if st.button(f"📁 {info['display_name']}", key=f"folder_{folder_name}",
                                 use_container_width=True):
                        st.session_state.selected_folder = folder_name
                        st.session_state.selected_index = 0
                        st.session_state.thumb_offset = 0
                        st.rerun()
                with col2:
                    st.caption(f"{len(info['files'])}")

            st.divider()
            total_imgs = sum(f['image_count'] for f in self.tree.values())
            total_vids = sum(f['video_count'] for f in self.tree.values())
            st.metric("Folders", len(self.tree))
            c1,c2 = st.columns(2)
            c1.metric("Images", total_imgs)
            c2.metric("Videos", total_vids)

            st.divider()
            st.session_state.expanded_view = st.toggle("🔍 Expanded View", value=st.session_state.expanded_view)

    def render_viewer(self):
        FrameRenderer.inject_css()
        files = self._current_files()
        current = self._current_file()
        if not files or not current:
            st.info("No media in this folder. Add images/videos to the data directory and click Refresh.")
            return

        idx = st.session_state.selected_index
        folder = st.session_state.selected_folder
        folder_info = self.tree[folder]

        col1, col2, col3 = st.columns([3,1,1])
        with col1:
            st.markdown(f"### {folder_info['display_name']}")
        with col2:
            st.markdown(f"<div style='text-align:right;color:#888;font-size:14px;'>{idx+1} / {len(files)}</div>", unsafe_allow_html=True)
        with col3:
            jump = st.number_input("Jump to", min_value=1, max_value=len(files), value=idx+1, label_visibility="collapsed")
            if jump != idx+1:
                st.session_state.selected_index = jump-1
                st.rerun()
        st.divider()

        col_prev, col_mid, col_next = st.columns([1, 8, 1])

        with col_prev:
            st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
            if idx > 0:
                if st.button("◀", key="prev_main", help="Previous"):
                    st.session_state.selected_index = idx - 1
                    st.rerun()
            else:
                st.markdown("<div class='nav-btn disabled'>◀</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_mid:
            if current['type'] == 'image':
                hd_url = MediaProcessor.get_hd_url(Path(current['path']))
                if hd_url:
                    st.markdown(FrameRenderer.wrap_detail(hd_url, st.session_state.frame_style,
                                                           st.session_state.expanded_view),
                                unsafe_allow_html=True)
                else:
                    st.error("Could not load image")
                with st.expander("ℹ️ Info", expanded=False):
                    st.write(f"**Name:** {current['name']}")
                    st.write(f"**Size:** {current['size']/(1024*1024):.2f} MB")
            else:
                fp = Path(current['path'])
                if fp.exists() and fp.stat().st_size < Config.MAX_VIDEO_SIZE:
                    with open(fp, 'rb') as f:
                        st.video(f.read())
                else:
                    st.warning("Video too large or missing")

        with col_next:
            st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
            if idx < len(files) - 1:
                if st.button("▶", key="next_main", help="Next"):
                    st.session_state.selected_index = idx + 1
                    st.rerun()
            else:
                st.markdown("<div class='nav-btn disabled'>▶</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Advanced thumbnail strip pagination
        if len(files) > 1:
            st.divider()
            st.markdown("#### Quick navigation – click thumbnail")

            total = len(files)
            limit = st.session_state.thumb_limit
            offset = st.session_state.thumb_offset
            max_offset = max(0, total - limit)

            col_prev_th, col_info, col_next_th = st.columns([1,4,1])
            with col_prev_th:
                if offset > 0:
                    if st.button("◀ Previous thumbnails", use_container_width=True):
                        st.session_state.thumb_offset = max(0, offset - limit)
                        st.rerun()
            with col_info:
                st.caption(f"Showing {offset+1} – {min(offset+limit, total)} of {total}")
            with col_next_th:
                if offset + limit < total:
                    if st.button("Next thumbnails ▶", use_container_width=True):
                        st.session_state.thumb_offset = min(max_offset, offset + limit)
                        st.rerun()

            batch = files[offset:offset+limit]
            st.markdown("<div class='thumb-strip-wrapper'><div class='thumb-strip'>", unsafe_allow_html=True)
            for i, f in enumerate(batch):
                actual_idx = offset + i
                fp = Path(f['path'])
                thumb_url = MediaProcessor.get_thumb_strip_url(fp, is_video=(f['type']=='video'))
                if not thumb_url:
                    thumb_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='90' height='68' viewBox='0 0 90 68'%3E%3Crect width='90' height='68' fill='%23333'/%3E%3Ctext x='45' y='38' fill='%23fff' font-size='16' text-anchor='middle'%3E📸%3C/text%3E%3C/svg%3E"
                active = (actual_idx == idx)
                st.markdown(FrameRenderer.wrap_thumb_strip_item(thumb_url, active, f['type']=='video'),
                            unsafe_allow_html=True)
                if st.button(str(actual_idx+1), key=f"thumb_{actual_idx}", use_container_width=True):
                    st.session_state.selected_index = actual_idx
                    st.rerun()
            st.markdown("</div></div>", unsafe_allow_html=True)

    def run(self):
        self.render_sidebar()
        self.render_viewer()


# ============================================================================
# MAIN
# ============================================================================
def main():
    if not check_password():
        return
    app = PhotoVideoAlbumApp()
    app.run()

if __name__ == "__main__":
    main()
