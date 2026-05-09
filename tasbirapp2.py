"""
COMPREHENSIVE WEB PHOTO & VIDEO ALBUM APPLICATION
Version: 5.0.0 - Luxury Frames, Cipher Auth, HD Slider, Directory Panel
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ExifTags, ImageDraw, ImageFilter
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

try:
    import cv2
    import moviepy.editor as mp
    from moviepy.editor import VideoFileClip
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False

# ============================================================================
# CIPHER-BASED PASSWORD AUTHENTICATION
# ============================================================================
# Digit-to-letter cipher used ONLY to obfuscate the password inside the
# source code so that casual readers cannot see the real numeric key.
# a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=0
_CIPHER = {'0': 'j', '1': 'a', '2': 'b', '3': 'c', '4': 'd',
           '5': 'e', '6': 'f', '7': 'g', '8': 'h', '9': 'i'}
_REV = {v: k for k, v in _CIPHER.items()}

# Stored in alphabet form so the numeric password is never visible in code.
# Decodes: a→1  i→9  h→8  g→7  j→0  e→5  j→0  e→5  →  "19870505"
_ACCESS_CIPHER = "aihgjeje"


def _cipher_to_numeric(cipher: str) -> str:
    """Decode the stored alphabet cipher back to the real numeric password."""
    return "".join(_REV.get(ch, ch) for ch in cipher)


_REAL_PASSWORD = _cipher_to_numeric(_ACCESS_CIPHER)


def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.markdown("""
    <style>
    .login-bg{background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
              padding:50px 30px;border-radius:20px;text-align:center;
              box-shadow:0 20px 60px rgba(0,0,0,.4);margin:30px 0;}
    .login-title{font-size:2.4em;font-weight:800;
                 background:linear-gradient(90deg,#f9d423,#ff4e50);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .login-sub{color:#a0a0c0;font-size:1.1em;margin-bottom:30px;}
    .lock-icon{font-size:64px;margin-bottom:10px;}
    </style>
    <div class="login-bg">
        <div class="lock-icon">🔐</div>
        <div class="login-title">MemoryVault Pro+</div>
        <div class="login-sub">Secure Photo &amp; Video Album</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Access Key", type="password", key="password_input",
                                 placeholder="Enter your access key", label_visibility="collapsed")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔓 Unlock", use_container_width=True, type="primary"):
                # ONLY the numeric form unlocks – alphabet input is rejected
                if password.strip().isdigit() and password.strip() == _REAL_PASSWORD:
                    st.session_state.authenticated = True
                    st.success("✅ Access granted!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Invalid access key.")
                    time.sleep(0.4)
        with col_b:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.authenticated = False
                st.rerun()

        with st.expander("🔑 Need a hint?"):
            # Deliberately MISLEADING hint – does NOT reveal the actual key
            st.info("💡 The access key is a **numeric** code — letters are not accepted.")
            st.warning("🤔 Think of something personal and significant that you'd never forget.")
            st.caption("It's 8 digits long. Only numbers work.")
    return False


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    APP_NAME = "MemoryVault Pro+"
    VERSION = "5.0.0"
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    VIDEO_THUMBNAIL_DIR = BASE_DIR / "video_thumbnails"
    METADATA_DIR = BASE_DIR / "metadata"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    VIDEO_CACHE_DIR = BASE_DIR / "video_cache"
    METADATA_FILE = METADATA_DIR / "album_metadata.json"
    DB_FILE = DB_DIR / "album.db"
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (800, 800)
    HD_SIZE = (1920, 1080)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024
    MAX_VIDEO_SIZE = 100 * 1024 * 1024
    VIDEO_THUMBNAIL_SIZE = (300, 300)
    VIDEO_PREVIEW_SIZE = (800, 450)
    VIDEO_CACHE_SIZE = 50 * 1024 * 1024
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v']
    ITEMS_PER_PAGE = 20
    GRID_COLUMNS = 4
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'} | set(SUPPORTED_VIDEO_FORMATS)
    MAX_COMMENT_LENGTH = 500
    MAX_CAPTION_LENGTH = 200
    CACHE_TTL = 3600
    FRAME_STYLES = ["Elegant Gold", "Polaroid", "Modern Shadow", "Dark Museum", "Vintage", "Gallery White"]
    DEFAULT_FRAME_STYLE = "Elegant Gold"
    SLIDER_THUMB_COUNT = 12

    @classmethod
    def init_directories(cls):
        for d in [cls.DATA_DIR, cls.THUMBNAIL_DIR, cls.VIDEO_THUMBNAIL_DIR,
                  cls.METADATA_DIR, cls.DB_DIR, cls.EXPORT_DIR, cls.VIDEO_CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_sample_structure()

    @classmethod
    def create_sample_structure(cls):
        for person in ["john-smith", "sarah-johnson", "michael-brown"]:
            pd_dir = cls.DATA_DIR / person
            pd_dir.mkdir(exist_ok=True)
            (pd_dir / "README.txt").write_text(f"Photos of {person.replace('-',' ').title()}\n")
            sp = pd_dir / "sample.jpg"
            if not sp.exists():
                try:
                    img = Image.new('RGB', (400, 300), color='#667eea')
                    draw = ImageDraw.Draw(img)
                    draw.ellipse((150, 50, 250, 150), fill='#fff', outline='#4a5568')
                    draw.rectangle((150, 150, 250, 280), fill='#fff', outline='#4a5568')
                    draw.text((120, 250), person.replace('-', ' ').title(), fill='#2d3748')
                    img.save(sp)
                except Exception:
                    pass


class UserRoles(Enum):
    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    EDITOR = "editor"
    ADMIN = "admin"


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class FrameStyle(Enum):
    ELEGANT_GOLD = "Elegant Gold"
    POLAROID = "Polaroid"
    MODERN_SHADOW = "Modern Shadow"
    DARK_MUSEUM = "Dark Museum"
    VINTAGE = "Vintage"
    GALLERY_WHITE = "Gallery White"


# ============================================================================
# FRAME RENDERER
# ============================================================================
class FrameRenderer:
    @staticmethod
    def wrap_thumbnail(image_src: str, caption: str = "", frame_style: str = "Elegant Gold",
                       is_video: bool = False, duration: float = None) -> str:
        dur = ""
        if is_video and duration:
            m, s = int(duration // 60), int(duration % 60)
            dur = (f'<div style="position:absolute;bottom:8px;right:8px;'
                   f'background:rgba(0,0,0,.75);color:#fff;padding:2px 7px;'
                   f'border-radius:4px;font-size:10px;font-weight:700;">{m:02d}:{s:02d}</div>')
        play = ''
        if is_video:
            play = ('<div style="position:absolute;top:50%;left:50%;'
                    'transform:translate(-50%,-50%);font-size:36px;color:#fff;'
                    'opacity:.85;text-shadow:1px 1px 6px rgba(0,0,0,.6);">▶</div>')

        styles = {
            "Elegant Gold": (
                'background:linear-gradient(135deg,#d4a574,#f0d9b5,#c9956b,#f0d9b5,#d4a574);'
                'padding:8px;border-radius:6px;'
                'box-shadow:0 6px 18px rgba(0,0,0,.25),inset 0 1px 0 rgba(255,255,255,.4);',
                'background:#fff;padding:6px;border-radius:3px;box-shadow:inset 0 0 10px rgba(0,0,0,.06);'),
            "Polaroid": ('background:#fff;padding:10px 10px 38px 10px;'
                         'box-shadow:0 4px 12px rgba(0,0,0,.18);border-radius:2px;', ''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:12px;'
                              'box-shadow:0 8px 24px rgba(0,0,0,.15);overflow:hidden;', ''),
            "Dark Museum": ('background:linear-gradient(145deg,#1a1a2e,#16213e);padding:14px;'
                            'border-radius:10px;box-shadow:0 12px 36px rgba(0,0,0,.45),'
                            '0 0 0 1px rgba(255,255,255,.06);',
                            'background:#fff;padding:6px;border-radius:3px;box-shadow:inset 0 0 12px rgba(0,0,0,.04);'),
            "Vintage": ('background:linear-gradient(135deg,#e8d5b7,#f5e6cc,#d4b896);padding:10px;'
                        'border-radius:4px;box-shadow:0 4px 14px rgba(0,0,0,.2),'
                        'inset 0 0 30px rgba(139,109,63,.15);border:1px solid #c9a96e;',
                        'background:#faf5ee;padding:5px;border-radius:2px;'),
            "Gallery White": ('background:#fafafa;padding:12px;border-radius:2px;'
                              'box-shadow:0 2px 10px rgba(0,0,0,.08);border:1px solid #e8e8e8;', ''),
        }
        outer, inner = styles.get(frame_style, styles["Elegant Gold"])
        cap_html = ""
        if frame_style == "Polaroid" and caption:
            cap_html = (f'<div style="text-align:center;padding-top:8px;font-family:Georgia,serif;'
                        f'font-size:12px;color:#444;white-space:nowrap;overflow:hidden;'
                        f'text-overflow:ellipsis;">{caption}</div>')
        return f"""
        <div style="{outer}position:relative;transition:transform .25s ease,box-shadow .25s ease;"
             onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 32px rgba(0,0,0,.3)'"
             onmouseout="this.style.transform='translateY(0)';this.style.boxShadow=''">
            <div style="{inner}position:relative;overflow:hidden;border-radius:3px;">
                <img src="{image_src}" style="width:100%;height:200px;object-fit:cover;display:block;border-radius:2px;">
                {play}{dur}
            </div>{cap_html}
        </div>"""

    @staticmethod
    def wrap_detail(image_src: str, frame_style: str = "Elegant Gold") -> str:
        styles = {
            "Elegant Gold": (
                'background:linear-gradient(135deg,#b8860b,#daa520,#ffd700,#daa520,#b8860b);'
                'padding:12px;border-radius:8px;box-shadow:0 16px 48px rgba(0,0,0,.35),'
                'inset 0 2px 0 rgba(255,255,255,.3),inset 0 -2px 0 rgba(0,0,0,.2);',
                'background:#fffff5;padding:18px;border-radius:4px;box-shadow:inset 0 0 20px rgba(0,0,0,.06);'),
            "Polaroid": ('background:#fff;padding:18px 18px 64px 18px;'
                         'box-shadow:0 8px 28px rgba(0,0,0,.18);border-radius:2px;', ''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:14px;'
                              'box-shadow:0 12px 40px rgba(0,0,0,.18);overflow:hidden;', ''),
            "Dark Museum": ('background:linear-gradient(160deg,#0d0d1a,#1a1a30,#0d0d1a);padding:24px;'
                            'border-radius:14px;box-shadow:0 20px 60px rgba(0,0,0,.5),'
                            '0 0 0 1px rgba(255,255,255,.04);',
                            'background:#fffff8;padding:16px;border-radius:4px;box-shadow:inset 0 0 16px rgba(0,0,0,.04);'),
            "Vintage": ('background:linear-gradient(135deg,#d4b896,#e8d5b7,#c9a96e);padding:14px;'
                        'border-radius:4px;box-shadow:0 10px 30px rgba(0,0,0,.25),'
                        'inset 0 0 40px rgba(139,109,63,.12);border:2px solid #a08050;',
                        'background:#faf5ee;padding:12px;border-radius:2px;box-shadow:inset 0 0 10px rgba(0,0,0,.04);'),
            "Gallery White": ('background:#fff;padding:20px;border-radius:4px;'
                              'box-shadow:0 4px 20px rgba(0,0,0,.08);border:1px solid #e0e0e0;', ''),
        }
        outer, inner = styles.get(frame_style, styles["Elegant Gold"])
        return f"""<div style="{outer}"><div style="{inner}">
            <img src="{image_src}" style="width:100%;display:block;border-radius:2px;">
        </div></div>"""

    @staticmethod
    def inject_global_css():
        st.markdown("""
        <style>
        .stApp{scroll-behavior:smooth;}
        .breadcrumb{display:flex;align-items:center;gap:6px;font-size:13px;color:#888;margin-bottom:16px;flex-wrap:wrap;}
        .breadcrumb a{color:#667eea;text-decoration:none;}
        .breadcrumb a:hover{text-decoration:underline;}
        .breadcrumb .sep{color:#ccc;}
        .fullscreen-overlay{position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,.92);
            z-index:9999;display:flex;align-items:center;justify-content:center;cursor:zoom-out;}
        .fullscreen-overlay img{max-width:95vw;max-height:92vh;object-fit:contain;border-radius:4px;
            box-shadow:0 0 60px rgba(0,0,0,.5);}
        .slider-nav-btn{background:rgba(102,126,234,.85);color:#fff;border:none;padding:12px 20px;
            border-radius:50%;cursor:pointer;font-size:22px;transition:background .2s,transform .2s;
            box-shadow:0 4px 12px rgba(0,0,0,.2);}
        .slider-nav-btn:hover{background:#764ba2;transform:scale(1.1);}
        .thumb-strip{display:flex;gap:6px;overflow-x:auto;padding:8px 0;scroll-behavior:smooth;
            -webkit-overflow-scrolling:touch;}
        .thumb-strip::-webkit-scrollbar{height:6px;}
        .thumb-strip::-webkit-scrollbar-thumb{background:#667eea;border-radius:3px;}
        .thumb-item{min-width:72px;height:54px;border-radius:4px;overflow:hidden;cursor:pointer;
            border:2px solid transparent;transition:border-color .2s,transform .2s;opacity:.6;}
        .thumb-item:hover{opacity:1;transform:scale(1.05);}
        .thumb-item.active{border-color:#667eea;opacity:1;box-shadow:0 2px 8px rgba(102,126,234,.4);}
        .thumb-item img{width:100%;height:100%;object-fit:cover;display:block;}
        .dir-tree-item{padding:6px 10px;border-radius:6px;cursor:pointer;transition:background .15s;
            display:flex;align-items:center;gap:8px;font-size:13px;}
        .dir-tree-item:hover{background:rgba(102,126,234,.1);}
        .dir-tree-item.active{background:rgba(102,126,234,.15);font-weight:600;}
        .dir-file-item{padding:4px 10px 4px 28px;border-radius:4px;cursor:pointer;transition:background .15s;
            display:flex;align-items:center;gap:6px;font-size:12px;color:#666;}
        .dir-file-item:hover{background:rgba(102,126,234,.08);color:#333;}
        .slider-counter{text-align:center;color:#888;font-size:13px;padding:4px 0;}
        </style>
        """, unsafe_allow_html=True)


def render_breadcrumb(trail: List[Tuple[str, str]]):
    parts = []
    for i, (label, key) in enumerate(trail):
        if i < len(trail) - 1:
            parts.append(f'<a href="#" data-key="{key}">{label}</a><span class="sep">›</span>')
        else:
            parts.append(f'<span style="color:#333;font-weight:600;">{label}</span>')
    st.markdown(f'<div class="breadcrumb">{"".join(parts)}</div>', unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class MediaMetadata:
    media_id: str
    filename: str
    filepath: str
    file_size: int
    media_type: str
    dimensions: Tuple[int, int]
    format: str
    duration: Optional[float]
    frame_rate: Optional[float]
    created_date: datetime.datetime
    modified_date: datetime.datetime
    exif_data: Optional[Dict]
    checksum: str

    @classmethod
    def from_file(cls, file_path: Path) -> 'MediaMetadata':
        if not file_path.exists():
            raise FileNotFoundError(f"Not found: {file_path}")
        mt = cls._detect_media_type(file_path)
        stats = file_path.stat()
        if mt == MediaType.IMAGE.value:
            return cls._from_image(file_path, stats, mt)
        return cls._from_video(file_path, stats, mt)

    @staticmethod
    def _detect_media_type(fp: Path) -> str:
        ext = fp.suffix.lower()
        if ext in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaType.VIDEO.value
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']:
            return MediaType.IMAGE.value
        raise ValueError(f"Unsupported: {ext}")

    @classmethod
    def _from_image(cls, ip: Path, stats, mt: str) -> 'MediaMetadata':
        with Image.open(ip) as img:
            return cls(media_id=str(uuid.uuid4()), filename=ip.name,
                       filepath=str(ip.relative_to(Config.DATA_DIR)),
                       file_size=stats.st_size, media_type=mt, dimensions=img.size,
                       format=img.format, duration=None, frame_rate=None,
                       created_date=datetime.datetime.fromtimestamp(stats.st_ctime),
                       modified_date=datetime.datetime.fromtimestamp(stats.st_mtime),
                       exif_data=cls._extract_exif(img), checksum=cls._calculate_checksum(ip))

    @classmethod
    def _from_video(cls, vp: Path, stats, mt: str) -> 'MediaMetadata':
        dims, dur, fr = (0, 0), 0.0, 0.0
        if VIDEO_SUPPORT:
            try:
                clip = VideoFileClip(str(vp))
                dims, dur, fr = clip.size, clip.duration, clip.fps
                clip.close()
            except Exception:
                try:
                    cap = cv2.VideoCapture(str(vp))
                    dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    fr = cap.get(cv2.CAP_PROP_FPS)
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    dur = fc / fr if fr > 0 else 0
                    cap.release()
                except Exception:
                    pass
        return cls(media_id=str(uuid.uuid4()), filename=vp.name,
                   filepath=str(vp.relative_to(Config.DATA_DIR)),
                   file_size=stats.st_size, media_type=mt, dimensions=dims,
                   format=vp.suffix[1:].upper(), duration=dur, frame_rate=fr,
                   created_date=datetime.datetime.fromtimestamp(stats.st_ctime),
                   modified_date=datetime.datetime.fromtimestamp(stats.st_mtime),
                   exif_data=None, checksum=cls._calculate_checksum(vp))

    @staticmethod
    def _extract_exif(img):
        try:
            exif = {}
            if hasattr(img, '_getexif') and img._getexif():
                for tid, val in img._getexif().items():
                    tag = ExifTags.TAGS.get(tid, tid)
                    if not isinstance(val, (bytes, np.ndarray)):
                        exif[tag] = str(val)
            return exif or None
        except Exception:
            return None

    @staticmethod
    def _calculate_checksum(fp: Path) -> str:
        h = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()


@dataclass
class AlbumEntry:
    entry_id: str; media_id: str; person_id: str; caption: str; description: str
    location: str; date_taken: Optional[datetime.datetime]; tags: List[str]
    privacy_level: str; created_by: str; created_at: datetime.datetime; updated_at: datetime.datetime
    def to_dict(self):
        d = asdict(self)
        d['date_taken'] = self.date_taken.isoformat() if self.date_taken else None
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d


@dataclass
class Comment:
    comment_id: str; entry_id: str; user_id: str; username: str; content: str
    created_at: datetime.datetime; is_edited: bool; parent_comment_id: Optional[str]
    def to_dict(self):
        d = asdict(self); d['created_at'] = self.created_at.isoformat(); return d


@dataclass
class Rating:
    rating_id: str; entry_id: str; user_id: str; rating_value: int
    created_at: datetime.datetime; updated_at: datetime.datetime
    def to_dict(self):
        d = asdict(self); d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat(); return d


@dataclass
class PersonProfile:
    person_id: str; folder_name: str; display_name: str; bio: str
    birth_date: Optional[datetime.date]; relationship: str; contact_info: str
    social_links: Dict[str, str]; profile_image: Optional[str]; created_at: datetime.datetime
    def to_dict(self):
        d = asdict(self)
        d['birth_date'] = self.birth_date.isoformat() if self.birth_date else None
        d['created_at'] = self.created_at.isoformat(); return d


# ============================================================================
# DATABASE
# ============================================================================
class DatabaseManager:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Config.DB_FILE
        self._init_database()

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except sqlite3.Error as e:
            st.error(f"DB error: {e}"); raise
        finally:
            if conn: conn.close()

    def _init_database(self):
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS media(
                    media_id TEXT PRIMARY KEY,filename TEXT NOT NULL,filepath TEXT NOT NULL,
                    file_size INTEGER,media_type TEXT NOT NULL,width INTEGER,height INTEGER,
                    format TEXT,duration REAL,frame_rate REAL,created_date TIMESTAMP,
                    modified_date TIMESTAMP,exif_data TEXT,checksum TEXT UNIQUE,
                    thumbnail_path TEXT,video_thumbnail_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
                c.execute('''CREATE TABLE IF NOT EXISTS people(
                    person_id TEXT PRIMARY KEY,folder_name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,bio TEXT,birth_date DATE,
                    relationship TEXT,contact_info TEXT,social_links TEXT,
                    profile_image TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
                c.execute('''CREATE TABLE IF NOT EXISTS album_entries(
                    entry_id TEXT PRIMARY KEY,media_id TEXT NOT NULL,person_id TEXT NOT NULL,
                    caption TEXT,description TEXT,location TEXT,date_taken TIMESTAMP,
                    tags TEXT,privacy_level TEXT DEFAULT 'public',created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(media_id)REFERENCES media(media_id),
                    FOREIGN KEY(person_id)REFERENCES people(person_id))''')
                c.execute('''CREATE TABLE IF NOT EXISTS comments(
                    comment_id TEXT PRIMARY KEY,entry_id TEXT NOT NULL,user_id TEXT NOT NULL,
                    username TEXT NOT NULL,content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,is_edited BOOLEAN DEFAULT 0,
                    parent_comment_id TEXT,FOREIGN KEY(entry_id)REFERENCES album_entries(entry_id))''')
                c.execute('''CREATE TABLE IF NOT EXISTS ratings(
                    rating_id TEXT PRIMARY KEY,entry_id TEXT NOT NULL,user_id TEXT NOT NULL,
                    rating_value INTEGER CHECK(rating_value BETWEEN 1 AND 5),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entry_id,user_id),FOREIGN KEY(entry_id)REFERENCES album_entries(entry_id))''')
                c.execute('''CREATE TABLE IF NOT EXISTS user_favorites(
                    user_id TEXT,entry_id TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_id,entry_id),
                    FOREIGN KEY(entry_id)REFERENCES album_entries(entry_id))''')
                for idx in ['idx_media_type','idx_ae_media','idx_ae_person','idx_ae_created','idx_comm_entry','idx_rat_entry']:
                    q = {'idx_media_type':'CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type)',
                         'idx_ae_media':'CREATE INDEX IF NOT EXISTS idx_ae_media ON album_entries(media_id)',
                         'idx_ae_person':'CREATE INDEX IF NOT EXISTS idx_ae_person ON album_entries(person_id)',
                         'idx_ae_created':'CREATE INDEX IF NOT EXISTS idx_ae_created ON album_entries(created_at)',
                         'idx_comm_entry':'CREATE INDEX IF NOT EXISTS idx_comm_entry ON comments(entry_id)',
                         'idx_rat_entry':'CREATE INDEX IF NOT EXISTS idx_rat_entry ON ratings(entry_id)'}[idx]
                    c.execute(q)
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
                if c.fetchone():
                    c.execute("SELECT COUNT(*) FROM media")
                    if c.fetchone()[0] == 0:
                        c.execute('''INSERT INTO media(media_id,filename,filepath,file_size,media_type,
                            width,height,format,duration,frame_rate,created_date,modified_date,
                            exif_data,checksum,thumbnail_path)
                            SELECT image_id,filename,filepath,file_size,'image',width,height,format,
                            NULL,NULL,created_date,modified_date,exif_data,checksum,thumbnail_path FROM images''')
                        c.execute("DROP TABLE images")
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"DB init error: {e}"); raise

    def add_media(self, meta: MediaMetadata, tp: str = None, vtp: str = None):
        with self.get_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO media(media_id,filename,filepath,file_size,
                media_type,width,height,format,duration,frame_rate,created_date,modified_date,
                exif_data,checksum,thumbnail_path,video_thumbnail_path)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (meta.media_id, meta.filename, meta.filepath, meta.file_size, meta.media_type,
                 meta.dimensions[0], meta.dimensions[1], meta.format, meta.duration, meta.frame_rate,
                 meta.created_date, meta.modified_date,
                 json.dumps(meta.exif_data) if meta.exif_data else None,
                 meta.checksum, tp, vtp))
            conn.commit()

    def get_media(self, mid: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute('SELECT * FROM media WHERE media_id=?', (mid,)).fetchone()
            return dict(r) if r else None

    def get_media_by_type(self, mt: str, limit: int = 100) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(
                'SELECT * FROM media WHERE media_type=? ORDER BY created_date DESC LIMIT ?', (mt, limit)).fetchall()]

    def add_person(self, p: PersonProfile):
        with self.get_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO people(person_id,folder_name,display_name,bio,
                birth_date,relationship,contact_info,social_links,profile_image)
                VALUES(?,?,?,?,?,?,?,?,?)''',
                (p.person_id, p.folder_name, p.display_name, p.bio,
                 p.birth_date.isoformat() if p.birth_date else None,
                 p.relationship, p.contact_info, json.dumps(p.social_links), p.profile_image))
            conn.commit()

    def get_all_people(self) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute('SELECT * FROM people ORDER BY display_name').fetchall()]

    def get_person_by_folder(self, fn: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute('SELECT * FROM people WHERE folder_name=?', (fn,)).fetchone()
            return dict(r) if r else None

    def add_album_entry(self, e: AlbumEntry):
        with self.get_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO album_entries(entry_id,media_id,person_id,
                caption,description,location,date_taken,tags,privacy_level,created_by)
                VALUES(?,?,?,?,?,?,?,?,?,?)''',
                (e.entry_id, e.media_id, e.person_id, e.caption, e.description, e.location,
                 e.date_taken, ','.join(e.tags) if e.tags else None, e.privacy_level, e.created_by))
            conn.commit()

    def add_comment(self, cm: Comment):
        with self.get_connection() as conn:
            conn.execute('''INSERT INTO comments(comment_id,entry_id,user_id,username,content,parent_comment_id)
                VALUES(?,?,?,?,?,?)''',
                (cm.comment_id, cm.entry_id, cm.user_id, cm.username, cm.content, cm.parent_comment_id))
            conn.commit()

    def add_rating(self, r: Rating):
        with self.get_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO ratings(rating_id,entry_id,user_id,rating_value)
                VALUES(?,?,?,?)''', (r.rating_id, r.entry_id, r.user_id, r.rating_value))
            conn.commit()

    def get_entry_comments(self, eid: str) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(
                'SELECT * FROM comments WHERE entry_id=? ORDER BY created_at DESC', (eid,)).fetchall()]

    def get_entry_ratings(self, eid: str) -> Tuple[float, int]:
        with self.get_connection() as conn:
            r = conn.execute('SELECT AVG(rating_value),COUNT(*) FROM ratings WHERE entry_id=?', (eid,)).fetchone()
            return (float(r[0]) if r and r[0] is not None else 0.0, r[1] if r else 0)

    def search_entries(self, q: str, pid: str = None, mt: str = None) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            sp = f'%{q}%'
            conds, params = [], []
            if pid: conds.append("ae.person_id=?"); params.append(pid)
            if mt and mt != 'all': conds.append("m.media_type=?"); params.append(mt)
            conds.append("(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)")
            params.extend([sp, sp, sp])
            w = " AND ".join(conds)
            return [dict(r) for r in conn.execute(f'''SELECT ae.*,p.display_name,m.filename,m.media_type,
                m.thumbnail_path,m.video_thumbnail_path FROM album_entries ae
                JOIN people p ON ae.person_id=p.person_id JOIN media m ON ae.media_id=m.media_id
                WHERE {w} ORDER BY ae.created_at DESC''', params).fetchall()]

    def get_entry_details(self, eid: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('''SELECT ae.*,p.display_name,p.folder_name,m.filename,m.filepath,
                m.media_type,m.file_size,m.format,m.duration,m.frame_rate,m.width,m.height,
                m.created_date,m.exif_data,m.thumbnail_path,m.video_thumbnail_path,
                (SELECT COUNT(*)FROM comments c WHERE c.entry_id=ae.entry_id)as comment_count,
                (SELECT AVG(rating_value)FROM ratings r WHERE r.entry_id=ae.entry_id)as avg_rating,
                (SELECT COUNT(*)FROM ratings r2 WHERE r2.entry_id=ae.entry_id)as rating_count
                FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                JOIN media m ON ae.media_id=m.media_id WHERE ae.entry_id=?''', (eid,)).fetchone()
            if row:
                res = dict(row)
                if res.get('exif_data'):
                    try: res['exif_data'] = json.loads(res['exif_data'])
                    except: res['exif_data'] = {}
                res['tags'] = [t.strip() for t in res['tags'].split(',') if t.strip()] if res.get('tags') else []
                return res
            return None

    def get_recent_entries(self, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute('''SELECT ae.*,p.display_name,m.filename,m.media_type,
                m.thumbnail_path,m.video_thumbnail_path FROM album_entries ae
                JOIN people p ON ae.person_id=p.person_id JOIN media m ON ae.media_id=m.media_id
                ORDER BY ae.created_at DESC LIMIT ?''', (limit,)).fetchall()]

    def get_all_entries_with_details(self, person_id: str = None, media_filter: str = 'all') -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            conds, params = [], []
            if person_id: conds.append("ae.person_id=?"); params.append(person_id)
            if media_filter != 'all': conds.append("m.media_type=?"); params.append(media_filter)
            w = " AND ".join(conds) if conds else "1=1"
            return [dict(r) for r in conn.execute(f'''SELECT ae.*,p.display_name,p.folder_name,
                m.filename,m.filepath,m.media_type,m.file_size,m.format,m.duration,m.width,m.height,
                m.thumbnail_path,m.video_thumbnail_path
                FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                JOIN media m ON ae.media_id=m.media_id WHERE {w}
                ORDER BY ae.created_at DESC''', params).fetchall()]


# ============================================================================
# MEDIA PROCESSOR
# ============================================================================
class MediaProcessor:
    @staticmethod
    def create_thumbnail(mp: Path, td: Path = None) -> Optional[Path]:
        if not mp.exists(): return None
        if mp.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaProcessor._video_thumb(mp, td)
        return MediaProcessor._img_thumb(mp, td)

    @staticmethod
    def _img_thumb(ip: Path, td: Path = None) -> Optional[Path]:
        td = td or Config.THUMBNAIL_DIR; os.makedirs(td, exist_ok=True)
        tp = td / f"{ip.stem}_thumb.jpg"
        try:
            with Image.open(ip) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1]) if img.mode in ('RGBA', 'LA') else bg.paste(img)
                    img = bg
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(tp, 'JPEG', quality=85, optimize=True)
            return tp
        except Exception as e:
            st.error(f"Thumb error {ip}: {e}"); return None

    @staticmethod
    def _video_thumb(vp: Path, td: Path = None) -> Optional[Path]:
        if not VIDEO_SUPPORT: return None
        td = td or Config.VIDEO_THUMBNAIL_DIR; os.makedirs(td, exist_ok=True)
        tp = td / f"{vp.stem}_thumb.jpg"
        try:
            cap = cv2.VideoCapture(str(vp))
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tf > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, tf // 2)
            ret, frame = cap.read(); cap.release()
            if ret and frame is not None:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail(Config.VIDEO_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(tp, 'JPEG', quality=85, optimize=True)
                return tp
        except Exception:
            pass
        return None

    @staticmethod
    def get_media_data_url(mp: Path) -> str:
        try:
            if not mp.exists(): return ""
            mt, _ = mimetypes.guess_type(str(mp))
            if not mt:
                mm = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                      '.gif': 'image/gif', '.mp4': 'video/mp4', '.webm': 'video/webm'}
                mt = mm.get(mp.suffix.lower(), 'application/octet-stream')
            with open(mp, "rb") as f:
                return f"data:{mt};base64,{base64.b64encode(f.read()).decode('utf-8')}"
        except Exception:
            return ""

    @staticmethod
    def get_hd_data_url(mp: Path) -> str:
        try:
            if not mp.exists(): return ""
            with Image.open(mp) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1]) if img.mode in ('RGBA', 'LA') else bg.paste(img)
                    img = bg
                max_w, max_h = Config.HD_SIZE
                if img.width > max_w or img.height > max_h:
                    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return MediaProcessor.get_media_data_url(mp)

    @staticmethod
    def prepare_video_stream(vp: Path, max_mb: int = 50) -> Optional[bytes]:
        try:
            if not vp.exists(): return None
            if vp.stat().st_size > max_mb * 1024 * 1024:
                st.warning(f"Video too large ({vp.stat().st_size/(1024*1024):.1f}MB)")
                return None
            with open(vp, "rb") as f: return f.read()
        except Exception:
            return None

    @staticmethod
    def get_video_info(vp: Path) -> Dict:
        info = {'duration': 0, 'dimensions': (0, 0), 'frame_rate': 0,
                'file_size': vp.stat().st_size if vp.exists() else 0}
        if not VIDEO_SUPPORT or not vp.exists(): return info
        try:
            clip = VideoFileClip(str(vp))
            info.update({'duration': clip.duration, 'dimensions': clip.size, 'frame_rate': clip.fps})
            clip.close()
        except Exception:
            try:
                cap = cv2.VideoCapture(str(vp))
                info['dimensions'] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                info['frame_rate'] = cap.get(cv2.CAP_PROP_FPS)
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                info['duration'] = fc / info['frame_rate'] if info['frame_rate'] > 0 else 0
                cap.release()
            except Exception:
                pass
        return info


# ============================================================================
# CACHE
# ============================================================================
class CacheManager:
    def __init__(self):
        self._cache = {}; self._ts = {}; self._vc = {}

    def get(self, k, default=None):
        if k in self._cache and time.time() - self._ts[k] < Config.CACHE_TTL:
            return self._cache[k]
        self._cache.pop(k, None); self._ts.pop(k, None)
        return default

    def set(self, k, v):
        self._cache[k] = v; self._ts[k] = time.time()

    def clear(self, k=None):
        if k: self._cache.pop(k, None); self._ts.pop(k, None)
        else: self._cache.clear(); self._ts.clear(); self._vc.clear()

    def get_or_set(self, k, fn):
        c = self.get(k)
        if c is not None: return c
        try:
            v = fn(); self.set(k, v); return v
        except Exception as e:
            st.error(f"Cache error: {e}"); return None

    def get_video(self, p): return self._vc.get(f"video_{p}")
    def set_video(self, p, d):
        self._vc[f"video_{p}"] = d
        if len(self._vc) > 10:
            del self._vc[next(iter(self._vc))]
    def clear_video_cache(self): self._vc.clear()


# ============================================================================
# UI COMPONENTS
# ============================================================================
class UIComponents:
    @staticmethod
    def rating_stars(rating, max_r=5, size=20):
        if not rating or rating <= 0: rating = 0
        full = int(rating); half = rating - full >= 0.5
        stars = []
        for i in range(max_r):
            if i < full: stars.append('⭐')
            elif i == full and half: stars.append('⭐')
            else: stars.append('☆')
        return (f'<div style="color:#FFD700;font-size:{size}px;letter-spacing:1px;display:inline-block;">'
                f'{"".join(stars)}<span style="color:#666;font-size:14px;margin-left:8px;">'
                f'{rating:.1f}/{max_r}</span></div>')

    @staticmethod
    def tag_badges(tags, max_d=5):
        if not tags: return ""
        shown = tags[:max_d]; extra = len(tags) - max_d if len(tags) > max_d else 0
        b = [f'<span style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:4px 12px;'
             f'border-radius:20px;font-size:12px;margin:2px;display:inline-block;box-shadow:0 2px 4px rgba(0,0,0,.1);'
             f'text-transform:capitalize;">{t.replace("-"," ")}</span>' for t in shown]
        h = ' '.join(b)
        if extra > 0:
            h += f'<span style="background:#f0f0f0;color:#666;padding:4px 8px;border-radius:20px;font-size:12px;margin:2px;display:inline-block;">+{extra}</span>'
        return h

    @staticmethod
    def media_type_badge(mt):
        if mt == MediaType.VIDEO.value:
            return '<span style="background:linear-gradient(135deg,#FF416C,#FF4B2B);color:#fff;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700;margin-left:8px;display:inline-block;">🎬 VIDEO</span>'
        return '<span style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700;margin-left:8px;display:inline-block;">📸 IMAGE</span>'


# ============================================================================
# ALBUM MANAGER
# ============================================================================
class AlbumManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.mp = MediaProcessor()
        self._init_ss()

    def _init_ss(self):
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'initialized': True, 'current_page': 'dashboard',
                'selected_person': None, 'selected_media': None,
                'search_query': '', 'view_mode': 'grid', 'sort_by': 'date',
                'sort_order': 'desc', 'media_filter': 'all', 'selected_tags': [],
                'user_id': str(uuid.uuid4()), 'username': 'Guest',
                'user_role': UserRoles.VIEWER.value, 'favorites': set(),
                'recently_viewed': [], 'toc_page': 1, 'gallery_page': 1,
                'show_directory_info': True, 'video_autoplay': False,
                'frame_style': Config.DEFAULT_FRAME_STYLE,
                'slider_entries': [], 'slider_index': 0,
                'slider_page': 'gallery',
                'fullscreen_media': None,
                'dir_expanded': {},
            })

    def scan_directory(self, dd: Path = None) -> Dict:
        dd = dd or Config.DATA_DIR
        res = {'total_media': 0, 'new_media': 0, 'updated_media': 0,
               'images_found': 0, 'videos_found': 0, 'people_found': 0, 'errors': []}
        try:
            if not dd.exists(): dd.mkdir(parents=True); return res
            skip = {'thumbnails', 'video_thumbnails', 'video_cache', 'metadata', 'database', 'exports'}
            pdirs = [d for d in dd.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name not in skip]
            if not pdirs:
                Config.create_sample_structure()
                pdirs = [d for d in dd.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name not in skip]
            res['people_found'] = len(pdirs)
            pb = st.progress(0)
            total = sum(1 for pd in pdirs for f in pd.iterdir() if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS)
            proc = 0
            for pdir in pdirs:
                dn = ' '.join(p.capitalize() for p in pdir.name.replace('-', ' ').replace('_', ' ').split())
                ep = self.db.get_person_by_folder(pdir.name)
                if not ep:
                    pp = PersonProfile(person_id=str(uuid.uuid4()), folder_name=pdir.name, display_name=dn,
                                       bio=f"Photos of {dn}", birth_date=None, relationship="Other",
                                       contact_info="", social_links={}, profile_image=None,
                                       created_at=datetime.datetime.now())
                    self.db.add_person(pp); pid = pp.person_id
                else:
                    pid = ep['person_id']
                for mf in pdir.iterdir():
                    if not mf.is_file() or mf.suffix.lower() not in Config.ALLOWED_EXTENSIONS: continue
                    try:
                        proc += 1; pb.progress(proc / max(total, 1))
                        cs = MediaMetadata._calculate_checksum(mf)
                        with sqlite3.connect(self.db.db_path) as conn:
                            if conn.execute('SELECT 1 FROM media WHERE checksum=?', (cs,)).fetchone():
                                res['updated_media'] += 1; continue
                        meta = MediaMetadata.from_file(mf)
                        th = vth = None
                        if meta.media_type == MediaType.IMAGE.value:
                            th = self.mp.create_thumbnail(mf); res['images_found'] += 1
                        else:
                            vth = self.mp.create_thumbnail(mf); res['videos_found'] += 1
                        self.db.add_media(meta, str(th) if th else None, str(vth) if vth else None)
                        ae = AlbumEntry(entry_id=str(uuid.uuid4()), media_id=meta.media_id, person_id=pid,
                                        caption=mf.stem.replace('_', ' ').title(),
                                        description=f"Media of {dn}", location="",
                                        date_taken=meta.created_date,
                                        tags=[dn.lower().replace(' ', '-'), meta.media_type, 'memory'],
                                        privacy_level='public', created_by='system',
                                        created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
                        self.db.add_album_entry(ae)
                        res['new_media'] += 1; res['total_media'] += 1
                    except Exception as e:
                        res['errors'].append(str(e))
            pb.empty(); self.cache.clear(); return res
        except Exception as e:
            res['errors'].append(str(e)); return res

    def get_person_stats(self, pid):
        def gen():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    c = conn.cursor()
                    mc = c.execute('SELECT COUNT(*)FROM album_entries WHERE person_id=?', (pid,)).fetchone()[0]
                    c.execute('''SELECT m.media_type,COUNT(*)FROM album_entries ae JOIN media m ON ae.media_id=m.media_id
                        WHERE ae.person_id=? GROUP BY m.media_type''', (pid,))
                    ic = vc = 0
                    for r in c.fetchall():
                        if r[0] == 'image': ic = r[1]
                        elif r[0] == 'video': vc = r[1]
                    cc = c.execute('''SELECT COUNT(DISTINCT c.comment_id)FROM comments c
                        JOIN album_entries ae ON c.entry_id=ae.entry_id WHERE ae.person_id=?''', (pid,)).fetchone()[0]
                    ar = c.execute('''SELECT AVG(r.rating_value)FROM ratings r
                        JOIN album_entries ae ON r.entry_id=ae.entry_id WHERE ae.person_id=?''', (pid,)).fetchone()[0] or 0.0
                    la = c.execute('SELECT MAX(created_at)FROM album_entries WHERE person_id=?', (pid,)).fetchone()[0]
                    return {'media_count': mc, 'image_count': ic, 'video_count': vc,
                            'comment_count': cc, 'avg_rating': float(ar), 'last_activity': la}
            except Exception:
                return {'media_count': 0, 'image_count': 0, 'video_count': 0,
                        'comment_count': 0, 'avg_rating': 0.0, 'last_activity': None}
        return self.cache.get_or_set(f"pstats_{pid}", gen)

    def add_to_favorites(self, eid):
        uid = st.session_state['user_id']
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute('INSERT OR IGNORE INTO user_favorites(user_id,entry_id)VALUES(?,?)', (uid, eid))
            conn.commit()
        st.session_state.favorites.add(eid)

    def remove_from_favorites(self, eid):
        uid = st.session_state['user_id']
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute('DELETE FROM user_favorites WHERE user_id=? AND entry_id=?', (uid, eid))
            conn.commit()
        st.session_state.favorites.discard(eid)

    def get_user_favorites(self):
        uid = st.session_state['user_id']
        with sqlite3.connect(self.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute('''SELECT ae.*,p.display_name,m.filename,m.media_type,
                m.thumbnail_path,m.video_thumbnail_path FROM user_favorites uf
                JOIN album_entries ae ON uf.entry_id=ae.entry_id
                JOIN people p ON ae.person_id=p.person_id JOIN media m ON ae.media_id=m.media_id
                WHERE uf.user_id=? ORDER BY uf.created_at DESC''', (uid,)).fetchall()]

    def add_comment_to_entry(self, eid, content, parent=None):
        if not content or not content.strip(): return False
        if len(content) > Config.MAX_COMMENT_LENGTH: return False
        cm = Comment(comment_id=str(uuid.uuid4()), entry_id=eid,
                     user_id=st.session_state['user_id'], username=st.session_state['username'],
                     content=content.strip(), created_at=datetime.datetime.now(),
                     is_edited=False, parent_comment_id=parent)
        self.db.add_comment(cm); self.cache.clear(f"comments_{eid}")
        return True

    def add_rating_to_entry(self, eid, val):
        if val < 1 or val > 5: return False
        r = Rating(rating_id=str(uuid.uuid4()), entry_id=eid, user_id=st.session_state['user_id'],
                   rating_value=val, created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
        self.db.add_rating(r); self.cache.clear(f"ratings_{eid}")
        return True

    def get_all_people_with_stats(self):
        def gen():
            people = self.db.get_all_people(); result = []
            for p in people:
                s = self.get_person_stats(p['person_id']); pw = {**p, **s}
                if p.get('profile_image'):
                    pi = Config.DATA_DIR / p['folder_name'] / p['profile_image']
                    if pi.exists(): pw['profile_image_data'] = self.mp.get_media_data_url(pi)
                result.append(pw)
            return result
        return self.cache.get_or_set("all_pstats", gen)

    def get_entries_by_person(self, pid, page=1, sq=None, mf='all'):
        def gen():
            off = (page - 1) * Config.ITEMS_PER_PAGE
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row; c = conn.cursor()
                conds, params = ["ae.person_id=?"], [pid]
                if sq:
                    sp = f'%{sq}%'; conds.append("(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)")
                    params.extend([sp, sp, sp])
                if mf != 'all': conds.append("m.media_type=?"); params.append(mf)
                w = " AND ".join(conds); params.extend([Config.ITEMS_PER_PAGE, off])
                c.execute(f'''SELECT ae.*,p.display_name,m.filename,m.media_type,m.thumbnail_path,
                    m.video_thumbnail_path,m.duration,m.filepath,
                    (SELECT AVG(rating_value)FROM ratings r WHERE r.entry_id=ae.entry_id)as avg_rating,
                    (SELECT COUNT(*)FROM comments c2 WHERE c2.entry_id=ae.entry_id)as comment_count
                    FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                    JOIN media m ON ae.media_id=m.media_id WHERE {w}
                    ORDER BY ae.created_at DESC LIMIT ? OFFSET ?''', params)
                entries = [dict(r) for r in c.fetchall()]
                c.execute(f'SELECT COUNT(*)FROM album_entries ae JOIN media m ON ae.media_id=m.media_id WHERE {w}',
                          params[:-2])
                total = c.fetchone()[0]
                return {'entries': entries, 'total_count': total,
                        'total_pages': max(1, math.ceil(total / Config.ITEMS_PER_PAGE)), 'current_page': page}
        return self.cache.get_or_set(f"ep_{pid}_p{page}_q{sq}_f{mf}", gen)

    def get_recent_entries(self, limit=10):
        return self.cache.get_or_set(f"recent_{limit}", lambda: self.db.get_recent_entries(limit))

    def get_top_rated_entries(self, limit=10):
        def gen():
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute('''SELECT ae.*,p.display_name,m.filename,m.media_type,
                    m.thumbnail_path,m.video_thumbnail_path,
                    (SELECT AVG(rating_value)FROM ratings r WHERE r.entry_id=ae.entry_id)as avg_rating,
                    (SELECT COUNT(*)FROM ratings r2 WHERE r2.entry_id=ae.entry_id)as rating_count
                    FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                    JOIN media m ON ae.media_id=m.media_id
                    WHERE ae.entry_id IN(SELECT r.entry_id FROM ratings r GROUP BY r.entry_id
                        HAVING AVG(r.rating_value)>=4.0)
                    ORDER BY avg_rating DESC LIMIT ?''', (limit,)).fetchall()]
        return self.cache.get_or_set(f"top_{limit}", gen)

    def get_entry_with_details(self, eid):
        def gen():
            entry = self.db.get_entry_details(eid)
            if not entry: return None
            entry['comments'] = self.db.get_entry_comments(eid)
            avg, cnt = self.db.get_entry_ratings(eid)
            entry['avg_rating'] = avg; entry['rating_count'] = cnt
            if entry.get('filepath'):
                mp = Config.DATA_DIR / entry['filepath']
                if mp.exists():
                    if entry['media_type'] == MediaType.IMAGE.value:
                        entry['media_data_url'] = self.mp.get_hd_data_url(mp)
                        entry['media_path'] = str(mp)
                    else:
                        entry['media_path'] = str(mp)
            for k in ['thumbnail_path', 'video_thumbnail_path']:
                if entry.get(k):
                    tp = Path(entry[k])
                    if tp.exists():
                        entry['thumbnail_data_url'] = self.mp.get_media_data_url(tp); break
            uid = st.session_state['user_id']
            with sqlite3.connect(self.db.db_path) as conn:
                c = conn.cursor()
                c.execute('SELECT 1 FROM user_favorites WHERE user_id=? AND entry_id=?', (uid, eid))
                entry['is_favorited'] = c.fetchone() is not None
                c.execute('SELECT rating_value FROM ratings WHERE user_id=? AND entry_id=?', (uid, eid))
                ur = c.fetchone(); entry['user_rating'] = ur[0] if ur else 0
            return entry
        return self.cache.get_or_set(f"entry_{eid}", gen)

    def stream_video(self, vp):
        try:
            vp = Path(vp)
            if not vp.exists(): st.error("Video not found"); return
            vd = self.cache.get_video(vp)
            if not vd:
                vd = self.mp.prepare_video_stream(vp)
                if vd: self.cache.set_video(vp, vd)
            if vd:
                mt, _ = mimetypes.guess_type(str(vp))
                st.video(vd, format=mt or "video/mp4")
            else: st.error("Could not load video")
        except Exception as e:
            st.error(f"Video error: {e}")

    def get_directory_tree(self) -> Dict[str, List[Dict]]:
        tree = {}
        people = self.db.get_all_people()
        for p in people:
            pid = p['person_id']
            entries = self.db.get_all_entries_with_details(pid)
            tree[p['folder_name']] = {
                'person_id': pid, 'display_name': p['display_name'],
                'entries': entries, 'image_count': sum(1 for e in entries if e.get('media_type') == MediaType.IMAGE.value),
                'video_count': sum(1 for e in entries if e.get('media_type') == MediaType.VIDEO.value),
            }
        return tree


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PhotoVideoAlbumApp:
    def __init__(self):
        self.mgr = AlbumManager()
        self.setup_page_config()
        self.check_init()

    def setup_page_config(self):
        st.set_page_config(page_title=Config.APP_NAME, page_icon="🎬📸", layout="wide",
                           initial_sidebar_state="expanded",
                           menu_items={'About': f"# {Config.APP_NAME} v{Config.VERSION}"})

    def check_init(self):
        try: Config.init_directories(); self.initialized = True
        except Exception as e: st.error(f"Init error: {e}"); self.initialized = False

    @property
    def fs(self): return st.session_state.get('frame_style', Config.DEFAULT_FRAME_STYLE)

    def _thumb_url(self, entry):
        for k in ['thumbnail_path', 'video_thumbnail_path']:
            if entry.get(k):
                tp = Path(entry[k])
                if tp.exists(): return self.mgr.mp.get_media_data_url(tp)
        return None

    def _hd_url(self, entry):
        if entry.get('media_type') == MediaType.IMAGE.value and entry.get('filepath'):
            mp = Config.DATA_DIR / entry['filepath']
            if mp.exists(): return self.mgr.mp.get_hd_data_url(mp)
        return self._thumb_url(entry)

    # ── SIDEBAR with DIRECTORY PANEL ──────────────────────────────────
    def render_sidebar(self):
        with st.sidebar:
            st.title(f"🎬📸 {Config.APP_NAME}")
            st.caption(f"v{Config.VERSION}")
            st.divider()

            c1, c2 = st.columns([1, 3])
            with c1: st.markdown("👤")
            with c2:
                st.markdown(f"**{st.session_state['username']}**")
                st.caption(st.session_state['user_role'].title())

            st.divider()
            st.subheader("🧭 Navigation")
            nav = {"🏠 Dashboard": "dashboard", "📁 Media Gallery": "gallery",
                   "⭐ Favorites": "favorites", "🎬 Videos": "videos",
                   "📸 Photos": "photos", "🔍 Search": "search",
                   "📊 Statistics": "statistics", "⚙️ Settings": "settings",
                   "📤 Import/Export": "import_export"}
            for label, key in nav.items():
                if st.button(label, use_container_width=True, key=f"nav_{key}"):
                    st.session_state['current_page'] = key
                    st.session_state['selected_person'] = None
                    st.rerun()

            st.divider()
            if st.button("🔄 Scan Directory", use_container_width=True):
                with st.spinner("Scanning…"):
                    r = self.mgr.scan_directory()
                    if r['new_media'] > 0:
                        st.success(f"Found {r['new_media']} new media!")
                    st.rerun()
            if st.button("🗑️ Clear Cache", use_container_width=True):
                self.mgr.cache.clear(); st.success("Cleared!"); st.rerun()

            st.divider()
            st.subheader("🖼️ Frame Style")
            fsi = st.selectbox("Frame", Config.FRAME_STYLES,
                               index=Config.FRAME_STYLES.index(self.fs), key="fs_sel")
            if fsi != self.fs:
                st.session_state['frame_style'] = fsi; st.rerun()

            # ── DIRECTORY TREE PANEL ──────────────────────────────────
            st.divider()
            st.subheader("📂 Directories")
            tree = self.mgr.get_directory_tree()

            for folder, info in tree.items():
                is_expanded = st.session_state.get('dir_expanded', {}).get(folder, False)
                is_active = st.session_state.get('selected_person') == info['person_id']

                dir_label = f"{'📂' if is_expanded else '📁'} {info['display_name']}"
                count_label = f"📸{info['image_count']} 🎬{info['video_count']}"

                dir_col1, dir_col2 = st.columns([4, 1])
                with dir_col1:
                    if st.button(dir_label, key=f"dir_{folder}", use_container_width=True):
                        st.session_state['selected_person'] = info['person_id']
                        st.session_state['current_page'] = 'gallery'
                        if folder not in st.session_state.get('dir_expanded', {}):
                            st.session_state.setdefault('dir_expanded', {})[folder] = True
                        else:
                            st.session_state['dir_expanded'][folder] = not st.session_state.get('dir_expanded', {}).get(folder, False)
                        st.rerun()
                with dir_col2:
                    st.caption(count_label)

                # Show files inside if expanded
                if st.session_state.get('dir_expanded', {}).get(folder, False) and info['entries']:
                    for entry in info['entries'][:20]:
                        icon = "🎬" if entry.get('media_type') == MediaType.VIDEO.value else "🖼️"
                        fname = entry.get('caption', entry.get('filename', ''))
                        if st.button(f"  {icon} {fname[:30]}", key=f"df_{entry['entry_id']}",
                                     use_container_width=True):
                            st.session_state['selected_media'] = entry['entry_id']
                            st.session_state['current_page'] = 'media_detail'
                            st.rerun()
                    if len(info['entries']) > 20:
                        st.caption(f"  … +{len(info['entries'])-20} more")

            # Global stats
            st.divider()
            people = self.mgr.db.get_all_people()
            ti = tv = 0
            for p in people:
                s = self.mgr.get_person_stats(p['person_id'])
                ti += s['image_count']; tv += s['video_count']
            st.metric("People", len(people))
            sc1, sc2 = st.columns(2)
            with sc1: st.metric("Images", ti)
            with sc2: st.metric("Videos", tv)

    # ── DASHBOARD ─────────────────────────────────────────────────────
    def render_dashboard(self):
        render_breadcrumb([("🏠 Home", "dashboard")])
        st.title("📊 Dashboard")
        people = self.mgr.db.get_all_people()
        tm = ti = tv = 0
        for p in people:
            s = self.mgr.get_person_stats(p['person_id'])
            tm += s['media_count']; ti += s['image_count']; tv += s['video_count']
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("👥 People", len(people))
        with c2: st.metric("📸🎬 Media", tm)
        with c3:
            r = self.mgr.get_recent_entries(1)
            st.metric("🕐 Last", r[0].get('caption', 'N/A')[:15] if r else 'None')
        with c4:
            t = self.mgr.get_top_rated_entries(1)
            st.metric("⭐ Top", f"{t[0].get('avg_rating',0):.1f}/5" if t else "N/A")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Distribution")
            if tm > 0:
                st.bar_chart(pd.DataFrame({'Type': ['Images', 'Videos'], 'Count': [ti, tv]}).set_index('Type'))
        with c2:
            st.subheader("⚡ Quick Stats")
            ca, cb = st.columns(2)
            with ca: st.metric("Images", ti); st.metric("Videos", tv)
            with cb: st.metric("People", len(people))
        st.divider()
        st.subheader("📅 Recent")
        for e in self.mgr.get_recent_entries(5):
            with st.container():
                ca, cb = st.columns([1, 4])
                with ca:
                    tu = self._thumb_url(e)
                    if tu:
                        st.markdown(FrameRenderer.wrap_thumbnail(tu, e.get('caption', ''), self.fs,
                            e.get('media_type') == MediaType.VIDEO.value, e.get('duration')), unsafe_allow_html=True)
                    else:
                        st.markdown("🎬" if e.get('media_type') == MediaType.VIDEO.value else "📸")
                with cb:
                    st.markdown(f"**{e.get('caption','Untitled')}**")
                    st.markdown(UIComponents.media_type_badge(e.get('media_type', 'image')), unsafe_allow_html=True)
                    st.caption(f"👤 {e.get('display_name','')}")
                st.divider()

    # ── GALLERY ───────────────────────────────────────────────────────
    def render_gallery_page(self):
        sp = st.session_state.get('selected_person')
        if sp:
            people = self.mgr.db.get_all_people()
            sel = next((p for p in people if p['person_id'] == sp), None)
            if sel:
                render_breadcrumb([("🏠 Home", "dashboard"), ("📁 Gallery", "gallery"),
                                   (sel['display_name'], "gallery")])
        else:
            render_breadcrumb([("🏠 Home", "dashboard"), ("📁 Gallery", "gallery")])
        st.title("📁 Media Gallery")

        with st.container():
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1: sq = st.text_input("Search…", key="gsearch")
            with c2:
                mf_opts = ["All", "Image", "Video"]
                cur = st.session_state.get('media_filter', 'all')
                di = {'all': 0, 'image': 1, 'video': 2}.get(cur, 0)
                mfd = st.selectbox("Type", mf_opts, index=di, key="mfd")
                st.session_state['media_filter'] = {'All': 'all', 'Image': 'image', 'Video': 'video'}[mfd]
            with c3:
                vm = st.selectbox("View", ["Grid", "List", "Slider"], key="vm")

        mf = st.session_state['media_filter']
        page = st.session_state.get('gallery_page', 1)

        if sp:
            data = self.mgr.get_entries_by_person(sp, page, sq, mf)
            all_entries = self.mgr.db.get_all_entries_with_details(sp, mf)
        else:
            data = self._get_all_entries_page(page, sq, mf)
            all_entries = self.mgr.db.get_all_entries_with_details(None, mf)

        if vm == "Slider":
            self._render_slider(all_entries)
        elif vm == "Grid":
            cols = st.columns(4)
            for idx, e in enumerate(data['entries']):
                with cols[idx % 4]:
                    self._render_gallery_item(e)
        else:
            for e in data['entries']:
                self._render_gallery_item_list(e)

        if vm != "Slider" and data.get('total_pages', 1) > 1:
            st.divider()
            self._render_pagination(data, page)

    def _get_all_entries_page(self, page, sq, mf):
        ipp = Config.ITEMS_PER_PAGE; off = (page - 1) * ipp
        with sqlite3.connect(self.mgr.db.db_path) as conn:
            conn.row_factory = sqlite3.Row; c = conn.cursor()
            conds, params = [], []
            if sq:
                sp = f'%{sq}%'; conds.append("(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)")
                params.extend([sp, sp, sp])
            if mf != 'all': conds.append("m.media_type=?"); params.append(mf)
            w = " AND ".join(conds) if conds else "1=1"
            params.extend([ipp, off])
            c.execute(f'''SELECT ae.*,p.display_name,m.filename,m.media_type,m.thumbnail_path,
                m.video_thumbnail_path,m.duration,m.filepath,
                (SELECT AVG(rating_value)FROM ratings r WHERE r.entry_id=ae.entry_id)as avg_rating
                FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                JOIN media m ON ae.media_id=m.media_id WHERE {w}
                ORDER BY ae.created_at DESC LIMIT ? OFFSET ?''', params)
            entries = [dict(r) for r in c.fetchall()]
            c.execute(f'SELECT COUNT(*)FROM album_entries ae JOIN media m ON ae.media_id=m.media_id WHERE {w}',
                      params[:-2])
            total = c.fetchone()[0]
            return {'entries': entries, 'total_count': total,
                    'total_pages': max(1, math.ceil(total / ipp)), 'current_page': page}

    def _render_pagination(self, data, page):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            pnums = []
            for i in range(1, data['total_pages'] + 1):
                if i == 1 or i == data['total_pages'] or abs(i - page) <= 2:
                    pnums.append(i)
                elif pnums[-1] != "...": pnums.append("...")
            btns = st.columns(len(pnums) + 2)
            with btns[0]:
                if page > 1 and st.button("◀", key="pp"):
                    st.session_state.gallery_page = page - 1; st.rerun()
            for bi, pn in enumerate(pnums, 1):
                with btns[bi]:
                    if pn == "...": st.markdown("…")
                    elif pn == page: st.markdown(f"**{pn}**")
                    elif st.button(str(pn), key=f"pg{pn}"):
                        st.session_state.gallery_page = pn; st.rerun()
            with btns[-1]:
                if page < data['total_pages'] and st.button("▶", key="np"):
                    st.session_state.gallery_page = page + 1; st.rerun()

    # ── HD SLIDER VIEW ────────────────────────────────────────────────
    def _render_slider(self, entries):
        if not entries:
            st.info("No media to display in slider.")
            return

        image_entries = [e for e in entries if e.get('media_type') == MediaType.IMAGE.value]
        if not image_entries:
            st.info("No images for slider view.")
            return

        idx = st.session_state.get('slider_index', 0)
        if idx >= len(image_entries): idx = 0; st.session_state['slider_index'] = 0
        entry = image_entries[idx]

        # Caption / counter
        st.markdown(f'<div class="slider-counter">{idx+1} of {len(image_entries)} — '
                    f'{entry.get("caption","Untitled")}</div>', unsafe_allow_html=True)

        # ── Navigation + HD image ────────────────────────────────────
        nav_cols = st.columns([1, 14, 1])

        with nav_cols[0]:
            st.markdown("<div style='display:flex;align-items:center;height:100%;justify-content:center;'>",
                        unsafe_allow_html=True)
            if idx > 0:
                if st.button("⬅", key="sl_prev", use_container_width=True):
                    st.session_state['slider_index'] = idx - 1; st.rerun()
            else:
                st.markdown("<div style='opacity:0.3;text-align:center;font-size:24px;'>⬅</div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with nav_cols[1]:
            hd_url = self._hd_url(entry)
            if hd_url:
                st.markdown(FrameRenderer.wrap_detail(hd_url, self.fs), unsafe_allow_html=True)
            else:
                st.warning("Image not available")

            # Info bar under image
            info_c1, info_c2, info_c3 = st.columns([2, 2, 1])
            with info_c1:
                st.markdown(f"**{entry.get('caption','Untitled')}**")
                st.caption(f"👤 {entry.get('display_name','')}")
            with info_c2:
                if entry.get('filepath'):
                    mp = Config.DATA_DIR / entry['filepath']
                    if mp.exists():
                        with open(mp, 'rb') as f:
                            st.download_button("💾 Download Original", data=f.read(),
                                               file_name=mp.name, key=f"sldl_{idx}", use_container_width=True)
            with info_c3:
                if st.button("👁️ Detail", key=f"slview_{idx}", use_container_width=True):
                    st.session_state['selected_media'] = entry['entry_id']
                    st.session_state['current_page'] = 'media_detail'
                    st.rerun()

        with nav_cols[2]:
            st.markdown("<div style='display:flex;align-items:center;height:100%;justify-content:center;'>",
                        unsafe_allow_html=True)
            if idx < len(image_entries) - 1:
                if st.button("➡", key="sl_next", use_container_width=True):
                    st.session_state['slider_index'] = idx + 1; st.rerun()
            else:
                st.markdown("<div style='opacity:0.3;text-align:center;font-size:24px;'>➡</div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── CLICKABLE THUMBNAIL STRIP ────────────────────────────────
        st.markdown("---")
        self._render_thumb_strip(image_entries, idx)

    def _render_thumb_strip(self, entries, current_idx):
        """Render a scrollable clickable thumbnail strip at the bottom."""
        total = len(entries)
        window_size = Config.SLIDER_THUMB_COUNT

        # Calculate visible window centered on current
        half = window_size // 2
        start = max(0, current_idx - half)
        end = min(total, start + window_size)
        start = max(0, end - window_size)

        visible = entries[start:end]

        # Scroll controls
        strip_cols = st.columns([1, 20, 1])
        with strip_cols[0]:
            if start > 0:
                if st.button("◀◀", key="thumb_scroll_left"):
                    new_idx = max(0, current_idx - window_size)
                    st.session_state['slider_index'] = new_idx; st.rerun()

        with strip_cols[1]:
            # Build HTML thumbnail strip
            thumb_html = '<div class="thumb-strip">'
            for i, e in enumerate(visible):
                real_idx = start + i
                tu = self._thumb_url(e)
                active = " active" if real_idx == current_idx else ""
                if tu:
                    thumb_html += f'''<div class="thumb-item{active}" title="{e.get('caption','')}"
                        onclick="document.getElementById('thumb_btn_{real_idx}').click()">
                        <img src="{tu}" alt="{e.get('caption','')}"></div>'''
                else:
                    thumb_html += f'''<div class="thumb-item{active}"
                        onclick="document.getElementById('thumb_btn_{real_idx}').click()">
                        <div style="width:72px;height:54px;background:#f0f0f0;display:flex;align-items:center;
                        justify-content:center;font-size:20px;">🖼️</div></div>'''
            thumb_html += '</div>'
            st.markdown(thumb_html, unsafe_allow_html=True)

            # Hidden buttons for each thumbnail (triggered by JS onclick)
            btn_cols = st.columns(len(visible))
            for i, e in enumerate(visible):
                real_idx = start + i
                with btn_cols[i]:
                    # Use a very small button that acts as a click target
                    if st.button("·", key=f"thumb_btn_{real_idx}",
                                 help=f"{e.get('caption','')} ({real_idx+1}/{total})"):
                        st.session_state['slider_index'] = real_idx; st.rerun()

        with strip_cols[2]:
            if end < total:
                if st.button("▶▶", key="thumb_scroll_right"):
                    new_idx = min(total - 1, current_idx + window_size)
                    st.session_state['slider_index'] = new_idx; st.rerun()

        # Page indicator
        st.markdown(f'<div class="slider-counter">Showing {start+1}–{end} of {total}</div>',
                    unsafe_allow_html=True)

    # ── GALLERY ITEMS ─────────────────────────────────────────────────
    def _render_gallery_item(self, entry):
        with st.container(border=True):
            st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')), unsafe_allow_html=True)
            tu = self._thumb_url(entry)
            is_vid = entry.get('media_type') == MediaType.VIDEO.value
            if tu:
                st.markdown(FrameRenderer.wrap_thumbnail(tu, entry.get('caption', ''), self.fs,
                    is_vid, entry.get('duration')), unsafe_allow_html=True)
            else:
                st.markdown("🎬" if is_vid else "📸")
            st.markdown(f"**{entry.get('caption','Untitled')}**")
            st.caption(f"👤 {entry.get('display_name','')}")
            if entry.get('avg_rating'):
                st.markdown(UIComponents.rating_stars(entry['avg_rating'], size=15), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("👁️", key=f"gv_{entry['entry_id']}", use_container_width=True):
                    st.session_state['selected_media'] = entry['entry_id']
                    st.session_state['current_page'] = 'media_detail'; st.rerun()
            with c2:
                fav = entry['entry_id'] in st.session_state.get('favorites', set())
                if fav:
                    if st.button("⭐", key=f"gf_{entry['entry_id']}", use_container_width=True):
                        self.mgr.remove_from_favorites(entry['entry_id']); st.rerun()
                else:
                    if st.button("☆", key=f"guf_{entry['entry_id']}", use_container_width=True):
                        self.mgr.add_to_favorites(entry['entry_id']); st.rerun()
            with c3:
                if not is_vid and entry.get('filepath'):
                    mp = Config.DATA_DIR / entry['filepath']
                    if mp.exists():
                        with open(mp, 'rb') as f:
                            st.download_button("💾", data=f.read(), file_name=mp.name,
                                               key=f"gdl_{entry['entry_id']}", use_container_width=True)

    def _render_gallery_item_list(self, entry):
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                tu = self._thumb_url(entry)
                is_vid = entry.get('media_type') == MediaType.VIDEO.value
                if tu:
                    st.markdown(FrameRenderer.wrap_thumbnail(tu, '', self.fs, is_vid, entry.get('duration')),
                                unsafe_allow_html=True)
                else:
                    st.markdown("🎬" if is_vid else "📸")
            with c2:
                st.markdown(f"### {entry.get('caption','Untitled')}")
                st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')), unsafe_allow_html=True)
                st.caption(f"👤 {entry.get('display_name','')}")
            with c3:
                if entry.get('avg_rating'):
                    st.markdown(UIComponents.rating_stars(entry['avg_rating']), unsafe_allow_html=True)
                if st.button("View", key=f"glv_{entry['entry_id']}", use_container_width=True):
                    st.session_state['selected_media'] = entry['entry_id']
                    st.session_state['current_page'] = 'media_detail'; st.rerun()

    # ── MEDIA DETAIL (HD + Frame) ────────────────────────────────────
    def render_media_detail_page(self):
        eid = st.session_state.get('selected_media')
        if not eid:
            st.error("No media selected")
            if st.button("Back"): st.session_state['current_page'] = 'gallery'; st.rerun()
            return
        entry = self.mgr.get_entry_with_details(eid)
        if not entry:
            st.error("Not found")
            if st.button("Back"): st.session_state['current_page'] = 'gallery'; st.rerun()
            return

        render_breadcrumb([("🏠 Home", "dashboard"), ("📁 Gallery", "gallery"),
                           (entry.get('caption', 'Detail'), "media_detail")])

        if st.button("← Back"):
            st.session_state['current_page'] = st.session_state.get('slider_page', 'gallery'); st.rerun()

        c1, c2 = st.columns([2, 1])
        with c1:
            if entry.get('media_type') == MediaType.IMAGE.value:
                if entry.get('media_data_url'):
                    st.markdown(FrameRenderer.wrap_detail(entry['media_data_url'], self.fs), unsafe_allow_html=True)

                    # Slider entry: navigate through all images for this person
                    fc1, fc2, fc3 = st.columns([1, 1, 1])
                    with fc1:
                        if st.button("🔍 Full Screen"):
                            st.session_state['fullscreen_media'] = entry['media_data_url']; st.rerun()
                    with fc2:
                        if entry.get('media_path'):
                            mp = Path(entry['media_path'])
                            if mp.exists():
                                with open(mp, 'rb') as f:
                                    st.download_button("💾 Download HD", data=f.read(),
                                                       file_name=mp.name, use_container_width=True)
                    with fc3:
                        # Open slider from detail
                        pid = entry.get('person_id')
                        all_e = self.mgr.db.get_all_entries_with_details(pid, 'image')
                        img_entries = [e for e in all_e if e.get('media_type') == MediaType.IMAGE.value]
                        if len(img_entries) > 1:
                            cur = next((i for i, e in enumerate(img_entries) if e['entry_id'] == eid), 0)
                            if st.button("🎞️ Slideshow", use_container_width=True):
                                st.session_state['slider_entries'] = img_entries
                                st.session_state['slider_index'] = cur
                                st.session_state['slider_page'] = 'media_detail'
                                st.session_state['current_page'] = 'gallery'
                                st.session_state['view_mode_override'] = 'Slider'
                                st.rerun()
                else:
                    st.error("Image unavailable")
            elif entry.get('media_type') == MediaType.VIDEO.value:
                st.subheader("🎬 Video Player")
                if entry.get('media_path'):
                    self.mgr.stream_video(entry['media_path'])
                    vi = self.mgr.mp.get_video_info(Path(entry['media_path']))
                    va, vb, vc = st.columns(3)
                    with va:
                        m, s = int(vi['duration'] // 60), int(vi['duration'] % 60)
                        st.metric("Duration", f"{m}:{s:02d}")
                    with vb: st.metric("Resolution", f"{vi['dimensions'][0]}×{vi['dimensions'][1]}")
                    with vc: st.metric("FPS", f"{vi['frame_rate']:.1f}")

        with c2:
            st.title(entry.get('caption', 'Untitled'))
            st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')), unsafe_allow_html=True)
            st.markdown(f"👤 **{entry.get('display_name','')}**")
            avg = entry.get('avg_rating', 0); rc = entry.get('rating_count', 0)
            st.markdown(UIComponents.rating_stars(avg), unsafe_allow_html=True)
            st.caption(f"{rc} ratings")
            st.subheader("Rate")
            rcols = st.columns(5)
            for i in range(1, 6):
                with rcols[i - 1]:
                    if st.button(f"{i}⭐", key=f"r{i}", use_container_width=True):
                        self.mgr.add_rating_to_entry(eid, i); st.rerun()
            if entry.get('is_favorited'):
                if st.button("⭐ Unfavorite", use_container_width=True):
                    self.mgr.remove_from_favorites(eid); st.rerun()
            else:
                if st.button("☆ Favorite", use_container_width=True):
                    self.mgr.add_to_favorites(eid); st.rerun()

            with st.expander("📊 Info"):
                for lbl, val in [("Description", entry.get('description')), ("Location", entry.get('location')),
                                 ("Date", entry.get('date_taken')),
                                 ("Size", f"{entry['file_size']/(1024*1024):.2f}MB" if entry.get('file_size') else None),
                                 ("Format", entry.get('format')),
                                 ("Dims", f"{entry['width']}×{entry['height']}" if entry.get('width') else None)]:
                    if val: st.markdown(f"**{lbl}:** {val}")
                if entry.get('tags'):
                    st.markdown(UIComponents.tag_badges(
                        entry['tags'] if isinstance(entry['tags'], list) else entry['tags'].split(','), 10),
                        unsafe_allow_html=True)

        # Fullscreen overlay
        if st.session_state.get('fullscreen_media'):
            st.markdown(f"""<div class="fullscreen-overlay" onclick="this.style.display='none'">
                <img src="{st.session_state['fullscreen_media']}"></div>""", unsafe_allow_html=True)
            if st.button("✖ Close Full Screen"):
                st.session_state['fullscreen_media'] = None; st.rerun()

        # Comments
        st.divider()
        st.subheader("💬 Comments")
        with st.form("cmt_form"):
            ct = st.text_area("Comment…", height=100, max_chars=Config.MAX_COMMENT_LENGTH)
            if st.form_submit_button("Post") and ct.strip():
                if self.mgr.add_comment_to_entry(eid, ct.strip()): st.rerun()
        for c in entry.get('comments', []):
            with st.container(border=True):
                ca, cb = st.columns([1, 4])
                with ca: st.markdown(f"**{c.get('username','Anon')}**"); st.caption(c.get('created_at', ''))
                with cb: st.markdown(c.get('content', ''))

    # ── VIDEOS PAGE ───────────────────────────────────────────────────
    def render_videos_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("🎬 Videos", "videos")])
        st.title("🎬 Video Library")
        vids = self.mgr.db.get_media_by_type(MediaType.VIDEO.value)
        if not vids: st.info("No videos found."); return
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Videos", len(vids))
        with c2:
            td = sum(v.get('duration', 0) for v in vids)
            st.metric("Duration", f"{int(td//3600)}h{int((td%3600)//60)}m")
        with c3:
            ad = td / len(vids) if vids else 0
            st.metric("Avg", f"{int(ad//60)}:{int(ad%60):02d}")
        with c4: st.metric("Size", f"{sum(v.get('file_size',0) for v in vids)/(1024*1024):.1f}MB")
        st.divider()
        cols = st.columns(4)
        for idx, v in enumerate(vids):
            with cols[idx % 4]:
                with st.container(border=True):
                    if v.get('video_thumbnail_path'):
                        tp = Path(v['video_thumbnail_path'])
                        if tp.exists():
                            du = self.mgr.mp.get_media_data_url(tp)
                            st.markdown(FrameRenderer.wrap_thumbnail(du, '', self.fs, True, v.get('duration')),
                                        unsafe_allow_html=True)
                    st.markdown(f"**{v.get('filename','')}**")
                    if v.get('duration'):
                        m, s = int(v['duration'] // 60), int(v['duration'] % 60)
                        st.caption(f"⏱️ {m:02d}:{s:02d}")
                    with sqlite3.connect(self.mgr.db.db_path) as conn:
                        r = conn.execute('SELECT entry_id FROM album_entries WHERE media_id=?', (v['media_id'],)).fetchone()
                    if r and st.button("Play", key=f"vp_{v['media_id']}", use_container_width=True):
                        st.session_state['selected_media'] = r[0]
                        st.session_state['current_page'] = 'media_detail'; st.rerun()

    # ── PHOTOS PAGE ───────────────────────────────────────────────────
    def render_photos_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("📸 Photos", "photos")])
        st.title("📸 Photo Library")
        imgs = self.mgr.db.get_media_by_type(MediaType.IMAGE.value)
        if not imgs: st.info("No photos found."); return
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Photos", len(imgs))
        with c2: st.metric("Size", f"{sum(i.get('file_size',0) for i in imgs)/(1024*1024):.1f}MB")
        with c3: st.metric("Avg", f"{sum(i.get('file_size',0) for i in imgs)/(1024*1024*len(imgs)):.1f}MB" if imgs else "0")
        st.divider()
        cols = st.columns(4)
        for idx, img in enumerate(imgs):
            with cols[idx % 4]:
                with st.container(border=True):
                    if img.get('thumbnail_path'):
                        tp = Path(img['thumbnail_path'])
                        if tp.exists():
                            du = self.mgr.mp.get_media_data_url(tp)
                            st.markdown(FrameRenderer.wrap_thumbnail(du, img.get('filename', ''), self.fs),
                                        unsafe_allow_html=True)
                    st.markdown(f"**{img.get('filename','')}**")
                    if img.get('width'): st.caption(f"📐 {img['width']}×{img['height']}")
                    with sqlite3.connect(self.mgr.db.db_path) as conn:
                        r = conn.execute('SELECT entry_id FROM album_entries WHERE media_id=?', (img['media_id'],)).fetchone()
                    if r and st.button("View", key=f"ppv_{img['media_id']}", use_container_width=True):
                        st.session_state['selected_media'] = r[0]
                        st.session_state['current_page'] = 'media_detail'; st.rerun()

    # ── FAVORITES ─────────────────────────────────────────────────────
    def render_favorites_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("⭐ Favorites", "favorites")])
        st.title("⭐ Favorites")
        favs = self.mgr.get_user_favorites()
        if not favs: st.info("No favorites yet."); return
        cols = st.columns(4)
        for idx, e in enumerate(favs):
            with cols[idx % 4]: self._render_gallery_item(e)

    # ── SEARCH ────────────────────────────────────────────────────────
    def render_search_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("🔍 Search", "search")])
        st.title("🔍 Search")
        c1, c2 = st.columns([3, 1])
        with c1: sq = st.text_input("Search", key="gsearch2")
        with c2: si = st.selectbox("In", ["All", "Captions", "Tags", "People"])
        if sq:
            results = self.mgr.db.search_entries(sq)
            if si == "People":
                for p in self.mgr.db.get_all_people():
                    if sq.lower() in p['display_name'].lower():
                        results.extend(self.mgr.db.get_all_entries_with_details(p['person_id']))
            seen = set(); unique = []
            for r in results:
                if r['entry_id'] not in seen: seen.add(r['entry_id']); unique.append(r)
            st.subheader(f"{len(unique)} results")
            for e in unique: self._render_gallery_item_list(e)
        else:
            st.info("Enter a search term")

    # ── STATISTICS ────────────────────────────────────────────────────
    def render_statistics_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("📊 Statistics", "statistics")])
        st.title("📊 Statistics")
        ap = self.mgr.get_all_people_with_stats()
        if not ap: st.info("No data."); return
        tm = sum(p['media_count'] for p in ap)
        ti = sum(p['image_count'] for p in ap)
        tv = sum(p['video_count'] for p in ap)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("People", len(ap))
        with c2: st.metric("Media", tm)
        with c3: st.metric("Comments", sum(p['comment_count'] for p in ap))
        ars = [p['avg_rating'] for p in ap if p['avg_rating'] > 0]
        with c4: st.metric("Avg Rating", f"{sum(ars)/len(ars):.1f}" if ars else "N/A")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribution")
            st.bar_chart(pd.DataFrame({'Type': ['Images', 'Videos'], 'Count': [ti, tv]}).set_index('Type'))
        with c2:
            st.subheader("Per Person")
            st.bar_chart(pd.DataFrame({'Person': [p['display_name'] for p in ap],
                                       'Media': [p['media_count'] for p in ap]}).set_index('Person'))
        st.divider()
        df = pd.DataFrame([{'Name': p['display_name'], 'Total': p['media_count'],
                             'Img': p['image_count'], 'Vid': p['video_count'],
                             'Rating': f"{p['avg_rating']:.1f}"} for p in ap])
        st.dataframe(df, use_container_width=True)
        if st.button("Export CSV"):
            st.download_button("Download", df.to_csv(index=False), "stats.csv", "text/csv")

    # ── SETTINGS ──────────────────────────────────────────────────────
    def render_settings_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("⚙️ Settings", "settings")])
        st.title("⚙️ Settings")
        tabs = st.tabs(["App", "User", "DB", "Advanced"])
        with tabs[0]:
            st.subheader("Frame Style")
            fs = st.selectbox("Frame", Config.FRAME_STYLES, index=Config.FRAME_STYLES.index(self.fs))
            if st.button("Apply Frame"):
                st.session_state['frame_style'] = fs; st.success("Updated!"); st.rerun()
            with st.expander("Directories"):
                for l, p in [("Data", Config.DATA_DIR), ("Thumbs", Config.THUMBNAIL_DIR),
                             ("DB", Config.DB_DIR)]: st.info(f"**{l}:** {p}")
                if st.button("Create Dirs"): Config.init_directories(); st.success("Done!")
        with tabs[1]:
            nu = st.text_input("Username", value=st.session_state['username'])
            nr = st.selectbox("Role", [r.value for r in UserRoles],
                              index=[r.value for r in UserRoles].index(st.session_state['user_role']))
            if st.button("Update User"):
                st.session_state['username'] = nu; st.session_state['user_role'] = nr; st.success("Updated!")
        with tabs[2]:
            if st.button("Rebuild DB"): self.mgr.db._init_database(); st.success("Done!")
            if st.button("Optimize"):
                with self.mgr.db.get_connection() as conn:
                    conn.execute("VACUUM"); conn.execute("ANALYZE")
                st.success("Optimized!")
            if st.button("Clear All Data"):
                if st.checkbox("Delete everything"):
                    if self.mgr.db.db_path.exists(): os.remove(self.mgr.db.db_path)
                    self.mgr.db = DatabaseManager(); self.mgr.cache.clear(); st.success("Cleared!"); st.rerun()
        with tabs[3]:
            st.info(f"Cache: {len(self.mgr.cache._cache)} | Video: {len(self.mgr.cache._vc)}")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Clear Cache"): self.mgr.cache.clear(); st.success("Done!")
            with c2:
                if st.button("Clear Video Cache"): self.mgr.cache.clear_video_cache(); st.success("Done!")
            if st.button("Session State"): st.write(st.session_state)

    # ── IMPORT/EXPORT ─────────────────────────────────────────────────
    def render_import_export_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("📤 Import/Export", "import_export")])
        st.title("📤 Import/Export")
        tabs = st.tabs(["Export", "Import", "Backup"])
        with tabs[0]:
            ef = st.selectbox("Format", ["CSV", "JSON", "Excel"], key="ef")
            if st.button("Export", type="primary"):
                with sqlite3.connect(self.mgr.db.db_path) as conn:
                    df = pd.read_sql_query('''SELECT ae.caption,ae.description,ae.location,ae.tags,
                        p.display_name as person,m.filename,m.media_type,m.file_size,m.format
                        FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                        JOIN media m ON ae.media_id=m.media_id ORDER BY ae.created_at DESC''', conn)
                    buf = io.BytesIO(); ext = ef.lower()
                    if ext == 'csv': buf.write(df.to_csv(index=False).encode())
                    elif ext == 'json': buf.write(df.to_json(orient='records', indent=2).encode())
                    else:
                        with pd.ExcelWriter(buf, engine='openpyxl') as w: df.to_excel(w, index=False)
                        ext = 'xlsx'
                    st.download_button("Download", buf.getvalue(),
                                       f"export_{datetime.datetime.now():%Y%m%d}.{ext}")
        with tabs[1]:
            uf = st.file_uploader("File", type=['csv', 'json', 'xlsx'], key="imp")
            if uf and st.button("Import"):
                st.info("Import functionality — add media files to person folders and scan.")
        with tabs[2]:
            if st.button("Create Backup"):
                import shutil
                bp = Config.EXPORT_DIR / f"backup_{datetime.datetime.now():%Y%m%d_%H%M%S}.db"
                shutil.copy2(self.mgr.db.db_path, bp)
                with open(bp, 'rb') as f:
                    st.download_button("Download Backup", f.read(), bp.name, "application/x-sqlite3")

    # ── MAIN RENDERER ─────────────────────────────────────────────────
    def render_main(self):
        FrameRenderer.inject_global_css()
        self.render_sidebar()
        page = st.session_state.get('current_page', 'dashboard')
        pages = {
            'dashboard': self.render_dashboard, 'people': self.render_gallery_page,
            'gallery': self.render_gallery_page, 'media_detail': self.render_media_detail_page,
            'videos': self.render_videos_page, 'photos': self.render_photos_page,
            'favorites': self.render_favorites_page, 'search': self.render_search_page,
            'statistics': self.render_statistics_page, 'settings': self.render_settings_page,
            'import_export': self.render_import_export_page,
        }
        pages.get(page, self.render_dashboard)()
        st.divider()
        st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} v{Config.VERSION}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    if not check_password():
        return
    try:
        app = PhotoVideoAlbumApp()
        if not app.initialized:
            st.error("Init failed."); return
        if not VIDEO_SUPPORT:
            st.warning("⚠️ pip install opencv-python moviepy")
        app.render_main()
    except Exception as e:
        st.error(f"Error: {e}")
        with st.expander("Details"): st.exception(e)
        if st.button("Retry"): st.rerun()


if __name__ == "__main__":
    main()
