"""
COMPREHENSIVE WEB PHOTO & VIDEO ALBUM APPLICATION
Version: 6.0.0 - HD Viewer, Prev/Next Navigation, Numeric Password, Thumb Strip
Features: Table of Contents, Image/Video Gallery, Comments, Ratings, Metadata,
          Search, Numeric Password Auth, Luxury Photo Frames, Slideshow,
          Breadcrumb Navigation, Download, Fullscreen View, Frame Style Selector,
          Enhanced Media Viewer with Prev/Next & Thumbnail Strip
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

# ============================================================================
# NUMERIC PASSWORD AUTHENTICATION (8-digit code only)
# ============================================================================
# The secret code is a numeric date: 19870505 (no letters, no cipher)
_REAL_PASSWORD = "19870505"

def check_password():
    """Numeric-only password gate. Only digits 0-9 are accepted."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("""
    <style>
    .login-bg {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 60px 30px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 24px 80px rgba(0,0,0,0.5);
        margin: 40px 0;
    }
    .login-title {
        font-size: 2.6em;
        font-weight: 800;
        background: linear-gradient(90deg, #f9d423, #ff4e50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .login-sub {
        color: #a0a0c0;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .lock-icon {
        font-size: 72px;
        margin-bottom: 10px;
    }
    </style>
    <div class="login-bg">
        <div class="lock-icon">🔐</div>
        <div class="login-title">MemoryVault Pro+</div>
        <div class="login-sub">Secure Photo &amp; Video Album</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input(
            "Access Key",
            type="password",
            key="password_input",
            placeholder="Enter 8‑digit numeric code",
            label_visibility="collapsed"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔓 Unlock", use_container_width=True, type="primary"):
                pwd_clean = password.strip()
                if pwd_clean.isdigit() and pwd_clean == _REAL_PASSWORD:
                    st.session_state.authenticated = True
                    st.success("✅ Access granted! Welcome back.")
                    time.sleep(0.6)
                    st.rerun()
                else:
                    st.error("❌ Invalid numeric key. Please try again.")
                    time.sleep(0.5)

        with col_b:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.authenticated = False
                st.rerun()

        with st.expander("🔑 Need a hint?"):
            st.info("💡 The access key is a **numeric** code – 8 digits.")
            st.warning("🤔 Think of a personal 8‑digit number you’d never forget.")
            st.caption("Only digits 0‑9 are allowed. Letters will not work.")

    return False

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================
class Config:
    """Application configuration constants"""
    APP_NAME = "MemoryVault Pro+"
    VERSION = "6.0.0"

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
    HD_SIZE = (1920, 1080)                     # New: HD image size for viewer
    THUMB_STRIP_SIZE = (120, 90)               # Small thumbnails for navigation strip

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

    # Frame style options
    FRAME_STYLES = [
        "Elegant Gold",
        "Polaroid",
        "Modern Shadow",
        "Dark Museum",
        "Vintage",
        "Gallery White"
    ]
    DEFAULT_FRAME_STYLE = "Elegant Gold"

    @classmethod
    def init_directories(cls):
        directories = [
            cls.DATA_DIR, cls.THUMBNAIL_DIR, cls.VIDEO_THUMBNAIL_DIR,
            cls.METADATA_DIR, cls.DB_DIR, cls.EXPORT_DIR, cls.VIDEO_CACHE_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_sample_structure()

    @classmethod
    def create_sample_structure(cls):
        sample_people = ["john-smith", "sarah-johnson", "michael-brown"]
        for person in sample_people:
            person_dir = cls.DATA_DIR / person
            person_dir.mkdir(exist_ok=True)
            readme_file = person_dir / "README.txt"
            readme_file.write_text(f"Photos/Videos of {person.replace('-', ' ').title()}\nAdd your media here!")
            sample_image_path = person_dir / "sample.jpg"
            if not sample_image_path.exists():
                try:
                    img = Image.new('RGB', (400, 300), color='#667eea')
                    draw = ImageDraw.Draw(img)
                    draw.ellipse((150, 50, 250, 150), fill='#ffffff', outline='#4a5568')
                    draw.rectangle((150, 150, 250, 280), fill='#ffffff', outline='#4a5568')
                    draw.text((120, 250), person.replace('-', ' ').title(), fill='#2d3748')
                    img.save(sample_image_path)
                except Exception as e:
                    st.warning(f"Could not create sample image for {person}: {str(e)}")


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
# CUSTOM CSS / FRAME RENDERER (extended for thumbnail strip & fullscreen)
# ============================================================================
class FrameRenderer:
    """Generates luxury HTML/CSS frames for images and thumbnails."""

    @staticmethod
    def wrap_thumbnail(image_src: str, caption: str = "",
                       frame_style: str = "Elegant Gold",
                       is_video: bool = False,
                       duration: float = None) -> str:
        """Return HTML wrapping a thumbnail in the chosen luxury frame."""
        duration_badge = ""
        if is_video and duration:
            m, s = int(duration // 60), int(duration % 60)
            duration_badge = (
                f'<div style="position:absolute;bottom:8px;right:8px;'
                f'background:rgba(0,0,0,.75);color:#fff;padding:2px 7px;'
                f'border-radius:4px;font-size:10px;font-weight:700;">'
                f'{m:02d}:{s:02d}</div>'
            )
        play_icon = ''
        if is_video:
            play_icon = (
                '<div style="position:absolute;top:50%;left:50%;'
                'transform:translate(-50%,-50%);font-size:36px;'
                'color:#fff;opacity:.85;text-shadow:1px 1px 6px rgba(0,0,0,.6);">▶</div>'
            )

        if frame_style == FrameStyle.ELEGANT_GOLD.value:
            outer = (
                'background:linear-gradient(135deg,#d4a574,#f0d9b5,#c9956b,#f0d9b5,#d4a574);'
                'padding:8px;border-radius:6px;'
                'box-shadow:0 6px 18px rgba(0,0,0,.25),inset 0 1px 0 rgba(255,255,255,.4);'
            )
            inner = (
                'background:#fff;padding:6px;border-radius:3px;'
                'box-shadow:inset 0 0 10px rgba(0,0,0,.06);'
            )
        elif frame_style == FrameStyle.POLAROID.value:
            outer = (
                'background:#fff;padding:10px 10px 38px 10px;'
                'box-shadow:0 4px 12px rgba(0,0,0,.18);'
                'border-radius:2px;'
            )
            inner = ''
        elif frame_style == FrameStyle.MODERN_SHADOW.value:
            outer = (
                'background:transparent;padding:0;border-radius:12px;'
                'box-shadow:0 8px 24px rgba(0,0,0,.15),0 2px 6px rgba(0,0,0,.08);'
                'overflow:hidden;'
            )
            inner = ''
        elif frame_style == FrameStyle.DARK_MUSEUM.value:
            outer = (
                'background:linear-gradient(145deg,#1a1a2e,#16213e);'
                'padding:14px;border-radius:10px;'
                'box-shadow:0 12px 36px rgba(0,0,0,.45),0 0 0 1px rgba(255,255,255,.06);'
            )
            inner = (
                'background:#fff;padding:6px;border-radius:3px;'
                'box-shadow:inset 0 0 12px rgba(0,0,0,.04);'
            )
        elif frame_style == FrameStyle.VINTAGE.value:
            outer = (
                'background:linear-gradient(135deg,#e8d5b7,#f5e6cc,#d4b896);'
                'padding:10px;border-radius:4px;'
                'box-shadow:0 4px 14px rgba(0,0,0,.2),inset 0 0 30px rgba(139,109,63,.15);'
                'border:1px solid #c9a96e;'
            )
            inner = (
                'background:#faf5ee;padding:5px;border-radius:2px;'
            )
        elif frame_style == FrameStyle.GALLERY_WHITE.value:
            outer = (
                'background:#fafafa;padding:12px;border-radius:2px;'
                'box-shadow:0 2px 10px rgba(0,0,0,.08);'
                'border:1px solid #e8e8e8;'
            )
            inner = ''
        else:
            outer = 'padding:6px;'
            inner = ''

        caption_html = ""
        if frame_style == FrameStyle.POLAROID.value and caption:
            caption_html = (
                f'<div style="text-align:center;padding-top:8px;'
                f'font-family:Georgia,serif;font-size:12px;color:#444;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                f'{caption}</div>'
            )

        html = f"""
        <div style="{outer}position:relative;transition:transform .25s ease,box-shadow .25s ease;"
             onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 32px rgba(0,0,0,.3)'"
             onmouseout="this.style.transform='translateY(0)';this.style.boxShadow=''">
            <div style="{inner}position:relative;overflow:hidden;border-radius:3px;">
                <img src="{image_src}" style="width:100%;height:200px;object-fit:cover;display:block;border-radius:2px;">
                {play_icon}
                {duration_badge}
            </div>
            {caption_html}
        </div>
        """
        return html

    @staticmethod
    def wrap_detail(image_src: str, frame_style: str = "Elegant Gold") -> str:
        """Return HTML wrapping a full-size image in an ornate frame."""
        if frame_style == FrameStyle.ELEGANT_GOLD.value:
            outer = (
                'background:linear-gradient(135deg,#b8860b,#daa520,#ffd700,#daa520,#b8860b);'
                'padding:12px;border-radius:8px;'
                'box-shadow:0 16px 48px rgba(0,0,0,.35),inset 0 2px 0 rgba(255,255,255,.3),'
                'inset 0 -2px 0 rgba(0,0,0,.2);'
            )
            mat = (
                'background:#fffff5;padding:18px;border-radius:4px;'
                'box-shadow:inset 0 0 20px rgba(0,0,0,.06);'
            )
        elif frame_style == FrameStyle.POLAROID.value:
            outer = (
                'background:#fff;padding:18px 18px 64px 18px;'
                'box-shadow:0 8px 28px rgba(0,0,0,.18);border-radius:2px;'
            )
            mat = ''
        elif frame_style == FrameStyle.MODERN_SHADOW.value:
            outer = (
                'background:transparent;padding:0;border-radius:14px;'
                'box-shadow:0 12px 40px rgba(0,0,0,.18);overflow:hidden;'
            )
            mat = ''
        elif frame_style == FrameStyle.DARK_MUSEUM.value:
            outer = (
                'background:linear-gradient(160deg,#0d0d1a,#1a1a30,#0d0d1a);'
                'padding:24px;border-radius:14px;'
                'box-shadow:0 20px 60px rgba(0,0,0,.5),0 0 0 1px rgba(255,255,255,.04);'
            )
            mat = (
                'background:#fffff8;padding:16px;border-radius:4px;'
                'box-shadow:inset 0 0 16px rgba(0,0,0,.04);'
            )
        elif frame_style == FrameStyle.VINTAGE.value:
            outer = (
                'background:linear-gradient(135deg,#d4b896,#e8d5b7,#c9a96e);'
                'padding:14px;border-radius:4px;'
                'box-shadow:0 10px 30px rgba(0,0,0,.25),inset 0 0 40px rgba(139,109,63,.12);'
                'border:2px solid #a08050;'
            )
            mat = (
                'background:#faf5ee;padding:12px;border-radius:2px;'
                'box-shadow:inset 0 0 10px rgba(0,0,0,.04);'
            )
        elif frame_style == FrameStyle.GALLERY_WHITE.value:
            outer = (
                'background:#fff;padding:20px;border-radius:4px;'
                'box-shadow:0 4px 20px rgba(0,0,0,.08);'
                'border:1px solid #e0e0e0;'
            )
            mat = ''
        else:
            outer = 'padding:8px;'
            mat = ''

        return f"""
        <div style="{outer}">
            <div style="{mat}">
                <img src="{image_src}" style="width:100%;display:block;border-radius:2px;">
            </div>
        </div>
        """

    # NEW: Thumbnail strip item frame (small, clickable)
    @staticmethod
    def wrap_thumb_strip_item(src: str, active: bool = False, is_video: bool = False) -> str:
        active_class = "active" if active else ""
        play_mark = '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:20px;color:#fff;opacity:.8;">▶</div>' if is_video else ''
        return f'''
        <div class="thumb-item {active_class}" style="cursor:pointer;">
            <img src="{src}" style="width:100%;height:100%;object-fit:cover;">
            {play_mark}
        </div>
        '''

    @staticmethod
    def inject_global_css():
        st.markdown("""
        <style>
        /* Smooth scrolling & nicer base font */
        .stApp {{ scroll-behavior: smooth; }}

        /* Breadcrumb bar */
        .breadcrumb {{ display:flex; align-items:center; gap:6px;
                       font-size:13px; color:#888; margin-bottom:16px;
                       flex-wrap:wrap; }}
        .breadcrumb a {{ color:#667eea; text-decoration:none; }}
        .breadcrumb a:hover {{ text-decoration:underline; }}
        .breadcrumb .sep {{ color:#ccc; }}

        /* Thumbnail strip */
        .thumb-strip-container {{
            margin-top: 20px;
            padding: 8px 0;
            overflow-x: auto;
            white-space: nowrap;
            scrollbar-width: thin;
        }}
        .thumb-strip {{
            display: flex;
            gap: 8px;
            padding: 4px 2px;
        }}
        .thumb-item {{
            width: 90px;
            height: 68px;
            background: #1e1e2e;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
            transition: transform 0.15s, opacity 0.15s;
            opacity: 0.6;
            cursor: pointer;
            flex-shrink: 0;
        }}
        .thumb-item:hover {{
            opacity: 1;
            transform: scale(1.05);
        }}
        .thumb-item.active {{
            opacity: 1;
            transform: scale(1.08);
            box-shadow: 0 0 0 2px #667eea;
        }}
        .thumb-item img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        /* Fullscreen overlay */
        .fullscreen-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.95);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: zoom-out;
        }}
        .fullscreen-overlay img {{
            max-width: 95vw;
            max-height: 92vh;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 0 80px rgba(0, 0, 0, 0.6);
        }}
        .close-fullscreen-btn {{
            position: fixed;
            top: 20px;
            right: 30px;
            background: rgba(0,0,0,0.6);
            color: white;
            border: none;
            border-radius: 30px;
            padding: 8px 16px;
            font-size: 16px;
            cursor: pointer;
            z-index: 10000;
        }}
        /* Navigation buttons */
        .nav-btn-container {{
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 300px;
        }}
        .nav-btn {{
            background: rgba(102, 126, 234, 0.9);
            color: #fff;
            border: none;
            border-radius: 50%;
            width: 56px;
            height: 56px;
            font-size: 28px;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .nav-btn:hover {{
            background: #764ba2;
            transform: scale(1.1);
        }}
        .nav-btn:disabled {{
            opacity: 0.3;
            cursor: not-allowed;
        }}
        </style>
        """, unsafe_allow_html=True)


# ============================================================================
# BREADCRUMB HELPER
# ============================================================================
def render_breadcrumb(trail: List[Tuple[str, str]]):
    """Render a breadcrumb navigation bar.
    trail: list of (label, page_key) tuples. Last item is current page (no link).
    """
    parts = []
    for i, (label, key) in enumerate(trail):
        if i < len(trail) - 1:
            parts.append(f'<a href="#" data-key="{key}">{label}</a>')
            parts.append('<span class="sep">›</span>')
        else:
            parts.append(f'<span style="color:#333;font-weight:600;">{label}</span>')
    st.markdown(f'<div class="breadcrumb">{"".join(parts)}</div>', unsafe_allow_html=True)


# ============================================================================
# DATA MODELS (unchanged)
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
            raise FileNotFoundError(f"Media file not found: {file_path}")
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
        ext = file_path.suffix.lower()
        if ext in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaType.VIDEO.value
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']:
            return MediaType.IMAGE.value
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @classmethod
    def _from_image(cls, image_path: Path, stats: os.stat_result, media_type: str) -> 'MediaMetadata':
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
        dimensions = (0, 0)
        duration = 0.0
        frame_rate = 0.0
        if VIDEO_SUPPORT:
            try:
                clip = VideoFileClip(str(video_path))
                dimensions = clip.size
                duration = clip.duration
                frame_rate = clip.fps
                clip.close()
            except Exception:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    frame_rate = fps
                    cap.release()
                except Exception as e2:
                    st.warning(f"Could not extract video metadata for {video_path}: {str(e2)}")
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
    def _calculate_checksum(file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found for checksum: {file_path}")
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


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
# DATABASE MANAGEMENT (unchanged)
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
            st.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def _init_database(self):
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS media (
                    media_id TEXT PRIMARY KEY, filename TEXT NOT NULL,
                    filepath TEXT NOT NULL, file_size INTEGER,
                    media_type TEXT NOT NULL, width INTEGER, height INTEGER,
                    format TEXT, duration REAL, frame_rate REAL,
                    created_date TIMESTAMP, modified_date TIMESTAMP,
                    exif_data TEXT, checksum TEXT UNIQUE,
                    thumbnail_path TEXT, video_thumbnail_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS people (
                    person_id TEXT PRIMARY KEY, folder_name TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL, bio TEXT, birth_date DATE,
                    relationship TEXT, contact_info TEXT, social_links TEXT,
                    profile_image TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS album_entries (
                    entry_id TEXT PRIMARY KEY, media_id TEXT NOT NULL,
                    person_id TEXT NOT NULL, caption TEXT, description TEXT,
                    location TEXT, date_taken TIMESTAMP, tags TEXT,
                    privacy_level TEXT DEFAULT 'public', created_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (media_id) REFERENCES media (media_id),
                    FOREIGN KEY (person_id) REFERENCES people (person_id)
                )''')
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY, entry_id TEXT NOT NULL,
                    user_id TEXT NOT NULL, username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_edited BOOLEAN DEFAULT 0, parent_comment_id TEXT,
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )''')
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ratings (
                    rating_id TEXT PRIMARY KEY, entry_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating_value INTEGER CHECK (rating_value BETWEEN 1 AND 5),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(entry_id, user_id),
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )''')
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_favorites (
                    user_id TEXT, entry_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, entry_id),
                    FOREIGN KEY (entry_id) REFERENCES album_entries (entry_id)
                )''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_media ON album_entries(media_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_person ON album_entries(person_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_album_entries_created ON album_entries(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_entry ON comments(entry_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_entry ON ratings(entry_id)')
                self._migrate_existing_data(conn)
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Database initialization error: {str(e)}")
            raise

    def _migrate_existing_data(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) FROM media")
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                INSERT INTO media (media_id, filename, filepath, file_size, media_type,
                                  width, height, format, duration, frame_rate,
                                  created_date, modified_date, exif_data, checksum, thumbnail_path)
                SELECT image_id, filename, filepath, file_size, 'image',
                       width, height, format, NULL, NULL,
                       created_date, modified_date, exif_data, checksum, thumbnail_path
                FROM images
                ''')
                cursor.execute("DROP TABLE images")

    def add_media(self, metadata: MediaMetadata, thumbnail_path: str = None, video_thumbnail_path: str = None):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO media
                (media_id, filename, filepath, file_size, media_type, width, height, format,
                duration, frame_rate, created_date, modified_date, exif_data, checksum,
                thumbnail_path, video_thumbnail_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''', (
                    metadata.media_id, metadata.filename, metadata.filepath,
                    metadata.file_size, metadata.media_type,
                    metadata.dimensions[0], metadata.dimensions[1], metadata.format,
                    metadata.duration, metadata.frame_rate,
                    metadata.created_date, metadata.modified_date,
                    json.dumps(metadata.exif_data) if metadata.exif_data else None,
                    metadata.checksum, thumbnail_path, video_thumbnail_path
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding media: {str(e)}")
            raise

    def get_media(self, media_id: str) -> Optional[Dict]:
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
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM media WHERE media_type = ? ORDER BY created_date DESC LIMIT ?',
                               (media_type, limit))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving {media_type}s: {str(e)}")
            return []

    def add_person(self, person: PersonProfile):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO people
                (person_id, folder_name, display_name, bio, birth_date,
                relationship, contact_info, social_links, profile_image)
                VALUES (?,?,?,?,?,?,?,?,?)
                ''', (
                    person.person_id, person.folder_name, person.display_name,
                    person.bio,
                    person.birth_date.isoformat() if person.birth_date else None,
                    person.relationship, person.contact_info,
                    json.dumps(person.social_links), person.profile_image
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding person: {str(e)}")
            raise

    def get_all_people(self) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM people ORDER BY display_name')
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving people: {str(e)}")
            return []

    def get_person_by_folder(self, folder_name: str) -> Optional[Dict]:
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM people WHERE folder_name = ?', (folder_name,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            st.error(f"Error retrieving person: {str(e)}")
            return None

    def add_album_entry(self, entry: AlbumEntry):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO album_entries
                (entry_id, media_id, person_id, caption, description, location,
                date_taken, tags, privacy_level, created_by)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ''', (
                    entry.entry_id, entry.media_id, entry.person_id,
                    entry.caption, entry.description, entry.location,
                    entry.date_taken,
                    ','.join(entry.tags) if entry.tags else None,
                    entry.privacy_level, entry.created_by
                ))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding album entry: {str(e)}")
            raise

    def add_comment(self, comment: Comment):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO comments (comment_id, entry_id, user_id, username, content, parent_comment_id)
                VALUES (?,?,?,?,?,?)
                ''', (comment.comment_id, comment.entry_id, comment.user_id,
                      comment.username, comment.content, comment.parent_comment_id))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding comment: {str(e)}")
            raise

    def add_rating(self, rating: Rating):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO ratings (rating_id, entry_id, user_id, rating_value)
                VALUES (?,?,?,?)
                ''', (rating.rating_id, rating.entry_id, rating.user_id, rating.rating_value))
                conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error adding rating: {str(e)}")
            raise

    def get_entry_comments(self, entry_id: str) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM comments WHERE entry_id = ? ORDER BY created_at DESC',
                               (entry_id,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving comments: {str(e)}")
            return []

    def get_entry_ratings(self, entry_id: str) -> Tuple[float, int]:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT AVG(rating_value), COUNT(*) FROM ratings WHERE entry_id = ?',
                               (entry_id,))
                result = cursor.fetchone()
                return (float(result[0]) if result and result[0] is not None else 0.0,
                        result[1] if result else 0)
        except sqlite3.Error as e:
            st.error(f"Error retrieving ratings: {str(e)}")
            return (0.0, 0)

    def search_entries(self, query: str, person_id: str = None, media_type: str = None) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                search_pattern = f'%{query}%'
                conditions, params = [], []
                if person_id:
                    conditions.append("ae.person_id = ?")
                    params.append(person_id)
                if media_type and media_type != 'all':
                    conditions.append("m.media_type = ?")
                    params.append(media_type)
                conditions.append("(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)")
                params.extend([search_pattern, search_pattern, search_pattern])
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                cursor.execute(f'''
                SELECT ae.*, p.display_name, m.filename, m.media_type,
                       m.thumbnail_path, m.video_thumbnail_path
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
                WHERE {where_clause} ORDER BY ae.created_at DESC
                ''', params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error searching: {str(e)}")
            return []

    def get_entry_details(self, entry_id: str) -> Optional[Dict]:
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                SELECT ae.entry_id, ae.media_id, ae.person_id, ae.caption,
                       ae.description, ae.location, ae.date_taken, ae.tags,
                       ae.privacy_level, ae.created_by, ae.created_at, ae.updated_at,
                       p.display_name, p.folder_name,
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
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    if result.get('exif_data'):
                        try:
                            result['exif_data'] = json.loads(result['exif_data'])
                        except Exception:
                            result['exif_data'] = {}
                    result['tags'] = ([t.strip() for t in result['tags'].split(',') if t.strip()]
                                      if result.get('tags') else [])
                    return result
                return None
        except sqlite3.Error as e:
            st.error(f"Database error: {str(e)}")
            return None

    def get_recent_entries(self, limit: int = 10) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                SELECT ae.*, p.display_name, m.filename, m.media_type,
                       m.thumbnail_path, m.video_thumbnail_path
                FROM album_entries ae
                JOIN people p ON ae.person_id = p.person_id
                JOIN media m ON ae.media_id = m.media_id
                ORDER BY ae.created_at DESC LIMIT ?
                ''', (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving recent entries: {str(e)}")
            return []


# ============================================================================
# MEDIA PROCESSOR (extended with HD and thumbnail strip methods)
# ============================================================================
class MediaProcessor:
    @staticmethod
    def create_thumbnail(media_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        if not media_path.exists():
            return None
        ext = media_path.suffix.lower()
        if ext in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaProcessor._create_video_thumbnail(media_path, thumbnail_dir)
        return MediaProcessor._create_image_thumbnail(media_path, thumbnail_dir)

    @staticmethod
    def _create_image_thumbnail(image_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        thumbnail_dir = thumbnail_dir or Config.THUMBNAIL_DIR
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_path = thumbnail_dir / f"{image_path.stem}_thumb.jpg"
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
            return thumbnail_path
        except Exception as e:
            st.error(f"Error creating thumbnail for {image_path}: {str(e)}")
            return None

    @staticmethod
    def _create_video_thumbnail(video_path: Path, thumbnail_dir: Path = None) -> Optional[Path]:
        if not VIDEO_SUPPORT:
            return None
        thumbnail_dir = thumbnail_dir or Config.VIDEO_THUMBNAIL_DIR
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_path = thumbnail_dir / f"{video_path.stem}_thumb.jpg"
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail(Config.VIDEO_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                return thumbnail_path
        except Exception as e:
            st.error(f"Error creating video thumbnail for {video_path}: {str(e)}")
        return None

    # NEW: Get HD data URL for images (resized to 1920x1080 max)
    @staticmethod
    def get_hd_data_url(media_path: Path) -> str:
        try:
            if not media_path.exists():
                return ""
            with Image.open(media_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    img = background
                # Resize to HD (max 1920x1080, preserve aspect ratio)
                img.thumbnail(Config.HD_SIZE, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95, optimize=True)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception as e:
            st.error(f"Error generating HD image: {str(e)}")
            return ""

    @staticmethod
    def get_thumb_strip_url(media_path: Path, is_video: bool = False) -> str:
        """Small thumbnail for navigation strip (max 120x90)."""
        try:
            if not media_path.exists():
                return ""
            if is_video:
                # Try to use existing video thumbnail
                video_thumb_path = Config.VIDEO_THUMBNAIL_DIR / f"{media_path.stem}_thumb.jpg"
                if video_thumb_path.exists():
                    return MediaProcessor.get_data_url(video_thumb_path)
                # Otherwise fallback to generic video icon placeholder
                return ""
            else:
                with Image.open(media_path) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode in ('RGBA', 'LA'):
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img)
                        img = background
                    img.thumbnail(Config.THUMB_STRIP_SIZE, Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=80)
                    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return ""

    @staticmethod
    def get_media_data_url(media_path: Path) -> str:
        try:
            if not media_path.exists():
                return ""
            mime_type, _ = mimetypes.guess_type(str(media_path))
            if not mime_type:
                ext = media_path.suffix.lower()
                mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                            '.gif': 'image/gif', '.mp4': 'video/mp4', '.webm': 'video/webm'}
                mime_type = mime_map.get(ext, 'application/octet-stream')
            with open(media_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            st.error(f"Error encoding media {media_path}: {str(e)}")
            return ""

    @staticmethod
    def prepare_video_stream(video_path: Path, max_size_mb: int = 50) -> Optional[bytes]:
        try:
            if not video_path.exists():
                return None
            if video_path.stat().st_size > max_size_mb * 1024 * 1024:
                st.warning(f"Video too large ({video_path.stat().st_size/(1024*1024):.1f}MB). Max {max_size_mb}MB.")
                return None
            with open(video_path, "rb") as f:
                return f.read()
        except Exception as e:
            st.error(f"Error reading video: {str(e)}")
            return None

    @staticmethod
    def get_video_info(video_path: Path) -> Dict[str, Any]:
        info = {'duration': 0, 'dimensions': (0, 0), 'frame_rate': 0,
                'file_size': video_path.stat().st_size if video_path.exists() else 0}
        if not VIDEO_SUPPORT or not video_path.exists():
            return info
        try:
            clip = VideoFileClip(str(video_path))
            info['duration'] = clip.duration
            info['dimensions'] = clip.size
            info['frame_rate'] = clip.fps
            clip.close()
        except Exception:
            try:
                cap = cv2.VideoCapture(str(video_path))
                info['dimensions'] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                info['frame_rate'] = cap.get(cv2.CAP_PROP_FPS)
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                info['duration'] = fc / info['frame_rate'] if info['frame_rate'] > 0 else 0
                cap.release()
            except Exception:
                pass
        return info


# ============================================================================
# CACHE MANAGEMENT (unchanged)
# ============================================================================
class CacheManager:
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self._video_cache = {}

    def get(self, key: str, default=None):
        if key in self._cache:
            if time.time() - self._timestamps[key] < Config.CACHE_TTL:
                return self._cache[key]
            del self._cache[key]
            del self._timestamps[key]
        return default

    def set(self, key: str, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self, key: str = None):
        if key:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._timestamps.clear()
            self._video_cache.clear()

    def get_or_set(self, key: str, generator_func):
        cached = self.get(key)
        if cached is not None:
            return cached
        try:
            value = generator_func()
            self.set(key, value)
            return value
        except Exception as e:
            st.error(f"Cache generator error for {key}: {str(e)}")
            return None

    def get_video(self, video_path: Path) -> Optional[bytes]:
        return self._video_cache.get(f"video_{video_path}")

    def set_video(self, video_path: Path, video_data: bytes):
        key = f"video_{video_path}"
        self._video_cache[key] = video_data
        if len(self._video_cache) > 10:
            oldest = next(iter(self._video_cache))
            del self._video_cache[oldest]

    def clear_video_cache(self):
        self._video_cache.clear()


# ============================================================================
# UI COMPONENTS (unchanged)
# ============================================================================
class UIComponents:
    @staticmethod
    def rating_stars(rating: float, max_rating: int = 5, size: int = 20) -> str:
        if rating is None or rating <= 0:
            rating = 0
        stars_html = []
        full_stars = int(rating)
        has_half = rating - full_stars >= 0.5
        for i in range(max_rating):
            if i < full_stars:
                stars_html.append('⭐')
            elif i == full_stars and has_half:
                stars_html.append('⭐')
            else:
                stars_html.append('☆')
        return (f'<div style="color:#FFD700;font-size:{size}px;letter-spacing:1px;display:inline-block;">'
                f'{"".join(stars_html)}'
                f'<span style="color:#666;font-size:14px;margin-left:8px;vertical-align:middle;">'
                f'{rating:.1f}/{max_rating}</span></div>')

    @staticmethod
    def tag_badges(tags: List[str], max_display: int = 5) -> str:
        if not tags:
            return ""
        displayed = tags[:max_display]
        extra = len(tags) - max_display if len(tags) > max_display else 0
        badges = []
        for tag in displayed:
            badges.append(
                f'<span style="background:linear-gradient(135deg,#667eea,#764ba2);'
                f'color:#fff;padding:4px 12px;border-radius:20px;font-size:12px;'
                f'margin:2px;display:inline-block;box-shadow:0 2px 4px rgba(0,0,0,.1);'
                f'text-transform:capitalize;">{tag.replace("-"," ")}</span>')
        html = ' '.join(badges)
        if extra > 0:
            html += (f'<span style="background:#f0f0f0;color:#666;padding:4px 8px;'
                     f'border-radius:20px;font-size:12px;margin:2px;display:inline-block;">'
                     f'+{extra} more</span>')
        return html

    @staticmethod
    def media_type_badge(media_type: str) -> str:
        if media_type == MediaType.VIDEO.value:
            return ('<span style="background:linear-gradient(135deg,#FF416C,#FF4B2B);'
                    'color:#fff;padding:2px 8px;border-radius:12px;font-size:10px;'
                    'font-weight:700;margin-left:8px;display:inline-block;">🎬 VIDEO</span>')
        return ('<span style="background:linear-gradient(135deg,#667eea,#764ba2);'
                'color:#fff;padding:2px 8px;border-radius:12px;font-size:10px;'
                'font-weight:700;margin-left:8px;display:inline-block;">📸 IMAGE</span>')


# ============================================================================
# ALBUM MANAGER (unchanged except added method to get all entries for a person)
# ============================================================================
class AlbumManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.cache = CacheManager()
        self.media_processor = MediaProcessor()
        self._init_session_state()

    def _init_session_state(self):
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'initialized': True,
                'current_page': 'dashboard',
                'selected_person': None,
                'selected_media': None,
                'search_query': '',
                'view_mode': 'grid',
                'sort_by': 'date',
                'sort_order': 'desc',
                'media_filter': 'all',
                'selected_tags': [],
                'user_id': str(uuid.uuid4()),
                'username': 'Guest',
                'user_role': UserRoles.VIEWER.value,
                'favorites': set(),
                'recently_viewed': [],
                'toc_page': 1,
                'gallery_page': 1,
                'show_directory_info': True,
                'video_autoplay': False,
                'frame_style': Config.DEFAULT_FRAME_STYLE,
                'slideshow_active': False,
                'slideshow_index': 0,
                'fullscreen_media': None,
                # NEW: for media navigation in enhanced viewer
                'media_nav_list': [],      # list of entry dicts
                'media_nav_index': 0,
            })

    def scan_directory(self, data_dir: Path = None) -> Dict:
        data_dir = data_dir or Config.DATA_DIR
        results = {'total_media': 0, 'new_media': 0, 'updated_media': 0,
                   'images_found': 0, 'videos_found': 0, 'people_found': 0, 'errors': []}
        try:
            if not data_dir.exists():
                data_dir.mkdir(parents=True)
                return results

            person_dirs = [d for d in data_dir.iterdir()
                           if d.is_dir() and not d.name.startswith('.')
                           and d.name not in ['thumbnails', 'video_thumbnails', 'video_cache',
                                              'metadata', 'database', 'exports']]

            if not person_dirs:
                Config.create_sample_structure()
                person_dirs = [d for d in data_dir.iterdir()
                               if d.is_dir() and not d.name.startswith('.')
                               and d.name not in ['thumbnails', 'video_thumbnails', 'video_cache',
                                                  'metadata', 'database', 'exports']]

            results['people_found'] = len(person_dirs)
            progress_bar = st.progress(0)
            total_files = 0
            processed = 0

            for pd_dir in person_dirs:
                total_files += sum(1 for f in pd_dir.iterdir()
                                   if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS)

            if total_files == 0:
                st.info("No media files found in any folders.")
                progress_bar.empty()
                return results

            for person_dir in person_dirs:
                if '-' in person_dir.name:
                    display_name = ' '.join(p.capitalize() for p in person_dir.name.split('-'))
                elif '_' in person_dir.name:
                    display_name = ' '.join(p.capitalize() for p in person_dir.name.split('_'))
                else:
                    display_name = ' '.join(w.capitalize() for w in person_dir.name.split())

                existing_person = self.db.get_person_by_folder(person_dir.name)
                if not existing_person:
                    pp = PersonProfile(person_id=str(uuid.uuid4()), folder_name=person_dir.name,
                                       display_name=display_name, bio=f"Photos/Videos of {display_name}",
                                       birth_date=None, relationship="Family/Friend", contact_info="",
                                       social_links={}, profile_image=None,
                                       created_at=datetime.datetime.now())
                    self.db.add_person(pp)
                    person_id = pp.person_id
                else:
                    person_id = existing_person['person_id']

                media_files = [f for f in person_dir.iterdir()
                               if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS]

                for media_path in media_files:
                    try:
                        processed += 1
                        progress_bar.progress(processed / max(total_files, 1))
                        checksum = MediaMetadata._calculate_checksum(media_path)
                        with sqlite3.connect(self.db.db_path) as conn:
                            cur = conn.cursor()
                            cur.execute('SELECT media_id FROM media WHERE checksum = ?', (checksum,))
                            if cur.fetchone():
                                results['updated_media'] += 1
                                continue
                        metadata = MediaMetadata.from_file(media_path)
                        thumb = vid_thumb = None
                        if metadata.media_type == MediaType.IMAGE.value:
                            thumb = self.media_processor.create_thumbnail(media_path)
                            results['images_found'] += 1
                        elif metadata.media_type == MediaType.VIDEO.value:
                            vid_thumb = self.media_processor.create_thumbnail(media_path)
                            results['videos_found'] += 1
                        self.db.add_media(metadata,
                                          str(thumb) if thumb else None,
                                          str(vid_thumb) if vid_thumb else None)
                        entry = AlbumEntry(
                            entry_id=str(uuid.uuid4()), media_id=metadata.media_id,
                            person_id=person_id,
                            caption=media_path.stem.replace('_', ' ').title(),
                            description=f"Media of {display_name}",
                            location="", date_taken=metadata.created_date,
                            tags=[display_name.lower().replace(' ', '-'), metadata.media_type, 'memory'],
                            privacy_level='public', created_by='system',
                            created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
                        self.db.add_album_entry(entry)
                        results['new_media'] += 1
                        results['total_media'] += 1
                    except Exception as e:
                        results['errors'].append(f"Error processing {media_path}: {str(e)}")

            progress_bar.empty()
            self.cache.clear()
            return results
        except Exception as e:
            st.error(f"Critical error during scan: {str(e)}")
            results['errors'].append(f"Critical: {str(e)}")
            return results

    def get_person_stats(self, person_id: str) -> Dict:
        cache_key = f"person_stats_{person_id}"

        def gen():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    c = conn.cursor()
                    c.execute('SELECT COUNT(*) FROM album_entries WHERE person_id=?', (person_id,))
                    mc = c.fetchone()[0]
                    c.execute('''SELECT m.media_type, COUNT(*) FROM album_entries ae
                                 JOIN media m ON ae.media_id=m.media_id
                                 WHERE ae.person_id=? GROUP BY m.media_type''', (person_id,))
                    ic = vc = 0
                    for row in c.fetchall():
                        if row[0] == 'image': ic = row[1]
                        elif row[0] == 'video': vc = row[1]
                    c.execute('''SELECT COUNT(DISTINCT c.comment_id) FROM comments c
                                 JOIN album_entries ae ON c.entry_id=ae.entry_id
                                 WHERE ae.person_id=?''', (person_id,))
                    cc = c.fetchone()[0]
                    c.execute('''SELECT AVG(r.rating_value) FROM ratings r
                                 JOIN album_entries ae ON r.entry_id=ae.entry_id
                                 WHERE ae.person_id=?''', (person_id,))
                    ar = c.fetchone()[0] or 0.0
                    c.execute('SELECT MAX(created_at) FROM album_entries WHERE person_id=?', (person_id,))
                    la = c.fetchone()[0]
                    return {'media_count': mc, 'image_count': ic, 'video_count': vc,
                            'comment_count': cc, 'avg_rating': float(ar), 'last_activity': la}
            except sqlite3.Error as e:
                st.error(f"Stats error: {str(e)}")
                return {'media_count': 0, 'image_count': 0, 'video_count': 0,
                        'comment_count': 0, 'avg_rating': 0.0, 'last_activity': None}

        return self.cache.get_or_set(cache_key, gen)

    def add_to_favorites(self, entry_id: str):
        uid = st.session_state['user_id']
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute('INSERT OR IGNORE INTO user_favorites (user_id,entry_id) VALUES (?,?)',
                             (uid, entry_id))
                conn.commit()
            st.session_state.favorites.add(entry_id)
        except sqlite3.Error as e:
            st.error(f"Error adding favorite: {str(e)}")

    def remove_from_favorites(self, entry_id: str):
        uid = st.session_state['user_id']
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute('DELETE FROM user_favorites WHERE user_id=? AND entry_id=?', (uid, entry_id))
                conn.commit()
            st.session_state.favorites.discard(entry_id)
        except sqlite3.Error as e:
            st.error(f"Error removing favorite: {str(e)}")

    def get_user_favorites(self) -> List[Dict]:
        uid = st.session_state['user_id']
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute('''SELECT ae.*, p.display_name, m.filename, m.media_type,
                             m.thumbnail_path, m.video_thumbnail_path
                             FROM user_favorites uf
                             JOIN album_entries ae ON uf.entry_id=ae.entry_id
                             JOIN people p ON ae.person_id=p.person_id
                             JOIN media m ON ae.media_id=m.media_id
                             WHERE uf.user_id=? ORDER BY uf.created_at DESC''', (uid,))
                return [dict(row) for row in c.fetchall()]
        except sqlite3.Error as e:
            st.error(f"Error retrieving favorites: {str(e)}")
            return []

    def add_comment_to_entry(self, entry_id: str, content: str, parent_comment_id: str = None):
        if not content or not content.strip():
            st.warning("Comment cannot be empty")
            return False
        if len(content) > Config.MAX_COMMENT_LENGTH:
            st.warning(f"Comment too long (max {Config.MAX_COMMENT_LENGTH})")
            return False
        try:
            comment = Comment(comment_id=str(uuid.uuid4()), entry_id=entry_id,
                              user_id=st.session_state['user_id'],
                              username=st.session_state['username'],
                              content=content.strip(), created_at=datetime.datetime.now(),
                              is_edited=False, parent_comment_id=parent_comment_id)
            self.db.add_comment(comment)
            self.cache.clear(f"comments_{entry_id}")
            st.success("Comment added!")
            return True
        except Exception as e:
            st.error(f"Error adding comment: {str(e)}")
            return False

    def add_rating_to_entry(self, entry_id: str, rating_value: int):
        if rating_value < 1 or rating_value > 5:
            st.warning("Rating must be 1-5")
            return False
        try:
            rating = Rating(rating_id=str(uuid.uuid4()), entry_id=entry_id,
                            user_id=st.session_state['user_id'], rating_value=rating_value,
                            created_at=datetime.datetime.now(), updated_at=datetime.datetime.now())
            self.db.add_rating(rating)
            self.cache.clear(f"ratings_{entry_id}")
            st.success(f"Rated {rating_value} ⭐!")
            return True
        except Exception as e:
            st.error(f"Error adding rating: {str(e)}")
            return False

    def get_all_people_with_stats(self) -> List[Dict]:
        def gen():
            try:
                people = self.db.get_all_people()
                result = []
                for person in people:
                    stats = self.get_person_stats(person['person_id'])
                    pw = {**person, **stats}
                    if person.get('profile_image'):
                        pimg = Config.DATA_DIR / person['folder_name'] / person['profile_image']
                        if pimg.exists():
                            pw['profile_image_data'] = self.media_processor.get_media_data_url(pimg)
                    result.append(pw)
                return result
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return []
        return self.cache.get_or_set("all_people_with_stats", gen)

    def get_entries_by_person(self, person_id: str, page: int = 1,
                              search_query: str = None, media_filter: str = 'all') -> Dict:
        cache_key = f"entries_person_{person_id}_p{page}_q{search_query}_f{media_filter}"

        def gen():
            try:
                offset = (page - 1) * Config.ITEMS_PER_PAGE
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    c = conn.cursor()
                    conds, params = ["ae.person_id = ?"], [person_id]
                    if search_query:
                        sp = f'%{search_query}%'
                        conds.append("(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)")
                        params.extend([sp, sp, sp])
                    if media_filter != 'all':
                        conds.append("m.media_type = ?")
                        params.append(media_filter)
                    where = " AND ".join(conds)
                    params.extend([Config.ITEMS_PER_PAGE, offset])
                    c.execute(f'''SELECT ae.*, p.display_name, m.filename, m.media_type,
                                  m.thumbnail_path, m.video_thumbnail_path, m.duration,
                                  (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id=ae.entry_id) as avg_rating,
                                  (SELECT COUNT(*) FROM comments c2 WHERE c2.entry_id=ae.entry_id) as comment_count
                                  FROM album_entries ae
                                  JOIN people p ON ae.person_id=p.person_id
                                  JOIN media m ON ae.media_id=m.media_id
                                  WHERE {where} ORDER BY ae.created_at DESC LIMIT ? OFFSET ?''', params)
                    entries = [dict(row) for row in c.fetchall()]
                    c.execute(f'SELECT COUNT(*) FROM album_entries ae JOIN media m ON ae.media_id=m.media_id WHERE {where}',
                              params[:-2])
                    total = c.fetchone()[0]
                    return {'entries': entries, 'total_count': total,
                            'total_pages': max(1, math.ceil(total / Config.ITEMS_PER_PAGE)),
                            'current_page': page}
            except sqlite3.Error as e:
                st.error(f"Error: {str(e)}")
                return {'entries': [], 'total_count': 0, 'total_pages': 1, 'current_page': page}

        return self.cache.get_or_set(cache_key, gen)

    # NEW: Get all entries for a person (unpaged) for navigation
    def get_all_entries_for_person(self, person_id: str, media_filter: str = 'all') -> List[Dict]:
        cache_key = f"all_entries_person_{person_id}_f{media_filter}"
        def gen():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    c = conn.cursor()
                    conds, params = ["ae.person_id = ?"], [person_id]
                    if media_filter != 'all':
                        conds.append("m.media_type = ?")
                        params.append(media_filter)
                    where = " AND ".join(conds)
                    c.execute(f'''SELECT ae.*, p.display_name, m.filename, m.media_type,
                                  m.thumbnail_path, m.video_thumbnail_path, m.duration,
                                  m.filepath, m.width, m.height, m.file_size, m.format,
                                  (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id=ae.entry_id) as avg_rating
                                  FROM album_entries ae
                                  JOIN people p ON ae.person_id=p.person_id
                                  JOIN media m ON ae.media_id=m.media_id
                                  WHERE {where} ORDER BY ae.created_at DESC''', params)
                    return [dict(row) for row in c.fetchall()]
            except sqlite3.Error as e:
                st.error(f"Error: {str(e)}")
                return []
        return self.cache.get_or_set(cache_key, gen)

    def get_recent_entries(self, limit: int = 10) -> List[Dict]:
        return self.cache.get_or_set(f"recent_{limit}", lambda: self.db.get_recent_entries(limit))

    def get_top_rated_entries(self, limit: int = 10) -> List[Dict]:
        def gen():
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    c = conn.cursor()
                    c.execute('''SELECT ae.*, p.display_name, m.filename, m.media_type,
                                 m.thumbnail_path, m.video_thumbnail_path,
                                 (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id=ae.entry_id) as avg_rating,
                                 (SELECT COUNT(*) FROM ratings r2 WHERE r2.entry_id=ae.entry_id) as rating_count
                                 FROM album_entries ae
                                 JOIN people p ON ae.person_id=p.person_id
                                 JOIN media m ON ae.media_id=m.media_id
                                 WHERE ae.entry_id IN (
                                   SELECT r.entry_id FROM ratings r GROUP BY r.entry_id
                                   HAVING AVG(r.rating_value)>=4.0 ORDER BY AVG(r.rating_value) DESC
                                 ) ORDER BY avg_rating DESC LIMIT ?''', (limit,))
                    return [dict(row) for row in c.fetchall()]
            except sqlite3.Error as e:
                st.error(f"Error: {str(e)}")
                return []
        return self.cache.get_or_set(f"top_rated_{limit}", gen)

    def get_entry_with_details(self, entry_id: str) -> Optional[Dict]:
        def gen():
            try:
                entry = self.db.get_entry_details(entry_id)
                if not entry:
                    return None
                entry['comments'] = self.db.get_entry_comments(entry_id)
                avg, cnt = self.db.get_entry_ratings(entry_id)
                entry['avg_rating'] = avg
                entry['rating_count'] = cnt
                if entry.get('filepath'):
                    mp = Config.DATA_DIR / entry['filepath']
                    if mp.exists():
                        if entry['media_type'] == MediaType.IMAGE.value:
                            entry['media_data_url'] = self.media_processor.get_media_data_url(mp)
                        else:
                            entry['media_path'] = str(mp)
                for key in ['thumbnail_path', 'video_thumbnail_path']:
                    if entry.get(key):
                        tp = Path(entry[key])
                        if tp.exists():
                            entry['thumbnail_data_url'] = self.media_processor.get_media_data_url(tp)
                            break
                uid = st.session_state['user_id']
                with sqlite3.connect(self.db.db_path) as conn:
                    c = conn.cursor()
                    c.execute('SELECT 1 FROM user_favorites WHERE user_id=? AND entry_id=?', (uid, entry_id))
                    entry['is_favorited'] = c.fetchone() is not None
                    c.execute('SELECT rating_value FROM ratings WHERE user_id=? AND entry_id=?', (uid, entry_id))
                    ur = c.fetchone()
                    entry['user_rating'] = ur[0] if ur else 0
                return entry
            except Exception as e:
                st.error(f"Error getting entry: {str(e)}")
                return None
        return self.cache.get_or_set(f"entry_{entry_id}", gen)

    def stream_video(self, video_path: str):
        try:
            vp = Path(video_path)
            if not vp.exists():
                st.error("Video file not found")
                return
            vd = self.cache.get_video(vp)
            if not vd:
                vd = self.media_processor.prepare_video_stream(vp)
                if vd:
                    self.cache.set_video(vp, vd)
            if vd:
                mt, _ = mimetypes.guess_type(video_path)
                st.video(vd, format=mt or "video/mp4", start_time=0)
            else:
                st.error("Could not load video")
        except Exception as e:
            st.error(f"Error streaming video: {str(e)}")


# ============================================================================
# MAIN APPLICATION CLASS (with enhanced media detail view)
# ============================================================================
class PhotoVideoAlbumApp:
    def __init__(self):
        self.manager = AlbumManager()
        self.setup_page_config()
        self.check_initialization()

    def setup_page_config(self):
        st.set_page_config(
            page_title=Config.APP_NAME, page_icon="🎬📸", layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/yourusername/photo-video-album',
                'Report a bug': 'https://github.com/yourusername/photo-video-album/issues',
                'About': f"# {Config.APP_NAME} v{Config.VERSION}"
            })

    def check_initialization(self):
        try:
            Config.init_directories()
            self.initialized = True
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            self.initialized = False

    @property
    def frame_style(self) -> str:
        return st.session_state.get('frame_style', Config.DEFAULT_FRAME_STYLE)

    def render_sidebar(self):
        with st.sidebar:
            st.title(f"🎬📸 {Config.APP_NAME}")
            st.caption(f"v{Config.VERSION}")
            st.divider()

            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("👤")
            with col2:
                st.markdown(f"**{st.session_state['username']}**")
                st.caption(st.session_state['user_role'].title())

            st.divider()
            st.subheader("Navigation")
            nav = {"🏠 Dashboard": "dashboard", "👥 People": "people",
                   "📁 Media Gallery": "gallery", "⭐ Favorites": "favorites",
                   "🎬 Video Library": "videos", "📸 Photo Library": "photos",
                   "🔍 Search": "search", "📊 Statistics": "statistics",
                   "⚙️ Settings": "settings", "📤 Import/Export": "import_export"}
            for label, key in nav.items():
                if st.button(label, use_container_width=True, key=f"nav_{key}"):
                    st.session_state['current_page'] = key
                    # Clear navigation list when leaving enhanced viewer
                    st.session_state['media_nav_list'] = []
                    st.rerun()

            st.divider()
            st.subheader("Quick Actions")
            if st.button("🔄 Scan Directory", use_container_width=True):
                with st.spinner("Scanning…"):
                    r = self.manager.scan_directory()
                    if r['new_media'] > 0:
                        st.success(f"Found {r['new_media']} new media ({r['images_found']} imgs, {r['videos_found']} vids)")
                    st.rerun()
            if st.button("🗑️ Clear Cache", use_container_width=True):
                self.manager.cache.clear()
                st.success("Cache cleared!")
                st.rerun()

            st.divider()
            st.subheader("Frame Style")
            fs = st.selectbox("Choose frame", Config.FRAME_STYLES,
                              index=Config.FRAME_STYLES.index(self.frame_style),
                              key="frame_style_selector")
            if fs != self.frame_style:
                st.session_state['frame_style'] = fs
                st.rerun()

            st.divider()
            people = self.manager.db.get_all_people()
            ti = tv = 0
            for p in people:
                s = self.manager.get_person_stats(p['person_id'])
                ti += s.get('image_count', 0)
                tv += s.get('video_count', 0)
            st.metric("People", len(people))
            c1, c2 = st.columns(2)
            with c1: st.metric("Images", ti)
            with c2: st.metric("Videos", tv)

            st.divider()
            autoplay = st.toggle("Video Autoplay", value=st.session_state.get('video_autoplay', False))
            if autoplay != st.session_state.get('video_autoplay', False):
                st.session_state['video_autoplay'] = autoplay

    # ── ENHANCED MEDIA DETAIL PAGE (with Prev/Next, HD, thumb strip) ──
    def render_enhanced_media_detail_page(self):
        entry_id = st.session_state.get('selected_media')
        if not entry_id:
            st.error("No media selected")
            if st.button("Back to Gallery"):
                st.session_state['current_page'] = 'gallery'
                st.rerun()
            return

        # Get current entry details
        entry = self.manager.get_entry_with_details(entry_id)
        if not entry:
            st.error("Media not found")
            if st.button("Back"):
                st.session_state['current_page'] = 'gallery'
                st.rerun()
            return

        # Build navigation list if not already present or if current person changed
        person_id = entry.get('person_id')
        media_filter = st.session_state.get('media_filter', 'all')
        nav_list_key = f"nav_list_{person_id}_{media_filter}"
        if (not st.session_state.get('media_nav_list') or
            st.session_state.get('nav_list_key') != nav_list_key):
            # Fetch all entries for this person (or all if no person? For now, use person)
            all_entries = self.manager.get_all_entries_for_person(person_id, media_filter)
            st.session_state['media_nav_list'] = all_entries
            st.session_state['nav_list_key'] = nav_list_key
            # Find index of current entry
            idx = next((i for i, e in enumerate(all_entries) if e['entry_id'] == entry_id), 0)
            st.session_state['media_nav_index'] = idx

        nav_list = st.session_state['media_nav_list']
        current_idx = st.session_state['media_nav_index']
        current_entry = nav_list[current_idx] if nav_list and 0 <= current_idx < len(nav_list) else entry

        # Breadcrumb
        render_breadcrumb([("🏠 Home", "dashboard"), ("📁 Gallery", "gallery"),
                           (current_entry.get('caption', 'Media'), "media_detail")])

        # Back button
        if st.button("← Back to Gallery"):
            st.session_state['current_page'] = 'gallery'
            st.session_state['media_nav_list'] = []  # clear navigation cache
            st.rerun()

        # Main content area with Prev/Next and HD viewer
        st.markdown(f"### {current_entry.get('caption', 'Untitled')}")
        st.markdown(UIComponents.media_type_badge(current_entry.get('media_type', 'image')), unsafe_allow_html=True)
        st.markdown(f"👤 **{current_entry.get('display_name', 'Unknown')}**")

        # Navigation columns: Prev | Main | Next
        col_prev, col_main, col_next = st.columns([1, 8, 1])

        with col_prev:
            st.markdown("<div class='nav-btn-container'>", unsafe_allow_html=True)
            if current_idx > 0:
                if st.button("◀", key="prev_media", help="Previous (←)"):
                    st.session_state['media_nav_index'] = current_idx - 1
                    st.session_state['selected_media'] = nav_list[current_idx - 1]['entry_id']
                    st.rerun()
            else:
                st.markdown("<div style='opacity:0.3;font-size:28px;text-align:center;'>◀</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_main:
            if current_entry['media_type'] == MediaType.IMAGE.value:
                # HD image display
                media_path = Config.DATA_DIR / current_entry['filepath']
                hd_url = MediaProcessor.get_hd_data_url(media_path)
                if hd_url:
                    st.markdown(FrameRenderer.wrap_detail(hd_url, self.frame_style), unsafe_allow_html=True)
                else:
                    st.error("Unable to load HD image")

                # Action buttons under image
                act_cols = st.columns([1, 1, 1, 2])
                with act_cols[0]:
                    if st.button("🔍 Full Screen", use_container_width=True):
                        st.session_state['fullscreen_media'] = hd_url
                        st.rerun()
                with act_cols[1]:
                    if media_path.exists():
                        with open(media_path, 'rb') as f:
                            st.download_button("💾 Download", data=f.read(),
                                               file_name=media_path.name, use_container_width=True)
                with act_cols[2]:
                    fav = current_entry['entry_id'] in st.session_state.get('favorites', set())
                    if fav:
                        if st.button("⭐ Unfavorite", use_container_width=True):
                            self.manager.remove_from_favorites(current_entry['entry_id'])
                            st.rerun()
                    else:
                        if st.button("☆ Favorite", use_container_width=True):
                            self.manager.add_to_favorites(current_entry['entry_id'])
                            st.rerun()
                with act_cols[3]:
                    st.caption(f"📸 {current_entry.get('width', '?')}×{current_entry.get('height', '?')}  •  {current_entry.get('file_size', 0)//1024} KB")

            else:  # Video
                media_path = Config.DATA_DIR / current_entry['filepath']
                st.markdown("### 🎬 Video Player")
                if media_path.exists() and media_path.stat().st_size < Config.MAX_VIDEO_SIZE:
                    with open(media_path, 'rb') as f:
                        vdata = f.read()
                    mt, _ = mimetypes.guess_type(str(media_path))
                    st.video(vdata, format=mt or "video/mp4")
                else:
                    st.warning("Video too large or not found.")

        with col_next:
            st.markdown("<div class='nav-btn-container'>", unsafe_allow_html=True)
            if current_idx < len(nav_list) - 1:
                if st.button("▶", key="next_media", help="Next (→)"):
                    st.session_state['media_nav_index'] = current_idx + 1
                    st.session_state['selected_media'] = nav_list[current_idx + 1]['entry_id']
                    st.rerun()
            else:
                st.markdown("<div style='opacity:0.3;font-size:28px;text-align:center;'>▶</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Thumbnail strip
        if len(nav_list) > 1:
            st.divider()
            st.markdown("#### 🖼️ Navigate – click any thumbnail")
            thumb_cols = st.columns(min(len(nav_list), 8))
            for i in range(min(len(nav_list), 8)):
                entry_thumb = nav_list[i]
                with thumb_cols[i]:
                    # Get thumbnail URL
                    thumb_url = None
                    if entry_thumb.get('thumbnail_path'):
                        tp = Path(entry_thumb['thumbnail_path'])
                        if tp.exists():
                            thumb_url = MediaProcessor.get_media_data_url(tp)
                    elif entry_thumb.get('video_thumbnail_path'):
                        tp = Path(entry_thumb['video_thumbnail_path'])
                        if tp.exists():
                            thumb_url = MediaProcessor.get_media_data_url(tp)
                    if not thumb_url and entry_thumb['media_type'] == MediaType.IMAGE.value:
                        # generate small thumb directly from original
                        media_fp = Config.DATA_DIR / entry_thumb['filepath']
                        if media_fp.exists():
                            thumb_url = MediaProcessor.get_thumb_strip_url(media_fp, is_video=False)
                    if not thumb_url:
                        thumb_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='90' height='68' viewBox='0 0 90 68'%3E%3Crect width='90' height='68' fill='%23333'/%3E%3Ctext x='45' y='38' fill='%23fff' font-size='16' text-anchor='middle'%3E📸%3C/text%3E%3C/svg%3E"
                    active = (i == current_idx)
                    st.markdown(FrameRenderer.wrap_thumb_strip_item(thumb_url, active, entry_thumb['media_type'] == MediaType.VIDEO.value), unsafe_allow_html=True)
                    if st.button(f"{i+1}", key=f"thumb_{i}", help=entry_thumb.get('caption', ''), use_container_width=True):
                        st.session_state['media_nav_index'] = i
                        st.session_state['selected_media'] = entry_thumb['entry_id']
                        st.rerun()

            if len(nav_list) > 8:
                st.caption(f"... and {len(nav_list)-8} more. Use ◀ ▶ to browse all.")

        # Fullscreen overlay
        if st.session_state.get('fullscreen_media'):
            st.markdown(f"""
            <div class="fullscreen-overlay" onclick="this.style.display='none'">
                <img src="{st.session_state['fullscreen_media']}" alt="Fullscreen image">
            </div>
            <button class="close-fullscreen-btn" onclick="document.querySelector('.fullscreen-overlay').style.display='none'">✖ Close</button>
            """, unsafe_allow_html=True)
            if st.button("Exit Fullscreen", key="exit_fs"):
                st.session_state['fullscreen_media'] = None
                st.rerun()

        # Comments & Ratings (original features)
        st.divider()
        st.subheader("💬 Comments & Ratings")

        # Rating section
        avg_rating = current_entry.get('avg_rating', 0)
        rating_count = current_entry.get('rating_count', 0)
        st.markdown(UIComponents.rating_stars(avg_rating), unsafe_allow_html=True)
        st.caption(f"{rating_count} ratings")

        st.subheader("Your Rating")
        rcols = st.columns(5)
        for i in range(1, 6):
            with rcols[i-1]:
                if st.button(f"{i}⭐", key=f"rate_{current_entry['entry_id']}_{i}", use_container_width=True):
                    self.manager.add_rating_to_entry(current_entry['entry_id'], i)
                    st.rerun()

        # Comments form
        with st.form("add_comment_form_detail"):
            ct = st.text_area("Add a comment…", height=100, max_chars=Config.MAX_COMMENT_LENGTH)
            if st.form_submit_button("Post Comment") and ct.strip():
                if self.manager.add_comment_to_entry(current_entry['entry_id'], ct.strip()):
                    st.rerun()

        # Display comments
        for comment in current_entry.get('comments', []):
            with st.container(border=True):
                ca, cb = st.columns([1, 4])
                with ca:
                    st.markdown(f"**{comment.get('username', 'Anonymous')}**")
                    st.caption(comment.get('created_at', ''))
                with cb:
                    st.markdown(comment.get('content', ''))

        # Metadata expander
        with st.expander("📊 Metadata & EXIF"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Filename:** {current_entry.get('filename')}")
                st.markdown(f"**Format:** {current_entry.get('format')}")
                st.markdown(f"**File Size:** {current_entry.get('file_size', 0) / (1024*1024):.2f} MB")
                if current_entry.get('duration'):
                    m, s = int(current_entry['duration'] // 60), int(current_entry['duration'] % 60)
                    st.markdown(f"**Duration:** {m:02d}:{s:02d}")
                if current_entry.get('date_taken'):
                    st.markdown(f"**Date Taken:** {current_entry['date_taken']}")
            with col2:
                if current_entry.get('width') and current_entry.get('height'):
                    st.markdown(f"**Dimensions:** {current_entry['width']} × {current_entry['height']}")
                if current_entry.get('frame_rate'):
                    st.markdown(f"**Frame Rate:** {current_entry['frame_rate']:.1f} fps")
                if current_entry.get('tags'):
                    st.markdown("**Tags:**")
                    st.markdown(UIComponents.tag_badges(current_entry['tags']), unsafe_allow_html=True)
            if current_entry.get('exif_data'):
                st.markdown("**EXIF Data:**")
                for k, v in list(current_entry['exif_data'].items())[:10]:
                    st.caption(f"{k}: {v}")

    # ── DASHBOARD (unchanged) ─────────────────────────────────────────
    def render_dashboard(self):
        render_breadcrumb([("🏠 Home", "dashboard")])
        st.title("📊 Dashboard")
        st.markdown("Welcome to your photo & video album!")

        people = self.manager.db.get_all_people()
        tm = ti = tv = 0
        for p in people:
            s = self.manager.get_person_stats(p['person_id'])
            tm += s['media_count']
            ti += s['image_count']
            tv += s['video_count']

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("👥 People", len(people))
        with c2: st.metric("📸🎬 Total Media", tm)
        with c3:
            recent = self.manager.get_recent_entries(1)
            st.metric("🕐 Last Added", recent[0].get('caption', 'N/A') if recent else 'None')
        with c4:
            top = self.manager.get_top_rated_entries(1)
            st.metric("⭐ Top Rated", f"{top[0].get('avg_rating',0):.1f}/5" if top else "N/A")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Media Distribution")
            if tm > 0:
                st.bar_chart(pd.DataFrame({'Type': ['Images', 'Videos'], 'Count': [ti, tv]}).set_index('Type'))
        with c2:
            st.subheader("⚡ Quick Stats")
            ca, cb = st.columns(2)
            with ca: st.metric("Images", ti); st.metric("Videos", tv)
            with cb: st.metric("People", len(people))
            if tm > 0 and len(people) > 0:
                st.metric("Avg/Person", f"{tm/len(people):.1f}")

        st.divider()
        st.subheader("📅 Recent Activity")
        for entry in self.manager.get_recent_entries(5):
            with st.container():
                ca, cb = st.columns([1, 4])
                with ca:
                    thumb = None
                    for k in ['thumbnail_path', 'video_thumbnail_path']:
                        if entry.get(k):
                            tp = Path(entry[k])
                            if tp.exists():
                                thumb = self.manager.media_processor.get_media_data_url(tp)
                                break
                    if thumb:
                        st.markdown(FrameRenderer.wrap_thumbnail(
                            thumb, entry.get('caption', ''),
                            self.frame_style,
                            entry.get('media_type') == MediaType.VIDEO.value,
                            entry.get('duration')), unsafe_allow_html=True)
                    else:
                        st.markdown("🎬" if entry.get('media_type') == MediaType.VIDEO.value else "📸")
                with cb:
                    st.markdown(f"**{entry.get('caption','Untitled')}**")
                    st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')),
                                unsafe_allow_html=True)
                    st.caption(f"👤 {entry.get('display_name','Unknown')}")
                st.divider()

        st.divider()
        ca, cb, cc = st.columns(3)
        with ca:
            if st.button("📁 Scan for New Media", use_container_width=True):
                r = self.manager.scan_directory()
                st.success(f"Found {r['new_media']} new media!")
                st.rerun()
        with cb:
            if st.button("👥 Manage People", use_container_width=True):
                st.session_state['current_page'] = 'people'; st.rerun()
        with cc:
            if st.button("🎬 View Videos", use_container_width=True):
                st.session_state['current_page'] = 'videos'; st.rerun()

    # ── PEOPLE PAGE (unchanged) ───────────────────────────────────────
    def render_people_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("👥 People", "people")])
        st.title("👥 People")

        c1, c2 = st.columns([3, 1])
        with c1: sq = st.text_input("Search people…", key="people_search")
        with c2: sb = st.selectbox("Sort by", ["Name", "Recently Added", "Media Count"], key="people_sort")

        all_people = self.manager.get_all_people_with_stats()
        if sq:
            all_people = [p for p in all_people if sq.lower() in p['display_name'].lower()]
        if sb == "Name": all_people.sort(key=lambda x: x['display_name'])
        elif sb == "Media Count": all_people.sort(key=lambda x: x['media_count'], reverse=True)
        elif sb == "Recently Added": all_people.sort(key=lambda x: x.get('last_activity', ''), reverse=True)

        cols = st.columns(3)
        for idx, person in enumerate(all_people):
            with cols[idx % 3]:
                with st.container(border=True):
                    if person.get('profile_image_data'):
                        st.image(person['profile_image_data'], use_column_width=True)
                    else:
                        colors = ['#667eea', '#764ba2', '#f56565', '#48bb78', '#ed8936']
                        color = colors[hash(person['person_id']) % len(colors)]
                        st.markdown(
                            f'<div style="background:{color};height:150px;border-radius:10px;'
                            f'display:flex;align-items:center;justify-content:center;">'
                            f'<span style="color:#fff;font-size:48px;">{person["display_name"][0].upper()}</span></div>',
                            unsafe_allow_html=True)
                    st.subheader(person['display_name'])
                    st.caption(f"📁 {person['folder_name']}")
                    ca, cb, cc = st.columns(3)
                    with ca: st.metric("Media", person.get('media_count', 0))
                    with cb: st.metric("Images", person.get('image_count', 0))
                    with cc: st.metric("Videos", person.get('video_count', 0))
                    if st.button("View Gallery", key=f"view_{person['person_id']}", use_container_width=True):
                        st.session_state['selected_person'] = person['person_id']
                        st.session_state['current_page'] = 'gallery'
                        st.rerun()

        st.divider()
        with st.expander("➕ Add New Person"):
            with st.form("add_person_form"):
                c1, c2 = st.columns(2)
                with c1:
                    fn = st.text_input("Folder Name*")
                    dn = st.text_input("Display Name*")
                    rel = st.selectbox("Relationship", ["Family", "Friend", "Colleague", "Relative", "Other"])
                with c2:
                    bd = st.date_input("Birth Date", value=None)
                    ci = st.text_input("Contact Info")
                    bio = st.text_area("Bio", height=100)
                if st.form_submit_button("Add Person"):
                    if fn and dn:
                        pd_dir = Config.DATA_DIR / fn
                        pd_dir.mkdir(exist_ok=True)
                        pp = PersonProfile(person_id=str(uuid.uuid4()), folder_name=fn,
                                           display_name=dn, bio=bio, birth_date=bd,
                                           relationship=rel, contact_info=ci,
                                           social_links={}, profile_image=None,
                                           created_at=datetime.datetime.now())
                        try:
                            self.manager.db.add_person(pp)
                            st.success(f"Person '{dn}' added!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Fill in required fields (*)")

    # ── GALLERY PAGE (modified: "View Details" uses enhanced viewer) ──
    def render_gallery_page(self):
        sp = st.session_state.get('selected_person')
        if sp:
            people = self.manager.db.get_all_people()
            sel = next((p for p in people if p['person_id'] == sp), None)
            if sel:
                render_breadcrumb([("🏠 Home", "dashboard"), ("👥 People", "people"),
                                   (sel['display_name'], "gallery")])
                st.subheader(f"📁 Media of {sel['display_name']}")
                if st.button("← Back to All People"):
                    st.session_state['selected_person'] = None
                    st.rerun()
        else:
            render_breadcrumb([("🏠 Home", "dashboard"), ("📁 Gallery", "gallery")])
            st.title("📁 Media Gallery")

        with st.container():
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1: sq = st.text_input("Search media…", key="gallery_search")
            with c2:
                mf_opts = ["All", "Image", "Video"]
                cur = st.session_state.get('media_filter', 'all')
                di = {'all': 0, 'image': 1, 'video': 2}.get(cur, 0)
                mfd = st.selectbox("Media Type", mf_opts, key="mf_display", index=di)
                st.session_state['media_filter'] = {'All': 'all', 'Image': 'image', 'Video': 'video'}[mfd]
            with c3: vm = st.selectbox("View", ["Grid", "List"], key="view_mode")
            with c4: ipp = st.selectbox("Per Page", [12, 24, 48], key="ipp")

        page = st.session_state.get('gallery_page', 1)
        if sp:
            data = self.manager.get_entries_by_person(sp, page, sq, st.session_state['media_filter'])
        else:
            offset = (page - 1) * ipp
            with sqlite3.connect(self.manager.db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                conds, params = [], []
                if sq:
                    sp2 = f'%{sq}%'
                    conds.append("(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)")
                    params.extend([sp2, sp2, sp2])
                mf = st.session_state['media_filter']
                if mf != 'all':
                    conds.append("m.media_type = ?")
                    params.append(mf)
                where = " AND ".join(conds) if conds else "1=1"
                params.extend([ipp, offset])
                c.execute(f'''SELECT ae.*, p.display_name, m.filename, m.media_type,
                              m.thumbnail_path, m.video_thumbnail_path, m.duration,
                              (SELECT AVG(rating_value) FROM ratings r WHERE r.entry_id=ae.entry_id) as avg_rating
                              FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                              JOIN media m ON ae.media_id=m.media_id
                              WHERE {where} ORDER BY ae.created_at DESC LIMIT ? OFFSET ?''', params)
                entries = [dict(row) for row in c.fetchall()]
                cnt_params = []
                if sq:
                    sp2 = f'%{sq}%'
                    cnt_conds = ["(ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?)"]
                    cnt_params.extend([sp2, sp2, sp2])
                    if mf != 'all':
                        cnt_conds.append("m.media_type = ?")
                        cnt_params.append(mf)
                    cnt_where = " AND ".join(cnt_conds)
                else:
                    if mf != 'all':
                        cnt_where = "m.media_type = ?"
                        cnt_params.append(mf)
                    else:
                        cnt_where = "1=1"
                c.execute(f'SELECT COUNT(*) FROM album_entries ae JOIN media m ON ae.media_id=m.media_id WHERE {cnt_where}',
                          cnt_params)
                total = c.fetchone()[0]
                data = {'entries': entries, 'total_count': total,
                        'total_pages': max(1, math.ceil(total / ipp)), 'current_page': page}

        if vm == "Grid":
            cols = st.columns(4)
            for idx, entry in enumerate(data['entries']):
                with cols[idx % 4]:
                    self._render_gallery_item(entry)
        else:
            for entry in data['entries']:
                self._render_gallery_item_list(entry)

        if data['total_pages'] > 1:
            st.divider()
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                pnums = []
                for i in range(1, data['total_pages'] + 1):
                    if i == 1 or i == data['total_pages'] or abs(i - page) <= 2:
                        pnums.append(i)
                    elif pnums[-1] != "...":
                        pnums.append("...")
                btns = st.columns(len(pnums) + 2)
                with btns[0]:
                    if page > 1 and st.button("◀", key="prev_page"):
                        st.session_state.gallery_page = page - 1; st.rerun()
                for bi, pn in enumerate(pnums, 1):
                    with btns[bi]:
                        if pn == "...":
                            st.markdown("…")
                        elif pn == page:
                            st.markdown(f"**{pn}**")
                        elif st.button(str(pn), key=f"pg_{pn}"):
                            st.session_state.gallery_page = pn; st.rerun()
                with btns[-1]:
                    if page < data['total_pages'] and st.button("▶", key="next_page"):
                        st.session_state.gallery_page = page + 1; st.rerun()

    def _get_thumbnail_url(self, entry: Dict) -> Optional[str]:
        for key in ['thumbnail_path', 'video_thumbnail_path']:
            if entry.get(key):
                tp = Path(entry[key])
                if tp.exists():
                    return self.manager.media_processor.get_media_data_url(tp)
        return None

    def _render_gallery_item(self, entry: Dict):
        with st.container(border=True):
            st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')),
                        unsafe_allow_html=True)
            thumb = self._get_thumbnail_url(entry)
            is_vid = entry.get('media_type') == MediaType.VIDEO.value

            if thumb:
                st.markdown(FrameRenderer.wrap_thumbnail(
                    thumb, entry.get('caption', ''),
                    self.frame_style, is_vid, entry.get('duration')),
                    unsafe_allow_html=True)
            else:
                st.markdown("🎬" if is_vid else "📸")

            st.markdown(f"**{entry.get('caption', 'Untitled')}**")
            st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
            if entry.get('avg_rating'):
                st.markdown(UIComponents.rating_stars(entry['avg_rating'], size=15),
                            unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("👁️ View", key=f"view_{entry['entry_id']}", use_container_width=True):
                    # Use enhanced viewer
                    st.session_state['selected_media'] = entry['entry_id']
                    # Pre-load navigation list for this person/media filter
                    person_id = entry.get('person_id')
                    media_filter = st.session_state.get('media_filter', 'all')
                    all_entries = self.manager.get_all_entries_for_person(person_id, media_filter)
                    st.session_state['media_nav_list'] = all_entries
                    idx = next((i for i, e in enumerate(all_entries) if e['entry_id'] == entry['entry_id']), 0)
                    st.session_state['media_nav_index'] = idx
                    st.session_state['nav_list_key'] = f"nav_list_{person_id}_{media_filter}"
                    st.session_state['current_page'] = 'media_detail'
                    st.rerun()
            with c2:
                fav = entry['entry_id'] in st.session_state.get('favorites', set())
                if fav:
                    if st.button("⭐", key=f"ufav_{entry['entry_id']}", use_container_width=True):
                        self.manager.remove_from_favorites(entry['entry_id']); st.rerun()
                else:
                    if st.button("☆", key=f"fav_{entry['entry_id']}", use_container_width=True):
                        self.manager.add_to_favorites(entry['entry_id']); st.rerun()
            with c3:
                # download button for images
                if not is_vid and entry.get('filepath'):
                    mp = Config.DATA_DIR / entry['filepath']
                    if mp.exists():
                        with open(mp, 'rb') as f:
                            st.download_button("💾", data=f.read(),
                                               file_name=mp.name,
                                               key=f"dl_{entry['entry_id']}",
                                               use_container_width=True)

    def _render_gallery_item_list(self, entry: Dict):
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                thumb = self._get_thumbnail_url(entry)
                is_vid = entry.get('media_type') == MediaType.VIDEO.value
                if thumb:
                    st.markdown(FrameRenderer.wrap_thumbnail(
                        thumb, '', self.frame_style, is_vid, entry.get('duration')),
                        unsafe_allow_html=True)
                else:
                    st.markdown("🎬" if is_vid else "📸")
            with c2:
                st.markdown(f"### {entry.get('caption', 'Untitled')}")
                st.markdown(UIComponents.media_type_badge(entry.get('media_type', 'image')),
                            unsafe_allow_html=True)
                st.caption(f"👤 {entry.get('display_name', 'Unknown')}")
                if entry.get('tags'):
                    tl = entry['tags'].split(',') if isinstance(entry['tags'], str) else entry['tags']
                    st.markdown(UIComponents.tag_badges(tl, 3), unsafe_allow_html=True)
            with c3:
                if entry.get('avg_rating'):
                    st.markdown(UIComponents.rating_stars(entry['avg_rating']), unsafe_allow_html=True)
                if is_vid and entry.get('duration'):
                    m, s = int(entry['duration'] // 60), int(entry['duration'] % 60)
                    st.caption(f"⏱️ {m:02d}:{s:02d}")
                if st.button("View Details", key=f"vdet_{entry['entry_id']}", use_container_width=True):
                    # Use enhanced viewer
                    st.session_state['selected_media'] = entry['entry_id']
                    person_id = entry.get('person_id')
                    media_filter = st.session_state.get('media_filter', 'all')
                    all_entries = self.manager.get_all_entries_for_person(person_id, media_filter)
                    st.session_state['media_nav_list'] = all_entries
                    idx = next((i for i, e in enumerate(all_entries) if e['entry_id'] == entry['entry_id']), 0)
                    st.session_state['media_nav_index'] = idx
                    st.session_state['nav_list_key'] = f"nav_list_{person_id}_{media_filter}"
                    st.session_state['current_page'] = 'media_detail'
                    st.rerun()

    # ── VIDEOS PAGE (unchanged) ───────────────────────────────────────
    def render_videos_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("🎬 Videos", "videos")])
        st.title("🎬 Video Library")
        videos = self.manager.db.get_media_by_type(MediaType.VIDEO.value)
        if not videos:
            st.info("No videos found. Add some to person folders and scan.")
            return
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Videos", len(videos))
        with c2:
            td = sum(v.get('duration', 0) for v in videos)
            st.metric("Total Duration", f"{int(td//3600)}h {int((td%3600)//60)}m")
        with c3:
            ad = td / len(videos) if videos else 0
            st.metric("Avg Duration", f"{int(ad//60)}:{int(ad%60):02d}")
        with c4:
            ts = sum(v.get('file_size', 0) for v in videos) / (1024 * 1024)
            st.metric("Total Size", f"{ts:.1f} MB")
        st.divider()
        cols = st.columns(4)
        for idx, video in enumerate(videos):
            with cols[idx % 4]:
                with st.container(border=True):
                    if video.get('video_thumbnail_path'):
                        tp = Path(video['video_thumbnail_path'])
                        if tp.exists():
                            du = self.manager.media_processor.get_media_data_url(tp)
                            st.markdown(FrameRenderer.wrap_thumbnail(du, '', self.frame_style, True, video.get('duration')),
                                        unsafe_allow_html=True)
                    st.markdown(f"**{video.get('filename', 'Untitled')}**")
                    if video.get('duration'):
                        m, s = int(video['duration'] // 60), int(video['duration'] % 60)
                        st.caption(f"⏱️ {m:02d}:{s:02d}")
                    if video.get('file_size'):
                        st.caption(f"📦 {video['file_size']/(1024*1024):.1f} MB")
                    with sqlite3.connect(self.manager.db.db_path) as conn:
                        c = conn.cursor()
                        c.execute('SELECT entry_id FROM album_entries WHERE media_id=?', (video['media_id'],))
                        row = c.fetchone()
                    if row and st.button("Play", key=f"play_{video['media_id']}", use_container_width=True):
                        st.session_state['selected_media'] = row[0]
                        st.session_state['current_page'] = 'media_detail'
                        st.rerun()

    # ── PHOTOS PAGE (unchanged) ───────────────────────────────────────
    def render_photos_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("📸 Photos", "photos")])
        st.title("📸 Photo Library")
        images = self.manager.db.get_media_by_type(MediaType.IMAGE.value)
        if not images:
            st.info("No photos found. Add some to person folders and scan.")
            return
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Photos", len(images))
        with c2:
            ts = sum(i.get('file_size', 0) for i in images) / (1024 * 1024)
            st.metric("Total Size", f"{ts:.1f} MB")
        with c3:
            st.metric("Avg Size", f"{ts/len(images):.1f} MB" if images else "0")
        st.divider()
        cols = st.columns(4)
        for idx, image in enumerate(images):
            with cols[idx % 4]:
                with st.container(border=True):
                    if image.get('thumbnail_path'):
                        tp = Path(image['thumbnail_path'])
                        if tp.exists():
                            du = self.manager.media_processor.get_media_data_url(tp)
                            st.markdown(FrameRenderer.wrap_thumbnail(du, image.get('filename', ''),
                                                                      self.frame_style, False),
                                        unsafe_allow_html=True)
                    st.markdown(f"**{image.get('filename', 'Untitled')}**")
                    if image.get('width') and image.get('height'):
                        st.caption(f"📐 {image['width']}×{image['height']}")
                    with sqlite3.connect(self.manager.db.db_path) as conn:
                        c = conn.cursor()
                        c.execute('SELECT entry_id FROM album_entries WHERE media_id=?', (image['media_id'],))
                        row = c.fetchone()
                    if row and st.button("View", key=f"vp_{image['media_id']}", use_container_width=True):
                        st.session_state['selected_media'] = row[0]
                        st.session_state['current_page'] = 'media_detail'
                        st.rerun()

    # ── FAVORITES PAGE (unchanged) ────────────────────────────────────
    def render_favorites_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("⭐ Favorites", "favorites")])
        st.title("⭐ Favorites")
        favs = self.manager.get_user_favorites()
        if not favs:
            st.info("No favorites yet. Browse the gallery and click ☆ to add!")
            return
        cols = st.columns(4)
        for idx, entry in enumerate(favs):
            with cols[idx % 4]:
                self._render_gallery_item(entry)

    # ── SEARCH PAGE (unchanged) ───────────────────────────────────────
    def render_search_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("🔍 Search", "search")])
        st.title("🔍 Search")
        c1, c2 = st.columns([3, 1])
        with c1: sq = st.text_input("Search term", key="global_search")
        with c2: si = st.selectbox("Search in", ["All Fields", "Captions", "Descriptions", "Tags", "People"])

        if sq:
            with st.spinner("Searching…"):
                results = []
                if si in ["All Fields", "Captions", "Descriptions", "Tags"]:
                    results.extend(self.manager.db.search_entries(sq))
                if si in ["All Fields", "People"]:
                    for person in self.manager.db.get_all_people():
                        if sq.lower() in person['display_name'].lower():
                            with sqlite3.connect(self.manager.db.db_path) as conn:
                                conn.row_factory = sqlite3.Row
                                c = conn.cursor()
                                c.execute('''SELECT ae.*, p.display_name, m.filename, m.media_type,
                                             m.thumbnail_path, m.video_thumbnail_path
                                             FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                                             JOIN media m ON ae.media_id=m.media_id
                                             WHERE ae.person_id=? LIMIT 5''', (person['person_id'],))
                                results.extend(dict(row) for row in c.fetchall())
                seen = set()
                unique = []
                for r in results:
                    if r['entry_id'] not in seen:
                        seen.add(r['entry_id'])
                        unique.append(r)
                st.subheader(f"Found {len(unique)} results")
                for e in unique:
                    self._render_gallery_item_list(e)
        else:
            st.info("Enter a search term")

    # ── STATISTICS PAGE (unchanged) ───────────────────────────────────
    def render_statistics_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("📊 Statistics", "statistics")])
        st.title("📊 Statistics")
        all_people = self.manager.get_all_people_with_stats()
        if not all_people:
            st.info("No data. Scan directory first.")
            return
        tm = sum(p['media_count'] for p in all_people)
        ti = sum(p['image_count'] for p in all_people)
        tv = sum(p['video_count'] for p in all_people)
        tc = sum(p['comment_count'] for p in all_people)
        ars = [p['avg_rating'] for p in all_people if p['avg_rating'] > 0]
        ar = sum(ars) / len(ars) if ars else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total People", len(all_people))
        with c2: st.metric("Total Media", tm)
        with c3: st.metric("Total Comments", tc)
        with c4: st.metric("Avg Rating", f"{ar:.1f}")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Media Distribution")
            st.bar_chart(pd.DataFrame({'Type': ['Images', 'Videos'], 'Count': [ti, tv]}).set_index('Type'))
        with c2:
            st.subheader("👥 Media per Person")
            st.bar_chart(pd.DataFrame({
                'Person': [p['display_name'] for p in all_people],
                'Media': [p['media_count'] for p in all_people]
            }).set_index('Person'))

        st.divider()
        st.subheader("👥 People Details")
        td = [{'Name': p['display_name'], 'Total': p['media_count'],
               'Images': p['image_count'], 'Videos': p['video_count'],
               'Comments': p['comment_count'], 'Avg Rating': f"{p['avg_rating']:.1f}",
               'Last Activity': p.get('last_activity', 'Never')} for p in all_people]
        df = pd.DataFrame(td)
        st.dataframe(df, use_container_width=True)
        if st.button("📊 Export CSV"):
            st.download_button("Download", df.to_csv(index=False),
                               "album_stats.csv", "text/csv")

    # ── SETTINGS PAGE (unchanged) ─────────────────────────────────────
    def render_settings_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("⚙️ Settings", "settings")])
        st.title("⚙️ Settings")
        tabs = st.tabs(["Application", "User", "Database", "Advanced"])

        with tabs[0]:
            st.subheader("Application Settings")
            with st.expander("🖼️ Frame Style"):
                fs = st.selectbox("Choose frame style", Config.FRAME_STYLES,
                                  index=Config.FRAME_STYLES.index(self.frame_style))
                if st.button("Apply Frame Style"):
                    st.session_state['frame_style'] = fs
                    st.success("Frame style updated!")
                    st.rerun()
            with st.expander("📁 Directory Settings"):
                for label, path in [("Data", Config.DATA_DIR), ("Thumbnails", Config.THUMBNAIL_DIR),
                                    ("Video Thumbnails", Config.VIDEO_THUMBNAIL_DIR),
                                    ("Database", Config.DB_DIR)]:
                    st.info(f"**{label}:** {path}")
                if st.button("Create Missing Directories"):
                    Config.init_directories(); st.success("Done!")
            with st.expander("🖼️ Image Settings"):
                c1, c2 = st.columns(2)
                with c1:
                    nt = st.number_input("Thumbnail Width", value=Config.THUMBNAIL_SIZE[0], min_value=100, max_value=800)
                    Config.THUMBNAIL_SIZE = (int(nt), int(nt))
                with c2:
                    np_ = st.number_input("Preview Width", value=Config.PREVIEW_SIZE[0], min_value=400, max_value=1200)
                    Config.PREVIEW_SIZE = (int(np_), int(np_))
                if st.button("Apply"):
                    st.success("Updated!")
            with st.expander("🎬 Video Settings"):
                c1, c2 = st.columns(2)
                with c1:
                    nvt = st.number_input("Video Thumb Width", value=Config.VIDEO_THUMBNAIL_SIZE[0], min_value=100, max_value=800)
                    Config.VIDEO_THUMBNAIL_SIZE = (int(nvt), int(nvt))
                with c2:
                    mvs = st.number_input("Max Video MB", value=Config.MAX_VIDEO_SIZE // (1024 * 1024), min_value=10, max_value=500)
                    Config.MAX_VIDEO_SIZE = int(mvs) * 1024 * 1024
                if st.button("Apply Video Settings"):
                    st.success("Updated!")

        with tabs[1]:
            st.subheader("User Settings")
            nu = st.text_input("Username", value=st.session_state['username'])
            nr = st.selectbox("Role", [r.value for r in UserRoles],
                              index=[r.value for r in UserRoles].index(st.session_state['user_role']))
            if st.button("Update User"):
                st.session_state['username'] = nu
                st.session_state['user_role'] = nr
                st.success("Updated!")

        with tabs[2]:
            st.subheader("Database Management")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Rebuild Database"):
                    self.manager.db._init_database(); st.success("Rebuilt!")
                if st.button("📊 Optimize"):
                    with self.manager.db.get_connection() as conn:
                        conn.execute("VACUUM"); conn.execute("ANALYZE")
                    st.success("Optimized!")
            with c2:
                if st.button("🗑️ Clear All Data"):
                    if st.checkbox("I understand this deletes ALL data"):
                        try:
                            if self.manager.db.db_path.exists():
                                os.remove(self.manager.db.db_path)
                            self.manager.db = DatabaseManager()
                            self.manager.cache.clear()
                            st.success("Cleared!"); st.rerun()
                        except Exception as e:
                            st.error(str(e))
                dbs = self.manager.db.db_path.stat().st_size if self.manager.db.db_path.exists() else 0
                st.info(f"**DB Size:** {dbs/(1024*1024):.2f} MB")

        with tabs[3]:
            st.subheader("Advanced")
            with st.expander("💾 Cache"):
                st.info(f"Cache items: {len(self.manager.cache._cache)} | Video cache: {len(self.manager.cache._video_cache)}")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Clear All Cache"):
                        self.manager.cache.clear(); st.success("Cleared!")
                with c2:
                    if st.button("Clear Video Cache"):
                        self.manager.cache.clear_video_cache(); st.success("Cleared!")
            with st.expander("🐛 Debug"):
                if st.button("Show Session State"):
                    st.write(st.session_state)
                if st.button("Show Config"):
                    st.write({k: str(v) for k, v in vars(Config).items() if not k.startswith('_')})

    # ── IMPORT/EXPORT PAGE (unchanged) ────────────────────────────────
    def render_import_export_page(self):
        render_breadcrumb([("🏠 Home", "dashboard"), ("📤 Import/Export", "import_export")])
        st.title("📤 Import/Export")
        tabs = st.tabs(["Export", "Import", "Backup"])

        with tabs[0]:
            st.subheader("Export Data")
            ef = st.selectbox("Format", ["CSV", "JSON", "Excel"], key="export_format")
            ep = st.multiselect("People", [p['display_name'] for p in self.manager.db.get_all_people()])
            mt = st.multiselect("Media Types", ["Image", "Video"], default=["Image", "Video"])
            if st.button("Generate Export", type="primary"):
                with st.spinner("Generating…"):
                    try:
                        with sqlite3.connect(self.manager.db.db_path) as conn:
                            conds, params = [], []
                            if ep:
                                params.extend(ep)
                                conds.append(f"p.display_name IN ({','.join(['?']*len(ep))})")
                            if mt:
                                params.extend([m.lower() for m in mt])
                                conds.append(f"m.media_type IN ({','.join(['?']*len(mt))})")
                            where = " AND ".join(conds) if conds else "1=1"
                            df = pd.read_sql_query(f'''SELECT ae.caption, ae.description, ae.location,
                                                       ae.tags, p.display_name as person_name,
                                                       m.filename, m.media_type, m.file_size, m.format,
                                                       m.duration, m.width, m.height
                                                       FROM album_entries ae
                                                       JOIN people p ON ae.person_id=p.person_id
                                                       JOIN media m ON ae.media_id=m.media_id
                                                       WHERE {where} ORDER BY ae.created_at DESC''', conn, params=params)
                            buf = io.BytesIO()
                            ext = ef.lower()
                            if ext == 'csv':
                                buf.write(df.to_csv(index=False).encode())
                            elif ext == 'json':
                                buf.write(df.to_json(orient='records', indent=2).encode())
                            elif ext == 'excel':
                                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                                    df.to_excel(w, index=False)
                                ext = 'xlsx'
                            st.download_button(f"Download {ef}", buf.getvalue(),
                                               f"album_export_{datetime.datetime.now():%Y%m%d_%H%M%S}.{ext}",
                                               {"csv": "text/csv", "json": "application/json",
                                                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}[ext])
                    except Exception as e:
                        st.error(str(e))

        with tabs[1]:
            st.subheader("Import Data")
            st.info("CSV/JSON/Excel with columns: caption, person_name, filename, media_type")
            uf = st.file_uploader("Choose file", type=['csv', 'json', 'xlsx'], key="import_file")
            if uf and st.button("Import", type="primary"):
                with st.spinner("Importing…"):
                    try:
                        ext = uf.name.split('.')[-1].lower()
                        if ext == 'csv': df = pd.read_csv(io.BytesIO(uf.getvalue()))
                        elif ext == 'json': df = pd.read_json(io.BytesIO(uf.getvalue()))
                        else: df = pd.read_excel(io.BytesIO(uf.getvalue()))
                        req = ['caption', 'person_name', 'filename', 'media_type']
                        miss = [c for c in req if c not in df.columns]
                        if miss:
                            st.error(f"Missing columns: {miss}")
                        else:
                            ok = err = 0
                            for _, row in df.iterrows():
                                try:
                                    fn = row['person_name'].lower().replace(' ', '-')
                                    ep2 = self.manager.db.get_person_by_folder(fn)
                                    if not ep2:
                                        pp = PersonProfile(person_id=str(uuid.uuid4()), folder_name=fn,
                                                           display_name=row['person_name'],
                                                           bio=f"Photos of {row['person_name']}",
                                                           birth_date=None, relationship="Other",
                                                           contact_info="", social_links={},
                                                           profile_image=None, created_at=datetime.datetime.now())
                                        self.manager.db.add_person(pp)
                                        pid = pp.person_id
                                    else:
                                        pid = ep2['person_id']
                                    mp = Config.DATA_DIR / fn / row['filename']
                                    if not mp.exists():
                                        err += 1; continue
                                    meta = MediaMetadata.from_file(mp)
                                    th = vth = None
                                    if meta.media_type == MediaType.IMAGE.value:
                                        th = self.manager.media_processor.create_thumbnail(mp)
                                    else:
                                        vth = self.manager.media_processor.create_thumbnail(mp)
                                    self.manager.db.add_media(meta, str(th) if th else None, str(vth) if vth else None)
                                    tags = [t.strip() for t in row['tags'].split(',') if t.strip()] if pd.notna(row.get('tags')) else []
                                    ae = AlbumEntry(entry_id=str(uuid.uuid4()), media_id=meta.media_id,
                                                    person_id=pid, caption=row['caption'],
                                                    description=row.get('description', ''),
                                                    location=row.get('location', ''),
                                                    date_taken=row.get('date_taken'),
                                                    tags=tags, privacy_level=row.get('privacy_level', 'public'),
                                                    created_by=st.session_state['username'],
                                                    created_at=datetime.datetime.now(),
                                                    updated_at=datetime.datetime.now())
                                    self.manager.db.add_album_entry(ae)
                                    ok += 1
                                except Exception:
                                    err += 1
                            self.manager.cache.clear()
                            st.success(f"Imported: {ok} OK, {err} failed")
                            if ok: st.rerun()
                    except Exception as e:
                        st.error(str(e))

        with tabs[2]:
            st.subheader("Backup & Restore")
            c1, c2 = st.columns(2)
            with c1:
                bn = st.text_input("Backup Name", value=f"backup_{datetime.datetime.now():%Y%m%d_%H%M%S}")
                if st.button("🔒 Create Backup", use_container_width=True):
                    try:
                        import shutil
                        bp = Config.EXPORT_DIR / f"{bn}.db"
                        shutil.copy2(self.manager.db.db_path, bp)
                        with open(bp, 'rb') as f:
                            st.download_button("Download Backup", f.read(), bp.name, "application/x-sqlite3")
                    except Exception as e:
                        st.error(str(e))
            with c2:
                rf = st.file_uploader("Restore file", type=['db', 'sqlite', 'sqlite3'], key="restore_file")
                if rf and st.checkbox("⚠️ Overwrite current data"):
                    try:
                        import shutil
                        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        shutil.copy2(self.manager.db.db_path, Config.EXPORT_DIR / f"pre_restore_{ts}.db")
                        with open(self.manager.db.db_path, 'wb') as f:
                            f.write(rf.getvalue())
                        self.manager = AlbumManager()
                        st.success("Restored!"); st.rerun()
                    except Exception as e:
                        st.error(str(e))

    # ── MAIN RENDERER ─────────────────────────────────────────────────
    def render_main(self):
        FrameRenderer.inject_global_css()
        self.render_sidebar()
        page = st.session_state.get('current_page', 'dashboard')
        page_map = {
            'dashboard': self.render_dashboard,
            'people': self.render_people_page,
            'gallery': self.render_gallery_page,
            'media_detail': self.render_enhanced_media_detail_page,  # changed
            'videos': self.render_videos_page,
            'photos': self.render_photos_page,
            'favorites': self.render_favorites_page,
            'search': self.render_search_page,
            'statistics': self.render_statistics_page,
            'settings': self.render_settings_page,
            'import_export': self.render_import_export_page,
        }
        page_map.get(page, self.render_dashboard)()

        st.divider()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
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
            st.error("Application failed to initialize.")
            return
        if not VIDEO_SUPPORT:
            st.warning("⚠️ Video libraries not installed. `pip install opencv-python moviepy`")
        app.render_main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        with st.expander("Error Details"):
            st.exception(e)
        if st.button("Try Again"):
            st.rerun()


if __name__ == "__main__":
    main()
