"""
COMPREHENSIVE WEB PHOTO & VIDEO ALBUM APPLICATION
Version: 7.0.0 - Full Feature Suite + Cipher Auth + Directory HD Viewer
Features: SQLite DB, Comments, Ratings, Favorites, Search, Statistics, 
          Import/Export, Settings, Dashboard + NEW Directory Sidebar, 
          HD Prev/Next Viewer, Luxury Frames, Clickable Thumbnail Strip
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ExifTags
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
_CIPHER = {'0': 'j', '1': 'a', '2': 'b', '3': 'c', '4': 'd',
           '5': 'e', '6': 'f', '7': 'g', '8': 'h', '9': 'i'}
_REV = {v: k for k, v in _CIPHER.items()}

# Stored as alphabet in code so the numeric password is never visible.
# a→1  i→9  h→8  g→7  j→0  e→5  j→0  e→5  =>  "19870505"
_ACCESS_CIPHER = "aihgjeje"
_REAL_PASSWORD = "".join(_REV.get(ch, ch) for ch in _ACCESS_CIPHER)


def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.markdown("""
    <style>
    .login-bg{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
              padding:60px 30px;border-radius:24px;text-align:center;
              box-shadow:0 24px 80px rgba(0,0,0,.5);margin:40px 0;}
    .login-title{font-size:2.6em;font-weight:800;
                 background:linear-gradient(90deg,#f9d423,#ff4e50);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
    .lock-icon{font-size:72px;margin-bottom:10px;}
    </style>
    <div class="login-bg">
        <div class="lock-icon">🔐</div>
        <div class="login-title">MemoryVault Pro+</div>
        <div style="color:#a0a0c0;font-size:1.1em;margin-bottom:30px;">Secure Photo &amp; Video Album</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Access Key", type="password", key="pwd_in",
                                 placeholder="Enter numeric access key", label_visibility="collapsed")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔓 Unlock", use_container_width=True, type="primary"):
                # ONLY numeric input unlocks. Alphabet 'aihgjeje' is rejected.
                if password.strip().isdigit() and password.strip() == _REAL_PASSWORD:
                    st.session_state.authenticated = True
                    st.success("✅ Access granted!")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error("❌ Invalid key.")
                    time.sleep(0.3)
        with col_b:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.authenticated = False
                st.rerun()
        with st.expander("🔑 Hint"):
            st.info("💡 The access key is a **numeric** code — letters will not work.")
            st.warning("🤔 Think of a personal 8-digit number you'd never forget.")
            st.caption("It's 8 digits. Only numbers are accepted.")
    return False


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    APP_NAME = "MemoryVault Pro+"
    VERSION = "7.0.0"
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
    HD_SIZE = (1920, 1080)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024
    MAX_VIDEO_SIZE = 100 * 1024 * 1024
    VIDEO_THUMBNAIL_SIZE = (300, 300)
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v']
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | set(SUPPORTED_VIDEO_FORMATS)
    ITEMS_PER_PAGE = 20
    MAX_COMMENT_LENGTH = 500
    CACHE_TTL = 3600
    FRAME_STYLES = ["Elegant Gold", "Polaroid", "Modern Shadow", "Dark Museum", "Vintage", "Gallery White"]
    DEFAULT_FRAME = "Elegant Gold"
    THUMB_STRIP_SIZE = (120, 90)

    @classmethod
    def init_directories(cls):
        for d in [cls.DATA_DIR, cls.THUMBNAIL_DIR, cls.VIDEO_THUMBNAIL_DIR,
                  cls.METADATA_DIR, cls.DB_DIR, cls.EXPORT_DIR, cls.VIDEO_CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_sample_structure()

    @classmethod
    def create_sample_structure(cls):
        for name in ["john-smith", "sarah-johnson", "michael-brown"]:
            pd_dir = cls.DATA_DIR / name
            pd_dir.mkdir(exist_ok=True)
            (pd_dir / "README.txt").write_text(f"Photos of {name.replace('-',' ').title()}\n")
            for i in range(1, 4):
                sp = pd_dir / f"photo_{i}.jpg"
                if not sp.exists():
                    try:
                        colors = ['#667eea', '#f56565', '#48bb78']
                        img = Image.new('RGB', (800, 600), color=colors[(i - 1) % 3])
                        draw = ImageDraw.Draw(img)
                        draw.rectangle((20, 20, 780, 580), outline='#fff', width=4)
                        draw.text((300, 280), f"{name.split('-')[0].title()} {i}", fill='#fff')
                        img.save(sp, 'JPEG', quality=90)
                    except Exception:
                        pass


class UserRoles(Enum):
    VIEWER = "viewer"; CONTRIBUTOR = "contributor"; EDITOR = "editor"; ADMIN = "admin"

class MediaType(Enum):
    IMAGE = "image"; VIDEO = "video"


# ============================================================================
# FRAME RENDERER & GLOBAL CSS
# ============================================================================
class FrameRenderer:
    @staticmethod
    def wrap_detail(src: str, style: str = "Elegant Gold") -> str:
        s = {
            "Elegant Gold": (
                'background:linear-gradient(135deg,#b8860b,#daa520,#ffd700,#daa520,#b8860b);padding:14px;border-radius:10px;box-shadow:0 20px 60px rgba(0,0,0,.4),inset 0 2px 0 rgba(255,255,255,.35);',
                'background:#fffff5;padding:20px;border-radius:6px;box-shadow:inset 0 0 24px rgba(0,0,0,.06);'),
            "Polaroid": ('background:#fff;padding:20px 20px 70px 20px;box-shadow:0 10px 36px rgba(0,0,0,.2);border-radius:3px;', ''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:16px;box-shadow:0 14px 48px rgba(0,0,0,.2);overflow:hidden;', ''),
            "Dark Museum": ('background:linear-gradient(160deg,#0d0d1a,#1a1a30);padding:28px;border-radius:16px;box-shadow:0 24px 72px rgba(0,0,0,.55);',
                            'background:#fffff8;padding:18px;border-radius:6px;'),
            "Vintage": ('background:linear-gradient(135deg,#d4b896,#e8d5b7);padding:16px;border-radius:6px;box-shadow:0 12px 36px rgba(0,0,0,.28);border:2px solid #a08050;',
                        'background:#faf5ee;padding:14px;border-radius:4px;'),
            "Gallery White": ('background:#fff;padding:24px;border-radius:4px;box-shadow:0 6px 24px rgba(0,0,0,.1);border:1px solid #e0e0e0;', ''),
        }
        outer, inner = s.get(style, s["Elegant Gold"])
        return f'<div style="{outer}"><div style="{inner}"><img src="{src}" style="width:100%;display:block;border-radius:4px;"></div></div>'

    @staticmethod
    def wrap_thumb(src: str, style: str = "Elegant Gold", active: bool = False) -> str:
        s = {
            "Elegant Gold": ('background:linear-gradient(135deg,#d4a574,#f0d9b5,#c9956b);padding:3px;border-radius:4px;', 'background:#fff;padding:2px;border-radius:2px;'),
            "Polaroid": ('background:#fff;padding:3px 3px 10px 3px;border-radius:2px;', ''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:6px;overflow:hidden;', ''),
            "Dark Museum": ('background:linear-gradient(145deg,#1a1a2e,#16213e);padding:4px;border-radius:4px;', 'background:#fff;padding:2px;border-radius:2px;'),
            "Vintage": ('background:linear-gradient(135deg,#e8d5b7,#d4b896);padding:3px;border-radius:3px;border:1px solid #c9a96e;', 'background:#faf5ee;padding:2px;border-radius:2px;'),
            "Gallery White": ('background:#fafafa;padding:3px;border-radius:2px;border:1px solid #e8e8e8;', ''),
        }
        outer, inner = s.get(style, s["Elegant Gold"])
        act = 'border:2px solid #667eea;box-shadow:0 0 12px rgba(102,126,234,.5);' if active else 'border:2px solid transparent;'
        return (f'<div style="{outer}{act}cursor:pointer;transition:all .2s;" onmouseover="this.style.transform=\'scale(1.05)\'" onmouseout="this.style.transform=\'scale(1)\'">'
                f'<div style="{inner}"><img src="{src}" style="width:100%;height:100%;object-fit:cover;display:block;border-radius:1px;"></div></div>')

    @staticmethod
    def inject_css():
        st.markdown("""
        <style>
        .stApp{scroll-behavior:smooth;}
        .breadcrumb{display:flex;gap:6px;font-size:13px;color:#888;margin-bottom:12px;flex-wrap:wrap;}
        .breadcrumb b{color:#333;}
        .thumb-strip{display:flex;gap:8px;overflow-x:auto;padding:8px 4px;scroll-behavior:smooth;}
        .thumb-strip::-webkit-scrollbar{height:6px;}
        .thumb-strip::-webkit-scrollbar-thumb{background:#667eea;border-radius:3px;}
        .thumb-item{min-width:90px;height:68px;border-radius:6px;overflow:hidden;flex-shrink:0;
                    cursor:pointer;transition:transform .2s,opacity .2s;opacity:.5;position:relative;border:2px solid transparent;}
        .thumb-item:hover{opacity:.85;transform:scale(1.05);}
        .thumb-item.active{opacity:1;transform:scale(1.08);border-color:#667eea;box-shadow:0 0 12px rgba(102,126,234,.4);}
        .thumb-item img{width:100%;height:100%;object-fit:cover;display:block;}
        .fs-overlay{position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,.95);z-index:9999;
                    display:flex;align-items:center;justify-content:center;cursor:zoom-out;}
        .fs-overlay img{max-width:98vw;max-height:96vh;object-fit:contain;border-radius:4px;}
        </style>
        """, unsafe_allow_html=True)

def render_breadcrumb(trail):
    parts = []
    for i, (label, _) in enumerate(trail):
        if i < len(trail) - 1: parts.append(f'<span>{label} ›</span>')
        else: parts.append(f'<b>{label}</b>')
    st.markdown(f'<div class="breadcrumb">{"".join(parts)}</div>', unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class MediaMetadata:
    media_id: str; filename: str; filepath: str; file_size: int; media_type: str
    dimensions: Tuple[int, int]; format: str; duration: Optional[float]; frame_rate: Optional[float]
    created_date: datetime.datetime; modified_date: datetime.datetime; exif_data: Optional[Dict]; checksum: str

    @classmethod
    def from_file(cls, fp: Path) -> 'MediaMetadata':
        if not fp.exists(): raise FileNotFoundError(str(fp))
        mt = 'video' if fp.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS else 'image'
        stats = fp.stat()
        if mt == 'image':
            with Image.open(fp) as img:
                return cls(str(uuid.uuid4()), fp.name, str(fp.relative_to(Config.DATA_DIR)), stats.st_size, mt,
                           img.size, img.format, None, None, datetime.datetime.fromtimestamp(stats.st_ctime),
                           datetime.datetime.fromtimestamp(stats.st_mtime), cls._exif(img), cls._cksum(fp))
        dims, dur, fr = (0,0), 0.0, 0.0
        if VIDEO_SUPPORT:
            try:
                c = VideoFileClip(str(fp)); dims, dur, fr = c.size, c.duration, c.fps; c.close()
            except: pass
        return cls(str(uuid.uuid4()), fp.name, str(fp.relative_to(Config.DATA_DIR)), stats.st_size, mt,
                   dims, fp.suffix[1:].upper(), dur, fr, datetime.datetime.fromtimestamp(stats.st_ctime),
                   datetime.datetime.fromtimestamp(stats.st_mtime), None, cls._cksum(fp))

    @staticmethod
    def _exif(img):
        try:
            e = {}
            if hasattr(img, '_getexif') and img._getexif():
                for tid, v in img._getexif().items():
                    t = ExifTags.TAGS.get(tid, tid)
                    if not isinstance(v, (bytes, np.ndarray)): e[t] = str(v)
            return e or None
        except: return None

    @staticmethod
    def _cksum(fp):
        h = hashlib.md5()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
        return h.hexdigest()

@dataclass
class AlbumEntry:
    entry_id: str; media_id: str; person_id: str; caption: str; description: str; location: str
    date_taken: Optional[datetime.datetime]; tags: List[str]; privacy_level: str; created_by: str
    created_at: datetime.datetime; updated_at: datetime.datetime
    def to_dict(self):
        d = asdict(self); d['date_taken'] = self.date_taken.isoformat() if self.date_taken else None
        d['created_at'] = self.created_at.isoformat(); d['updated_at'] = self.updated_at.isoformat(); return d

@dataclass
class Comment:
    comment_id: str; entry_id: str; user_id: str; username: str; content: str
    created_at: datetime.datetime; is_edited: bool; parent_comment_id: Optional[str]

@dataclass
class Rating:
    rating_id: str; entry_id: str; user_id: str; rating_value: int
    created_at: datetime.datetime; updated_at: datetime.datetime

@dataclass
class PersonProfile:
    person_id: str; folder_name: str; display_name: str; bio: str; birth_date: Optional[datetime.date]
    relationship: str; contact_info: str; social_links: Dict[str, str]; profile_image: Optional[str]
    created_at: datetime.datetime


# ============================================================================
# DATABASE MANAGER
# ============================================================================
class DatabaseManager:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Config.DB_FILE; self._init_db()

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path); conn.execute("PRAGMA foreign_keys = ON"); yield conn
        except sqlite3.Error as e: st.error(f"DB: {e}"); raise
        finally:
            if conn: conn.close()

    def _init_db(self):
        os.makedirs(self.db_path.parent, exist_ok=True)
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS media(
                media_id TEXT PRIMARY KEY,filename TEXT,filepath TEXT UNIQUE,file_size INTEGER,
                media_type TEXT,width INTEGER,height INTEGER,format TEXT,duration REAL,frame_rate REAL,
                created_date TIMESTAMP,modified_date TIMESTAMP,exif_data TEXT,checksum TEXT UNIQUE,
                thumbnail_path TEXT,video_thumbnail_path TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS people(
                person_id TEXT PRIMARY KEY,folder_name TEXT UNIQUE,display_name TEXT,bio TEXT,
                birth_date DATE,relationship TEXT,contact_info TEXT,social_links TEXT,
                profile_image TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS album_entries(
                entry_id TEXT PRIMARY KEY,media_id TEXT,person_id TEXT,caption TEXT,description TEXT,
                location TEXT,date_taken TIMESTAMP,tags TEXT,privacy_level TEXT DEFAULT 'public',
                created_by TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(media_id)REFERENCES media(media_id),
                FOREIGN KEY(person_id)REFERENCES people(person_id))''')
            c.execute('''CREATE TABLE IF NOT EXISTS comments(
                comment_id TEXT PRIMARY KEY,entry_id TEXT,user_id TEXT,username TEXT,content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,is_edited BOOLEAN DEFAULT 0,
                parent_comment_id TEXT,FOREIGN KEY(entry_id)REFERENCES album_entries(entry_id))''')
            c.execute('''CREATE TABLE IF NOT EXISTS ratings(
                rating_id TEXT PRIMARY KEY,entry_id TEXT,user_id TEXT,rating_value INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entry_id,user_id),FOREIGN KEY(entry_id)REFERENCES album_entries(entry_id))''')
            c.execute('''CREATE TABLE IF NOT EXISTS user_favorites(
                user_id TEXT,entry_id TEXT,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id,entry_id))''')
            for idx in ['idx_mt','idx_ae_m','idx_ae_p','idx_c_e','idx_r_e']:
                c.execute(f'CREATE INDEX IF NOT EXISTS {idx} ON ' + {
                    'idx_mt':'media(media_type)','idx_ae_m':'album_entries(media_id)',
                    'idx_ae_p':'album_entries(person_id)','idx_c_e':'comments(entry_id)',
                    'idx_r_e':'ratings(entry_id)'}[idx])
            conn.commit()

    def add_media(self, m: MediaMetadata, tp=None, vtp=None):
        with self.get_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO media(media_id,filename,filepath,file_size,media_type,
                width,height,format,duration,frame_rate,created_date,modified_date,exif_data,checksum,
                thumbnail_path,video_thumbnail_path) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (m.media_id,m.filename,m.filepath,m.file_size,m.media_type,m.dimensions[0],m.dimensions[1],
                 m.format,m.duration,m.frame_rate,m.created_date,m.modified_date,
                 json.dumps(m.exif_data) if m.exif_data else None,m.checksum,tp,vtp)); conn.commit()

    def add_person(self, p: PersonProfile):
        with self.get_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO people(person_id,folder_name,display_name,bio,birth_date,
                relationship,contact_info,social_links,profile_image) VALUES(?,?,?,?,?,?,?,?,?)''',
                (p.person_id,p.folder_name,p.display_name,p.bio,
                 p.birth_date.isoformat() if p.birth_date else None,p.relationship,p.contact_info,
                 json.dumps(p.social_links),p.profile_image)); conn.commit()

    def add_album_entry(self, e: AlbumEntry):
        with self.get_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO album_entries(entry_id,media_id,person_id,caption,
                description,location,date_taken,tags,privacy_level,created_by) VALUES(?,?,?,?,?,?,?,?,?,?)''',
                (e.entry_id,e.media_id,e.person_id,e.caption,e.description,e.location,e.date_taken,
                 ','.join(e.tags) if e.tags else None,e.privacy_level,e.created_by)); conn.commit()

    def add_comment(self, cm: Comment):
        with self.get_connection() as conn:
            conn.execute('''INSERT INTO comments(comment_id,entry_id,user_id,username,content,parent_comment_id)
                VALUES(?,?,?,?,?,?)''', (cm.comment_id,cm.entry_id,cm.user_id,cm.username,cm.content,cm.parent_comment_id)); conn.commit()

    def add_rating(self, r: Rating):
        with self.get_connection() as conn:
            conn.execute('''INSERT OR REPLACE INTO ratings(rating_id,entry_id,user_id,rating_value)
                VALUES(?,?,?,?)''', (r.rating_id,r.entry_id,r.user_id,r.rating_value)); conn.commit()

    def get_media_by_filepath(self, fp: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute('SELECT * FROM media WHERE filepath=?', (fp,)).fetchone()
            return dict(r) if r else None

    def get_entry_by_media_id(self, mid: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute('''SELECT ae.*,p.display_name FROM album_entries ae
                JOIN people p ON ae.person_id=p.person_id WHERE ae.media_id=?''', (mid,)).fetchone()
            return dict(r) if r else None

    def get_entry_details(self, eid: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute('''SELECT ae.*,p.display_name,m.filename,m.filepath,m.media_type,m.file_size,
                m.format,m.duration,m.width,m.height,m.thumbnail_path,m.video_thumbnail_path,
                (SELECT AVG(rating_value)FROM ratings WHERE entry_id=ae.entry_id)as avg_rating,
                (SELECT COUNT(*)FROM ratings WHERE entry_id=ae.entry_id)as rating_count
                FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                JOIN media m ON ae.media_id=m.media_id WHERE ae.entry_id=?''', (eid,)).fetchone()
            if r:
                res = dict(r); res['tags'] = [t.strip() for t in res['tags'].split(',') if t.strip()] if res.get('tags') else []
                return res
            return None

    def get_entry_comments(self, eid: str) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute('SELECT * FROM comments WHERE entry_id=? ORDER BY created_at DESC', (eid,)).fetchall()]

    def get_all_people(self) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute('SELECT * FROM people ORDER BY display_name').fetchall()]

    def get_person_by_folder(self, fn: str) -> Optional[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute('SELECT * FROM people WHERE folder_name=?', (fn,)).fetchone()
            return dict(r) if r else None

    def get_user_favorite(self, uid, eid) -> bool:
        with self.get_connection() as conn:
            return conn.execute('SELECT 1 FROM user_favorites WHERE user_id=? AND entry_id=?', (uid,eid)).fetchone() is not None

    def add_favorite(self, uid, eid):
        with self.get_connection() as conn:
            conn.execute('INSERT OR IGNORE INTO user_favorites(user_id,entry_id) VALUES(?,?)', (uid,eid)); conn.commit()

    def remove_favorite(self, uid, eid):
        with self.get_connection() as conn:
            conn.execute('DELETE FROM user_favorites WHERE user_id=? AND entry_id=?', (uid,eid)); conn.commit()

    def get_user_favorites(self, uid) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute('''SELECT ae.*,m.filename,m.media_type,m.thumbnail_path
                FROM user_favorites uf JOIN album_entries ae ON uf.entry_id=ae.entry_id
                JOIN media m ON ae.media_id=m.media_id WHERE uf.user_id=?''', (uid,)).fetchall()]

    def search_entries(self, q: str) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            sp = f'%{q}%'
            return [dict(r) for r in conn.execute('''SELECT ae.*,p.display_name,m.filename,m.media_type,
                m.thumbnail_path FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                JOIN media m ON ae.media_id=m.media_id
                WHERE ae.caption LIKE ? OR ae.description LIKE ? OR ae.tags LIKE ?''', (sp,sp,sp)).fetchall()]

    def get_recent_entries(self, limit=10) -> List[Dict]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute('''SELECT ae.*,p.display_name,m.filename,m.media_type,
                m.thumbnail_path FROM album_entries ae JOIN people p ON ae.person_id=p.person_id
                JOIN media m ON ae.media_id=m.media_id ORDER BY ae.created_at DESC LIMIT ?''', (limit,)).fetchall()]


# ============================================================================
# MEDIA PROCESSOR
# ============================================================================
class MediaProcessor:
    @staticmethod
    def get_hd_data_url(fp: Path) -> str:
        try:
            if not fp.exists(): return ""
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGB', img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[-1]) if img.mode in ('RGBA','LA') else bg.paste(img)
                    img = bg
                if img.width > Config.HD_SIZE[0] or img.height > Config.HD_SIZE[1]:
                    img.thumbnail(Config.HD_SIZE, Image.Resampling.LANCZOS)
                buf = io.BytesIO(); img.save(buf, format='JPEG', quality=95)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except: return MediaProcessor.get_data_url(fp)

    @staticmethod
    def get_data_url(fp: Path) -> str:
        try:
            if not fp.exists(): return ""
            mt, _ = mimetypes.guess_type(str(fp))
            if not mt: mt = 'image/jpeg' if fp.suffix.lower() in Config.IMAGE_EXTENSIONS else 'video/mp4'
            with open(fp, "rb") as f: return f"data:{mt};base64,{base64.b64encode(f.read()).decode()}"
        except: return ""

    @staticmethod
    def get_thumb_data_url(fp: Path, size=None) -> str:
        try:
            if not fp.exists(): return ""
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA','LA','P'):
                    bg = Image.new('RGB', img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[-1]) if img.mode in ('RGBA','LA') else bg.paste(img)
                    img = bg
                img.thumbnail(size or Config.THUMB_STRIP_SIZE, Image.Resampling.LANCZOS)
                buf = io.BytesIO(); img.save(buf, format='JPEG', quality=80)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except: return ""

    @staticmethod
    def create_thumbnail(fp: Path) -> Optional[Path]:
        td = Config.THUMBNAIL_DIR; os.makedirs(td, exist_ok=True)
        tp = td / f"{fp.stem}_thumb.jpg"
        try:
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA','LA','P'):
                    bg = Image.new('RGB', img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[-1]) if img.mode in ('RGBA','LA') else bg.paste(img)
                    img = bg
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(tp, 'JPEG', quality=85); return tp
        except: return None

    @staticmethod
    def create_video_thumbnail(fp: Path) -> Optional[Path]:
        if not VIDEO_SUPPORT: return None
        td = Config.VIDEO_THUMBNAIL_DIR; os.makedirs(td, exist_ok=True)
        tp = td / f"{fp.stem}_vthumb.jpg"
        try:
            cap = cv2.VideoCapture(str(fp))
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tf > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, tf // 2)
            ret, frame = cap.read(); cap.release()
            if ret and frame is not None:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(tp, 'JPEG', quality=85); return tp
        except: return None


# ============================================================================
# CACHE MANAGER
# ============================================================================
class CacheManager:
    def __init__(self): self._c = {}; self._t = {}; self._vc = {}
    def get(self, k, d=None):
        if k in self._c and time.time()-self._t[k] < Config.CACHE_TTL: return self._c[k]
        self._c.pop(k, None); self._t.pop(k, None); return d
    def set(self, k, v): self._c[k] = v; self._t[k] = time.time()
    def clear(self, k=None):
        if k: self._c.pop(k, None); self._t.pop(k, None)
        else: self._c.clear(); self._t.clear(); self._vc.clear()
    def get_or_set(self, k, fn):
        c = self.get(k)
        if c is not None: return c
        try: v = fn(); self.set(k, v); return v
        except: return None


# ============================================================================
# UI COMPONENTS
# ============================================================================
class UIComponents:
    @staticmethod
    def rating_stars(r, sz=20):
        if not r or r <= 0: r = 0
        full = int(r); half = r - full >= 0.5; stars = []
        for i in range(5):
            if i < full: stars.append('⭐')
            elif i == full and half: stars.append('⭐')
            else: stars.append('☆')
        return f'<div style="color:#FFD700;font-size:{sz}px;letter-spacing:1px;">{"".join(stars)} <span style="color:#666;font-size:14px;">{r:.1f}/5</span></div>'

    @staticmethod
    def tag_badges(tags):
        if not tags: return ""
        return ' '.join([f'<span style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:3px 10px;'
                         f'border-radius:16px;font-size:11px;margin:2px;">{t.replace("-"," ")}</span>' for t in tags])

    @staticmethod
    def media_type_badge(mt):
        if mt == 'video':
            return '<span style="background:linear-gradient(135deg,#FF416C,#FF4B2B);color:#fff;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700;">🎬 VIDEO</span>'
        return '<span style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700;">📸 IMAGE</span>'


# ============================================================================
# ALBUM MANAGER
# ============================================================================
class AlbumManager:
    def __init__(self):
        self.db = DatabaseManager(); self.cache = CacheManager(); self.mp = MediaProcessor(); self._init_ss()

    def _init_ss(self):
        if 'am_init' not in st.session_state:
            st.session_state.update({
                'am_init': True, 'user_id': str(uuid.uuid4()), 'username': 'Guest',
                'user_role': UserRoles.VIEWER.value, 'favorites': set(),
                'selected_folder': None, 'selected_file_index': 0,
                'frame_style': Config.DEFAULT_FRAME, 'fullscreen': False,
                'current_page': 'viewer', 'selected_media': None,
            })

    @property
    def fs(self): return st.session_state.get('frame_style', Config.DEFAULT_FRAME)
    @property
    def uid(self): return st.session_state.get('user_id', '')

    def scan_directory(self):
        dd = Config.DATA_DIR; res = {'new':0, 'updated':0, 'img':0, 'vid':0, 'err':[]}
        skip = {'thumbnails','video_thumbnails','video_cache','database','metadata','exports'}
        pdirs = [d for d in dd.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name not in skip]
        if not pdirs: Config.create_sample_structure(); pdirs = [d for d in dd.iterdir() if d.is_dir() and d.name not in skip]
        pb = st.progress(0); total = sum(1 for pd in pdirs for f in pd.iterdir() if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS)
        proc = 0
        for pdir in pdirs:
            dn = ' '.join(p.capitalize() for p in pdir.name.replace('-',' ').replace('_',' ').split())
            ep = self.db.get_person_by_folder(pdir.name)
            if not ep:
                pp = PersonProfile(str(uuid.uuid4()), pdir.name, dn, f"Photos of {dn}", None, "Other", "", {}, None, datetime.datetime.now())
                self.db.add_person(pp); pid = pp.person_id
            else: pid = ep['person_id']
            for mf in pdir.iterdir():
                if not mf.is_file() or mf.suffix.lower() not in Config.ALLOWED_EXTENSIONS: continue
                try:
                    proc += 1; pb.progress(proc / max(total,1))
                    cs = MediaMetadata._cksum(mf)
                    with sqlite3.connect(self.db.db_path) as conn:
                        if conn.execute('SELECT 1 FROM media WHERE checksum=?', (cs,)).fetchone():
                            res['updated'] += 1; continue
                    meta = MediaMetadata.from_file(mf)
                    th = vth = None
                    if meta.media_type == 'image': th = self.mp.create_thumbnail(mf); res['img'] += 1
                    else: vth = self.mp.create_video_thumbnail(mf); res['vid'] += 1
                    self.db.add_media(meta, str(th) if th else None, str(vth) if vth else None)
                    ae = AlbumEntry(str(uuid.uuid4()), meta.media_id, pid, mf.stem.replace('_',' ').title(),
                                    f"Media of {dn}", "", meta.created_date, [dn.lower().replace(' ','-'), meta.media_type, 'memory'],
                                    'public', 'system', datetime.datetime.now(), datetime.datetime.now())
                    self.db.add_album_entry(ae); res['new'] += 1
                except Exception as e: res['err'].append(str(e))
        pb.empty(); self.cache.clear(); return res

    def get_directory_tree(self) -> Dict[str, Dict]:
        tree = {}
        people = self.db.get_all_people()
        for p in people:
            pd = Config.DATA_DIR / p['folder_name']
            if not pd.exists(): continue
            files = []
            for f in sorted(pd.iterdir()):
                if not f.is_file() or f.suffix.lower() not in Config.ALLOWED_EXTENSIONS: continue
                mt = 'video' if f.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS else 'image'
                # Cross-reference DB
                rel_path = str(f.relative_to(Config.DATA_DIR))
                db_media = self.db.get_media_by_filepath(rel_path)
                entry = self.db.get_entry_by_media_id(db_media['media_id']) if db_media else None
                files.append({
                    'path': str(f), 'name': f.name, 'stem': f.stem, 'suffix': f.suffix.lower(),
                    'type': mt, 'size': f.stat().st_size, 'entry_id': entry['entry_id'] if entry else None,
                    'media_id': db_media['media_id'] if db_media else None,
                })
            if files:
                tree[p['folder_name']] = {
                    'display_name': p['display_name'], 'person_id': p['person_id'],
                    'files': files, 'image_count': sum(1 for f in files if f['type']=='image'),
                    'video_count': sum(1 for f in files if f['type']=='video'),
                }
        return tree

    def add_comment(self, eid, content):
        if not content.strip(): return False
        cm = Comment(str(uuid.uuid4()), eid, self.uid, st.session_state.username, content.strip(),
                     datetime.datetime.now(), False, None)
        self.db.add_comment(cm); return True

    def add_rating(self, eid, val):
        if val < 1 or val > 5: return False
        self.db.add_rating(Rating(str(uuid.uuid4()), eid, self.uid, val, datetime.datetime.now(), datetime.datetime.now()))
        return True

    def toggle_favorite(self, eid):
        if self.db.get_user_favorite(self.uid, eid):
            self.db.remove_favorite(self.uid, eid); st.session_state.favorites.discard(eid); return False
        else:
            self.db.add_favorite(self.uid, eid); st.session_state.favorites.add(eid); return True


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PhotoAlbumApp:
    def __init__(self):
        st.set_page_config(page_title=Config.APP_NAME, page_icon="🖼️", layout="wide", initial_sidebar_state="expanded")
        Config.init_directories()
        self.mgr = AlbumManager()
        FrameRenderer.inject_css()

    def _get_files(self) -> List[Dict]:
        folder = st.session_state.get('selected_folder')
        if folder:
            tree = self.mgr.get_directory_tree()
            return tree.get(folder, {}).get('files', [])
        return []

    def _get_current(self) -> Optional[Dict]:
        files = self._get_files()
        idx = st.session_state.get('selected_file_index', 0)
        return files[idx] if files and 0 <= idx < len(files) else None

    # ── SIDEBAR ───────────────────────────────────────────────────────
    def render_sidebar(self):
        with st.sidebar:
            st.title("🖼️ MemoryVault")
            st.caption(f"v{Config.VERSION}")
            st.divider()

            # Navigation
            st.subheader("🧭 Navigate")
            pages = [("📂 Viewer", "viewer"), ("📊 Dashboard", "dashboard"), ("👥 People", "people"),
                     ("⭐ Favorites", "favorites"), ("🔍 Search", "search"), ("⚙️ Settings", "settings")]
            for label, key in pages:
                if st.button(label, use_container_width=True, key=f"nav_{key}"):
                    st.session_state.current_page = key; st.rerun()

            st.divider()
            fsi = st.selectbox("🖼️ Frame", Config.FRAME_STYLES, index=Config.FRAME_STYLES.index(self.mgr.fs))
            if fsi != self.mgr.fs: st.session_state.frame_style = fsi

            st.divider()
            st.subheader("📂 Directories")
            if st.button("🔄 Scan & Refresh", use_container_width=True):
                with st.spinner("Scanning…"):
                    r = self.mgr.scan_directory()
                    st.success(f"New: {r['new']} | Img: {r['img']} | Vid: {r['vid']}")
                st.rerun()

            tree = self.mgr.get_directory_tree()
            for folder, info in tree.items():
                is_active = st.session_state.get('selected_folder') == folder
                exp = is_active  # auto-expand active folder
                with st.expander(f"{'📂' if exp else '📁'} {info['display_name']} (📸{info['image_count']} 🎬{info['video_count']})", expanded=exp):
                    if st.button(f"📂 Open Folder", key=f"fo_{folder}", use_container_width=True):
                        st.session_state.selected_folder = folder
                        st.session_state.selected_file_index = 0
                        st.session_state.current_page = 'viewer'
                        st.rerun()
                    for i, f in enumerate(info['files']):
                        icon = "🎬" if f['type'] == 'video' else "🖼️"
                        act = " **←**" if (is_active and st.session_state.get('selected_file_index',0) == i) else ""
                        if st.button(f"{icon} {f['stem'][:25]}{act}", key=f"fl_{folder}_{i}"):
                            st.session_state.selected_folder = folder
                            st.session_state.selected_file_index = i
                            st.session_state.current_page = 'viewer'
                            st.rerun()

            st.divider()
            total_i = sum(i['image_count'] for i in tree.values())
            total_v = sum(i['video_count'] for i in tree.values())
            c1, c2 = st.columns(2)
            with c1: st.metric("Images", total_i)
            with c2: st.metric("Videos", total_v)

    # ── HD VIEWER (Directory Mode) ────────────────────────────────────
    def render_viewer(self):
        files = self._get_files()
        current = self._get_current()
        if not files or not current:
            st.markdown("<div style='text-align:center;padding:80px;'><h2>🖼️ Select a folder from the sidebar</h2></div>", unsafe_allow_html=True)
            return

        idx = st.session_state.get('selected_file_index', 0)
        folder = st.session_state.get('selected_folder')
        tree = self.mgr.get_directory_tree()
        display_name = tree.get(folder, {}).get('display_name', folder)

        render_breadcrumb([("Home", ""), (display_name, ""), (current['name'], "")])

        # Top bar
        tc1, tc2, tc3 = st.columns([3, 2, 1])
        with tc1: st.markdown(f"### 📂 {display_name}")
        with tc2: st.markdown(f"<div style='text-align:center;padding-top:8px;color:#888;'>{current['name']}</div>", unsafe_allow_html=True)
        with tc3: st.markdown(f"<div style='text-align:right;padding-top:8px;'><span style='color:#667eea;font-weight:700;font-size:18px;'>{idx+1}</span><span style='color:#888;'> / {len(files)}</span></div>", unsafe_allow_html=True)
        st.divider()

        if current['type'] == 'image':
            self._render_image_viewer(current, idx, files)
        else:
            self._render_video_viewer(current, idx, files)

        st.divider()
        self._render_thumb_strip(files, idx)

    def _render_image_viewer(self, current, idx, files):
        fp = Path(current['path'])
        hd_url = MediaProcessor.get_hd_data_url(fp)

        nav_cols = st.columns([1, 16, 1])
        with nav_cols[0]:
            st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;min-height:300px;'>", unsafe_allow_html=True)
            if idx > 0:
                if st.button("◀", key="prev_btn", help="Previous"): st.session_state.selected_file_index = idx - 1; st.rerun()
            else: st.markdown("<div style='opacity:.2;font-size:28px;'>◀</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with nav_cols[1]:
            if hd_url: st.markdown(FrameRenderer.wrap_detail(hd_url, self.mgr.fs), unsafe_allow_html=True)
            else: st.error("Cannot load image")

            # Actions
            ac1, ac2, ac3, ac4 = st.columns([2, 1, 1, 1])
            with ac1: st.caption(f"🖼️ {current['name']} • {fp.stat().st_size/(1024*1024):.2f} MB")
            with ac2:
                if st.button("🔍 Full Screen", key="fs_btn", use_container_width=True):
                    st.session_state.fullscreen = True; st.rerun()
            with ac3:
                if fp.exists():
                    with open(fp, 'rb') as f:
                        st.download_button("💾 Save", data=f.read(), file_name=fp.name, key="dl_btn", use_container_width=True)
            with ac4:
                if current.get('entry_id'):
                    is_fav = current['entry_id'] in st.session_state.get('favorites', set())
                    lbl = "⭐ Unfav" if is_fav else "☆ Favorite"
                    if st.button(lbl, key="fav_btn", use_container_width=True):
                        self.mgr.toggle_favorite(current['entry_id']); st.rerun()

        with nav_cols[2]:
            st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;min-height:300px;'>", unsafe_allow_html=True)
            if idx < len(files) - 1:
                if st.button("▶", key="next_btn", help="Next"): st.session_state.selected_file_index = idx + 1; st.rerun()
            else: st.markdown("<div style='opacity:.2;font-size:28px;'>▶</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Fullscreen overlay
        if st.session_state.get('fullscreen') and hd_url:
            st.markdown(f"""<div class="fs-overlay" onclick="this.style.display='none'">
                <img src="{hd_url}"></div>""", unsafe_allow_html=True)
            if st.button("✖ Close"): st.session_state.fullscreen = False; st.rerun()

        # Comments & Ratings (if in DB)
        if current.get('entry_id'):
            self._render_comments_ratings(current['entry_id'], fp)

    def _render_video_viewer(self, current, idx, files):
        fp = Path(current['path'])
        nav_cols = st.columns([1, 16, 1])
        with nav_cols[0]:
            st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;min-height:200px;'>", unsafe_allow_html=True)
            if idx > 0:
                if st.button("◀", key="vprev"): st.session_state.selected_file_index = idx - 1; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with nav_cols[1]:
            st.markdown("### 🎬 Video Player")
            if fp.exists() and fp.stat().st_size < Config.MAX_VIDEO_SIZE:
                with open(fp, 'rb') as f: vdata = f.read()
                st.video(vdata)
            else: st.warning("Video unavailable")
            if current.get('entry_id'):
                self._render_comments_ratings(current['entry_id'], fp)
        with nav_cols[2]:
            st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;min-height:200px;'>", unsafe_allow_html=True)
            if idx < len(files) - 1:
                if st.button("▶", key="vnext"): st.session_state.selected_file_index = idx + 1; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    def _render_comments_ratings(self, eid, fp):
        entry = self.mgr.db.get_entry_details(eid)
        if not entry: return

        with st.expander("💬 Comments & ⭐ Ratings", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                avg = entry.get('avg_rating', 0) or 0
                cnt = entry.get('rating_count', 0) or 0
                st.markdown(UIComponents.rating_stars(avg), unsafe_allow_html=True)
                st.caption(f"{cnt} ratings")
                st.markdown("**Your Rating:**")
                rc = st.columns(5)
                for i in range(1, 6):
                    with rc[i-1]:
                        if st.button(f"{i}⭐", key=f"r_{i}_{eid}"):
                            self.mgr.add_rating(eid, i); st.rerun()
            with c2:
                comments = self.mgr.db.get_entry_comments(eid)
                st.markdown(f"**{len(comments)} Comments**")
                with st.form(f"cmt_{eid}"):
                    ct = st.text_area("Add comment…", height=60, max_chars=Config.MAX_COMMENT_LENGTH)
                    if st.form_submit_button("Post") and ct.strip():
                        self.mgr.add_comment(eid, ct); st.rerun()
                for c in comments[:10]:
                    st.markdown(f"**{c['username']}**: {c['content'][:100]}")
                    st.caption(c.get('created_at', ''))

    def _render_thumb_strip(self, files, current_idx):
        if not files: return
        st.markdown("#### 🖼️ Navigate")
        html = '<div class="thumb-strip">'
        for i, f in enumerate(files):
            fp = Path(f['path'])
            active = " active" if i == current_idx else ""
            if f['type'] == 'image':
                thumb_url = MediaProcessor.get_thumb_data_url(fp, Config.THUMB_STRIP_SIZE)
            else:
                vtp = Config.VIDEO_THUMBNAIL_DIR / f"{fp.stem}_vthumb.jpg"
                thumb_url = MediaProcessor.get_data_url(vtp) if vtp.exists() else ""
            if thumb_url:
                overlay = '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:18px;color:#fff;opacity:.8;">▶</div>' if f['type'] == 'video' else ''
                html += f'''<div class="thumb-item{active}" title="{f['stem']}">
                    <img src="{thumb_url}" alt="{f['stem']}">{overlay}</div>'''
            else:
                icon = "🎬" if f['type'] == 'video' else "🖼️"
                html += f'''<div class="thumb-item{active}"><div style="width:90px;height:68px;background:#2a2a3e;
                    display:flex;align-items:center;justify-content:center;font-size:20px;">{icon}</div></div>'''
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        # Clickable buttons for each thumb
        cols = st.columns(min(len(files), 20))
        for i in range(min(len(files), 20)):
            with cols[i]:
                if st.button(f"{i+1}", key=f"ts_{i}", help=files[i]['stem'][:30]):
                    st.session_state.selected_file_index = i; st.rerun()

    # ── DASHBOARD ─────────────────────────────────────────────────────
    def render_dashboard(self):
        render_breadcrumb([("Home", ""), ("Dashboard", "")])
        st.title("📊 Dashboard")
        people = self.mgr.db.get_all_people()
        recent = self.mgr.db.get_recent_entries(5)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("People", len(people))
        with c2: st.metric("Recent Media", len(recent))
        with c3:
            tree = self.mgr.get_directory_tree()
            total = sum(i['image_count'] + i['video_count'] for i in tree.values())
            st.metric("Total Media", total)
        st.divider()
        st.subheader("📅 Recent Activity")
        for e in recent:
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                with c1:
                    if e.get('thumbnail_path') and Path(e['thumbnail_path']).exists():
                        st.image(MediaProcessor.get_data_url(Path(e['thumbnail_path'])), width=80)
                    else: st.markdown("🖼️" if e.get('media_type') == 'image' else "🎬")
                with c2:
                    st.markdown(f"**{e.get('caption','')}**")
                    st.caption(f"👤 {e.get('display_name','')} | {e.get('media_type','').title()}")

    # ── PEOPLE ────────────────────────────────────────────────────────
    def render_people(self):
        render_breadcrumb([("Home", ""), ("People", "")])
        st.title("👥 People")
        people = self.mgr.db.get_all_people()
        cols = st.columns(3)
        for i, p in enumerate(people):
            with cols[i % 3]:
                with st.container(border=True):
                    colors = ['#667eea', '#764ba2', '#f56565', '#48bb78', '#ed8936']
                    c = colors[hash(p['person_id']) % len(colors)]
                    st.markdown(f'<div style="background:{c};height:120px;border-radius:10px;display:flex;align-items:center;justify-content:center;"><span style="color:#fff;font-size:48px;">{p["display_name"][0]}</span></div>', unsafe_allow_html=True)
                    st.subheader(p['display_name'])
                    st.caption(f"📁 {p['folder_name']}")
                    if st.button("📂 View Files", key=f"pv_{p['person_id']}", use_container_width=True):
                        st.session_state.selected_folder = p['folder_name']
                        st.session_state.selected_file_index = 0
                        st.session_state.current_page = 'viewer'
                        st.rerun()

    # ── FAVORITES ─────────────────────────────────────────────────────
    def render_favorites(self):
        render_breadcrumb([("Home", ""), ("Favorites", "")])
        st.title("⭐ Favorites")
        favs = self.mgr.db.get_user_favorites(self.mgr.uid)
        if not favs: st.info("No favorites yet."); return
        cols = st.columns(4)
        for i, f in enumerate(favs):
            with cols[i % 4]:
                with st.container(border=True):
                    if f.get('thumbnail_path') and Path(f['thumbnail_path']).exists():
                        st.image(MediaProcessor.get_data_url(Path(f['thumbnail_path'])), use_column_width=True)
                    st.markdown(f"**{f.get('caption','')}**")
                    if st.button("👁️ View", key=f"fv_{f['entry_id']}", use_container_width=True):
                        # Find folder and index for this file
                        tree = self.mgr.get_directory_tree()
                        for folder, info in tree.items():
                            for fi, file in enumerate(info['files']):
                                if file.get('entry_id') == f['entry_id']:
                                    st.session_state.selected_folder = folder
                                    st.session_state.selected_file_index = fi
                                    st.session_state.current_page = 'viewer'
                                    st.rerun()

    # ── SEARCH ────────────────────────────────────────────────────────
    def render_search(self):
        render_breadcrumb([("Home", ""), ("Search", "")])
        st.title("🔍 Search")
        sq = st.text_input("Search captions, descriptions, tags…")
        if sq:
            results = self.mgr.db.search_entries(sq)
            st.subheader(f"{len(results)} results")
            cols = st.columns(4)
            for i, r in enumerate(results):
                with cols[i % 4]:
                    with st.container(border=True):
                        if r.get('thumbnail_path') and Path(r['thumbnail_path']).exists():
                            st.image(MediaProcessor.get_data_url(Path(r['thumbnail_path'])), use_column_width=True)
                        st.markdown(f"**{r.get('caption','')}**")
                        st.caption(f"👤 {r.get('display_name','')}")
                        if st.button("View", key=f"sr_{r['entry_id']}", use_container_width=True):
                            tree = self.mgr.get_directory_tree()
                            for folder, info in tree.items():
                                for fi, file in enumerate(info['files']):
                                    if file.get('entry_id') == r['entry_id']:
                                        st.session_state.selected_folder = folder
                                        st.session_state.selected_file_index = fi
                                        st.session_state.current_page = 'viewer'
                                        st.rerun()

    # ── SETTINGS ──────────────────────────────────────────────────────
    def render_settings(self):
        render_breadcrumb([("Home", ""), ("Settings", "")])
        st.title("⚙️ Settings")
        tabs = st.tabs(["Application", "Database", "Advanced"])
        with tabs[0]:
            st.subheader("Frame Style")
            fs = st.selectbox("Frame", Config.FRAME_STYLES, index=Config.FRAME_STYLES.index(self.mgr.fs))
            if st.button("Apply Frame"): st.session_state.frame_style = fs; st.success("Updated!"); st.rerun()
            st.subheader("User Profile")
            nu = st.text_input("Username", value=st.session_state.username)
            if st.button("Update Name"): st.session_state.username = nu; st.success("Updated!")
        with tabs[1]:
            if st.button("Rebuild DB"): self.mgr.db._init_db(); st.success("Done!")
            if st.button("Optimize"):
                with self.mgr.db.get_connection() as conn: conn.execute("VACUUM"); conn.execute("ANALYZE")
                st.success("Optimized!")
            if st.button("Clear All Data"):
                if st.checkbox("Delete everything"):
                    if self.mgr.db.db_path.exists(): os.remove(self.mgr.db.db_path)
                    self.mgr.db = DatabaseManager(); self.mgr.cache.clear(); st.success("Cleared!"); st.rerun()
        with tabs[2]:
            if st.button("Clear Cache"): self.mgr.cache.clear(); st.success("Done!")
            if st.button("Session State"): st.write(st.session_state)

    # ── MAIN RENDERER ─────────────────────────────────────────────────
    def render_main(self):
        self.render_sidebar()
        page = st.session_state.get('current_page', 'viewer')
        if page == 'viewer': self.render_viewer()
        elif page == 'dashboard': self.render_dashboard()
        elif page == 'people': self.render_people()
        elif page == 'favorites': self.render_favorites()
        elif page == 'search': self.render_search()
        elif page == 'settings': self.render_settings()
        else: self.render_viewer()

        st.divider()
        st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} v{Config.VERSION}")


def main():
    if not check_password(): return
    try:
        app = PhotoAlbumApp()
        if not VIDEO_SUPPORT: st.sidebar.caption("⚠️ pip install opencv-python moviepy")
        app.render_main()
    except Exception as e:
        st.error(f"Error: {e}")
        with st.expander("Details"): import traceback; traceback.print_exc()
        if st.button("Retry"): st.rerun()

if __name__ == "__main__":
    main()
