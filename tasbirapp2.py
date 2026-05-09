"""
COMPREHENSIVE WEB PHOTO & VIDEO ALBUM APPLICATION
Version: 6.0.0 - Fixed Sidebar Tree, Image Fitting, Centered Nav, Reliable Fullscreen
Features: Directory Sidebar, HD Viewer, Prev/Next Navigation, Numeric Password,
          Fullscreen Fix, Proper Frame Scaling
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
# VIDEO PROCESSING (optional)
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
_REAL_PASSWORD = "19870505"

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
    .login-sub{color:#a0a0c0;font-size:1.1em;margin-bottom:30px;}
    .lock-icon{font-size:72px;margin-bottom:10px;}
    </style>
    <div class="login-bg">
        <div class="lock-icon">🔐</div>
        <div class="login-title">MemoryVault Pro+</div>
        <div class="login-sub">Secure Photo &amp; Video Album</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Access Key", type="password", key="pwd_in",
                                 placeholder="Enter 8‑digit numeric code",
                                 label_visibility="collapsed")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔓 Unlock", use_container_width=True, type="primary"):
                if password.strip().isdigit() and password.strip() == _REAL_PASSWORD:
                    st.session_state.authenticated = True
                    st.success("✅ Access granted!")
                    time.sleep(0.4)
                    st.rerun()
                else:
                    st.error("❌ Invalid numeric key.")
                    time.sleep(0.3)
        with col_b:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.authenticated = False
                st.rerun()
        with st.expander("🔑 Hint"):
            st.info("💡 The access key is a **numeric** code – 8 digits.")
            st.caption("Only digits 0‑9 are accepted.")
    return False


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    APP_NAME = "MemoryVault Pro+"
    VERSION = "6.0.0"
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    VIDEO_THUMBNAIL_DIR = BASE_DIR / "video_thumbnails"
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    VIDEO_CACHE_DIR = BASE_DIR / "video_cache"
    DB_FILE = DB_DIR / "album.db"
    THUMBNAIL_SIZE = (300, 300)
    HD_SIZE = (1920, 1080)
    MAX_VIDEO_SIZE = 100 * 1024 * 1024
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v']
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | set(SUPPORTED_VIDEO_FORMATS)
    CACHE_TTL = 3600
    FRAME_STYLES = ["Elegant Gold", "Polaroid", "Modern Shadow", "Dark Museum", "Vintage", "Gallery White"]
    DEFAULT_FRAME = "Elegant Gold"
    THUMB_STRIP_SIZE = (120, 90)

    @classmethod
    def init_directories(cls):
        for d in [cls.DATA_DIR, cls.THUMBNAIL_DIR, cls.VIDEO_THUMBNAIL_DIR,
                  cls.DB_DIR, cls.EXPORT_DIR, cls.VIDEO_CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_samples()

    @classmethod
    def create_samples(cls):
        for name in ["john-smith", "sarah-johnson", "michael-brown"]:
            pd = cls.DATA_DIR / name
            pd.mkdir(exist_ok=True)
            for i in range(1, 4):
                sp = pd / f"photo_{i}.jpg"
                if not sp.exists():
                    try:
                        colors = ['#667eea', '#f56565', '#48bb78']
                        img = Image.new('RGB', (600, 400), color=colors[(i - 1) % 3])
                        draw = ImageDraw.Draw(img)
                        draw.rectangle((20, 20, 580, 380), outline='#fff', width=3)
                        draw.text((200, 180), f"{name.split('-')[0].title()} Photo {i}",
                                  fill='#fff')
                        img.save(sp, 'JPEG', quality=90)
                    except Exception:
                        pass


# ============================================================================
# FRAME RENDERER (with improved image fitting)
# ============================================================================
class FrameRenderer:
    @staticmethod
    def wrap_detail(src: str, style: str = "Elegant Gold") -> str:
        s = {
            "Elegant Gold": (
                'background:linear-gradient(135deg,#b8860b,#daa520,#ffd700,#daa520,#b8860b);'
                'padding:14px;border-radius:10px;box-shadow:0 20px 60px rgba(0,0,0,.4),'
                'inset 0 2px 0 rgba(255,255,255,.35),inset 0 -2px 0 rgba(0,0,0,.2);',
                'background:#fffff5;padding:20px;border-radius:6px;box-shadow:inset 0 0 24px rgba(0,0,0,.06);'),
            "Polaroid": ('background:#fff;padding:20px 20px 70px 20px;'
                         'box-shadow:0 10px 36px rgba(0,0,0,.2);border-radius:3px;', ''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:16px;'
                              'box-shadow:0 14px 48px rgba(0,0,0,.2);overflow:hidden;', ''),
            "Dark Museum": ('background:linear-gradient(160deg,#0d0d1a,#1a1a30,#0d0d1a);padding:28px;'
                            'border-radius:16px;box-shadow:0 24px 72px rgba(0,0,0,.55),'
                            '0 0 0 1px rgba(255,255,255,.04);',
                            'background:#fffff8;padding:18px;border-radius:6px;box-shadow:inset 0 0 20px rgba(0,0,0,.04);'),
            "Vintage": ('background:linear-gradient(135deg,#d4b896,#e8d5b7,#c9a96e);padding:16px;'
                        'border-radius:6px;box-shadow:0 12px 36px rgba(0,0,0,.28),'
                        'inset 0 0 50px rgba(139,109,63,.12);border:2px solid #a08050;',
                        'background:#faf5ee;padding:14px;border-radius:4px;box-shadow:inset 0 0 14px rgba(0,0,0,.04);'),
            "Gallery White": ('background:#fff;padding:24px;border-radius:4px;'
                              'box-shadow:0 6px 24px rgba(0,0,0,.1);border:1px solid #e0e0e0;', ''),
        }
        outer, inner = s.get(style, s["Elegant Gold"])
        # Fix: image fits inside frame without stretching
        return f'''<div style="{outer}">
                    <div style="{inner}">
                        <img src="{src}" style="width:100%;max-height:75vh;object-fit:contain;display:block;margin:0 auto;border-radius:4px;">
                    </div>
                   </div>'''

    @staticmethod
    def wrap_thumb(src: str, style: str = "Elegant Gold", active: bool = False) -> str:
        s = {
            "Elegant Gold": (
                'background:linear-gradient(135deg,#d4a574,#f0d9b5,#c9956b);padding:3px;border-radius:4px;',
                'background:#fff;padding:2px;border-radius:2px;'),
            "Polaroid": ('background:#fff;padding:3px 3px 10px 3px;border-radius:2px;', ''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:6px;overflow:hidden;', ''),
            "Dark Museum": ('background:linear-gradient(145deg,#1a1a2e,#16213e);padding:4px;border-radius:4px;',
                            'background:#fff;padding:2px;border-radius:2px;'),
            "Vintage": ('background:linear-gradient(135deg,#e8d5b7,#d4b896);padding:3px;border-radius:3px;border:1px solid #c9a96e;',
                        'background:#faf5ee;padding:2px;border-radius:2px;'),
            "Gallery White": ('background:#fafafa;padding:3px;border-radius:2px;border:1px solid #e8e8e8;', ''),
        }
        outer, inner = s.get(style, s["Elegant Gold"])
        act = ('border:2px solid #667eea;box-shadow:0 0 12px rgba(102,126,234,.5);' if active
               else 'border:2px solid transparent;')
        return (f'<div style="{outer}{act}cursor:pointer;transition:border-color .2s,box-shadow .2s,transform .2s;"'
                f' onmouseover="this.style.transform=\'scale(1.05)\'"'
                f' onmouseout="this.style.transform=\'scale(1)\'">'
                f'<div style="{inner}"><img src="{src}" style="width:100%;height:100%;object-fit:cover;display:block;border-radius:1px;"></div></div>')

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
    def inject_css():
        st.markdown("""
        <style>
        .stApp{scroll-behavior:smooth;}
        /* Directory sidebar styles */
        .dir-folder{padding:8px 12px;border-radius:8px;cursor:pointer;display:flex;
                     align-items:center;gap:8px;font-size:14px;font-weight:500;
                     transition:background .15s,box-shadow .15s;margin-bottom:2px;
                     color:#e0e0e0;user-select:none;}
        .dir-folder:hover{background:rgba(102,126,234,.15);box-shadow:0 2px 8px rgba(0,0,0,.1);}
        .dir-folder.active{background:rgba(102,126,234,.25);color:#fff;font-weight:700;
                           box-shadow:0 2px 12px rgba(102,126,234,.2);}
        .dir-file{padding:5px 12px 5px 32px;border-radius:6px;cursor:pointer;display:flex;
                   align-items:center;gap:6px;font-size:12px;color:#a0a0a0;
                   transition:background .12s,color .12s;margin-bottom:1px;user-select:none;}
        .dir-file:hover{background:rgba(102,126,234,.1);color:#e0e0e0;}
        .dir-file.active{background:rgba(102,126,234,.2);color:#fff;font-weight:600;}
        .dir-count{font-size:10px;background:rgba(255,255,255,.1);padding:2px 6px;
                   border-radius:10px;color:#a0a0c0;margin-left:auto;}
        /* Centered navigation */
        .nav-container {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            min-height: 60vh;
        }
        .nav-btn {
            background: rgba(102,126,234,.9);
            color: #fff;
            border: none;
            border-radius: 50%;
            width: 56px;
            height: 56px;
            font-size: 28px;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .nav-btn:hover {
            background: #764ba2;
            transform: scale(1.1);
        }
        .nav-btn.disabled {
            opacity: 0.3;
            cursor: not-allowed;
            pointer-events: none;
        }
        /* Thumbnail strip */
        .thumb-strip{display:flex;gap:6px;overflow-x:auto;padding:8px 4px;
                     scroll-behavior:smooth;-webkit-overflow-scrolling:touch;}
        .thumb-strip::-webkit-scrollbar{height:6px;}
        .thumb-strip::-webkit-scrollbar-thumb{background:#667eea;border-radius:3px;}
        .thumb-strip::-webkit-scrollbar-track{background:rgba(255,255,255,.05);}
        .thumb-item{min-width:90px;height:68px;border-radius:6px;overflow:hidden;
                    cursor:pointer;flex-shrink:0;transition:transform .2s,opacity .2s;
                    opacity:.5;position:relative;}
        .thumb-item:hover{opacity:.85;transform:scale(1.05);}
        .thumb-item.active{opacity:1;transform:scale(1.08);}
        .thumb-item.active::after{content:'';position:absolute;bottom:0;left:0;right:0;
            height:3px;background:#667eea;border-radius:0 0 6px 6px;}
        .thumb-item img{width:100%;height:100%;object-fit:cover;display:block;}
        /* Fullscreen overlay (fixed, reliable) */
        .fs-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0,0,0,.96);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: zoom-out;
        }
        .fs-overlay img {
            max-width: 96vw;
            max-height: 96vh;
            object-fit: contain;
            border-radius: 6px;
            box-shadow: 0 0 80px rgba(0,0,0,.7);
        }
        .close-fs-btn {
            position: fixed;
            top: 20px;
            right: 30px;
            background: rgba(0,0,0,0.7);
            color: white;
            border: none;
            border-radius: 30px;
            padding: 8px 18px;
            font-size: 16px;
            cursor: pointer;
            z-index: 10000;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)


# ============================================================================
# DIRECTORY SCANNER (builds sidebar tree)
# ============================================================================
class DirectoryScanner:
    @staticmethod
    def scan(data_dir: Path = None) -> Dict[str, Dict]:
        data_dir = data_dir or Config.DATA_DIR
        tree = {}
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            Config.create_samples()
        skip = {'thumbnails', 'video_thumbnails', 'video_cache', 'database', 'metadata', 'exports'}
        for folder in sorted(data_dir.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.') or folder.name in skip:
                continue
            files = []
            for f in sorted(folder.iterdir()):
                if not f.is_file() or f.suffix.lower() not in Config.ALLOWED_EXTENSIONS:
                    continue
                media_type = 'video' if f.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS else 'image'
                files.append({
                    'path': str(f),
                    'name': f.name,
                    'stem': f.stem,
                    'suffix': f.suffix,
                    'type': media_type,
                    'size': f.stat().st_size,
                })
            if files:
                display = ' '.join(p.capitalize() for p in folder.name.replace('-', ' ').replace('_', ' ').split())
                tree[folder.name] = {
                    'display_name': display,
                    'folder_path': str(folder),
                    'files': files,
                    'image_count': sum(1 for f in files if f['type'] == 'image'),
                    'video_count': sum(1 for f in files if f['type'] == 'video'),
                }
        return tree


# ============================================================================
# MEDIA PROCESSOR (with HD and thumb strip helpers)
# ============================================================================
class MediaProcessor:
    @staticmethod
    def get_hd_data_url(fp: Path) -> str:
        try:
            if not fp.exists():
                return ""
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode in ('RGBA', 'LA'):
                        bg.paste(img, mask=img.split()[-1])
                    else:
                        bg.paste(img)
                    img = bg
                img.thumbnail(Config.HD_SIZE, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95, optimize=True)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return ""

    @staticmethod
    def get_thumb_strip_url(fp: Path, is_video: bool = False) -> str:
        try:
            if not fp.exists():
                return ""
            if is_video:
                vt = Config.VIDEO_THUMBNAIL_DIR / f"{fp.stem}_thumb.jpg"
                if vt.exists():
                    return MediaProcessor.get_data_url(vt)
                return ""
            else:
                with Image.open(fp) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        bg = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode in ('RGBA', 'LA'):
                            bg.paste(img, mask=img.split()[-1])
                        else:
                            bg.paste(img)
                        img = bg
                    img.thumbnail(Config.THUMB_STRIP_SIZE, Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG', quality=80)
                    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return ""

    @staticmethod
    def get_data_url(fp: Path) -> str:
        try:
            if not fp.exists():
                return ""
            mt, _ = mimetypes.guess_type(str(fp))
            if not mt:
                mm = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                      '.gif': 'image/gif', '.mp4': 'video/mp4', '.webm': 'video/webm'}
                mt = mm.get(fp.suffix.lower(), 'application/octet-stream')
            with open(fp, "rb") as f:
                return f"data:{mt};base64,{base64.b64encode(f.read()).decode('utf-8')}"
        except Exception:
            return ""

    @staticmethod
    def create_thumbnail(fp: Path) -> Optional[Path]:
        td = Config.THUMBNAIL_DIR
        os.makedirs(td, exist_ok=True)
        tp = td / f"{fp.stem}_thumb.jpg"
        try:
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA','LA') else None)
                    img = bg
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(tp, 'JPEG', quality=85)
            return tp
        except Exception:
            return None

    @staticmethod
    def create_video_thumbnail(fp: Path) -> Optional[Path]:
        if not VIDEO_SUPPORT:
            return None
        td = Config.VIDEO_THUMBNAIL_DIR
        os.makedirs(td, exist_ok=True)
        tp = td / f"{fp.stem}_thumb.jpg"
        try:
            cap = cv2.VideoCapture(str(fp))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(tp, 'JPEG', quality=85)
                return tp
        except Exception:
            pass
        return None


# ============================================================================
# DATABASE MODELS (simplified for tree view – we only need folder/file storage)
# ============================================================================
class DatabaseManager:
    def __init__(self):
        self.db_path = Config.DB_FILE
        self._init_db()

    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        Config.DB_DIR.mkdir(parents=True, exist_ok=True)
        with self.get_conn() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS media (
                media_id TEXT PRIMARY KEY, filename TEXT, filepath TEXT,
                file_size INTEGER, media_type TEXT, width INTEGER, height INTEGER,
                duration REAL, created_date TIMESTAMP, thumbnail_path TEXT
            )''')
            conn.execute('''CREATE TABLE IF NOT EXISTS people (
                person_id TEXT PRIMARY KEY, folder_name TEXT UNIQUE, display_name TEXT
            )''')
            conn.execute('''CREATE TABLE IF NOT EXISTS album_entries (
                entry_id TEXT PRIMARY KEY, media_id TEXT, person_id TEXT,
                caption TEXT, tags TEXT, created_at TIMESTAMP
            )''')
            # Additional tables for comments, ratings etc. (keep functionality)
            conn.execute('''CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY, entry_id TEXT, user_id TEXT,
                username TEXT, content TEXT, created_at TIMESTAMP
            )''')
            conn.execute('''CREATE TABLE IF NOT EXISTS ratings (
                rating_id TEXT PRIMARY KEY, entry_id TEXT, user_id TEXT,
                rating_value INTEGER, created_at TIMESTAMP
            )''')
            conn.execute('''CREATE TABLE IF NOT EXISTS user_favorites (
                user_id TEXT, entry_id TEXT, created_at TIMESTAMP,
                PRIMARY KEY (user_id, entry_id)
            )''')
            conn.commit()


# ============================================================================
# MAIN APPLICATION (with sidebar tree and centered navigation)
# ============================================================================
class PhotoAlbumApp:
    def __init__(self):
        self.setup_page_config()
        Config.init_directories()
        self.db = DatabaseManager()
        self._init_session()
        self.tree = {}
        self._refresh_tree()

    def setup_page_config(self):
        st.set_page_config(page_title=Config.APP_NAME, page_icon="🖼️", layout="wide",
                           initial_sidebar_state="expanded")

    def _init_session(self):
        defaults = {
            'selected_folder': None,
            'selected_file_index': 0,
            'frame_style': Config.DEFAULT_FRAME,
            'fullscreen': False,
            'dir_expanded': {},
            'show_info': True,
            'media_nav_list': [],
            'media_nav_index': 0,
            'current_page': 'viewer',   # only one main view
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _refresh_tree(self):
        self.tree = DirectoryScanner.scan()
        if st.session_state.selected_folder is None and self.tree:
            first = list(self.tree.keys())[0]
            st.session_state.selected_folder = first
            st.session_state.selected_file_index = 0
            st.session_state.dir_expanded[first] = True

    @property
    def fs(self):
        return st.session_state.get('frame_style', Config.DEFAULT_FRAME)

    def _get_current_files(self) -> List[Dict]:
        folder = st.session_state.selected_folder
        if folder and folder in self.tree:
            return self.tree[folder]['files']
        return []

    def _get_current_file(self) -> Optional[Dict]:
        files = self._get_current_files()
        idx = st.session_state.selected_file_index
        if files and 0 <= idx < len(files):
            return files[idx]
        return None

    # ── SIDEBAR: DIRECTORY TREE ───────────────────────────────────────
    def render_sidebar(self):
        with st.sidebar:
            st.title("🖼️ MemoryVault")
            st.caption(f"v{Config.VERSION}")
            st.divider()

            # Frame style selector
            fs_sel = st.selectbox("🖼️ Frame", Config.FRAME_STYLES,
                                  index=Config.FRAME_STYLES.index(self.fs), key="fs_sidebar")
            if fs_sel != self.fs:
                st.session_state.frame_style = fs_sel

            st.divider()
            st.subheader("📂 Directories")
            if st.button("🔄 Refresh", use_container_width=True):
                self._refresh_tree()
                st.rerun()
            st.markdown("---")

            # Render tree
            for folder_name, info in self.tree.items():
                is_active = st.session_state.selected_folder == folder_name
                is_expanded = st.session_state.dir_expanded.get(folder_name, is_active)
                icon = "📂" if is_expanded else "📁"
                active_cls = " active" if is_active else ""

                # Folder button
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(f"{icon} {info['display_name']}", key=f"fld_{folder_name}",
                                 use_container_width=True):
                        if st.session_state.selected_folder == folder_name:
                            # Toggle expand
                            st.session_state.dir_expanded[folder_name] = not is_expanded
                        else:
                            st.session_state.selected_folder = folder_name
                            st.session_state.selected_file_index = 0
                            st.session_state.dir_expanded[folder_name] = True
                        st.rerun()
                with col2:
                    st.caption(f"📸{info['image_count']}" + (f" 🎬{info['video_count']}" if info['video_count'] else ""))

                # Files list (if expanded)
                if st.session_state.dir_expanded.get(folder_name, False):
                    for fi, f in enumerate(info['files']):
                        ficon = "🎬" if f['type'] == 'video' else "🖼️"
                        fname = f['stem'][:28] + ("…" if len(f['stem']) > 28 else "")
                        is_file_active = (st.session_state.selected_folder == folder_name and
                                          st.session_state.selected_file_index == fi)
                        if st.button(f"  {ficon} {fname}", key=f"file_{folder_name}_{fi}",
                                     use_container_width=True):
                            st.session_state.selected_folder = folder_name
                            st.session_state.selected_file_index = fi
                            # Also preload navigation list for the enhanced viewer
                            st.session_state.media_nav_list = info['files']
                            st.session_state.media_nav_index = fi
                            st.rerun()

            # Stats
            st.divider()
            total_imgs = sum(info['image_count'] for info in self.tree.values())
            total_vids = sum(info['video_count'] for info in self.tree.values())
            st.metric("Folders", len(self.tree))
            c1, c2 = st.columns(2)
            with c1: st.metric("Images", total_imgs)
            with c2: st.metric("Videos", total_vids)

            st.divider()
            st.session_state.show_info = st.toggle("ℹ️ Show Info", value=st.session_state.show_info)

    # ── MAIN VIEWER (with centered Prev/Next and fullscreen fix) ───────
    def render_viewer(self):
        FrameRenderer.inject_css()

        files = self._get_current_files()
        current = self._get_current_file()

        if not files or not current:
            self._render_empty_state()
            return

        idx = st.session_state.selected_file_index
        folder_name = st.session_state.selected_folder
        folder_info = self.tree.get(folder_name, {})
        display_name = folder_info.get('display_name', folder_name)

        # Top bar
        st.markdown(f"### 📂 {display_name}  •  {current['name']}")
        st.markdown(f"<div style='text-align:right;color:#888;'>{idx+1} / {len(files)}</div>", unsafe_allow_html=True)
        st.divider()

        # Main area with centered Prev/Next
        # Use three columns: left (prev), middle (media), right (next)
        col_prev, col_mid, col_next = st.columns([1, 8, 1])

        with col_prev:
            # Center the button vertically using CSS
            st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
            if idx > 0:
                if st.button("◀", key="prev_btn_center", help="Previous"):
                    st.session_state.selected_file_index = idx - 1
                    st.session_state.media_nav_index = idx - 1
                    st.rerun()
            else:
                st.markdown("<div class='nav-btn disabled' style='opacity:0.3;'>◀</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_mid:
            if current['type'] == 'image':
                fp = Path(current['path'])
                hd_url = MediaProcessor.get_hd_data_url(fp)
                if hd_url:
                    st.markdown(FrameRenderer.wrap_detail(hd_url, self.fs), unsafe_allow_html=True)
                else:
                    st.error("Could not load image")

                # Action buttons under the image
                act_cols = st.columns([1,1,1,2])
                with act_cols[0]:
                    if st.button("🔍 Full Screen", use_container_width=True):
                        st.session_state.fullscreen = hd_url   # store fullscreen image URL
                        st.rerun()
                with act_cols[1]:
                    if fp.exists():
                        with open(fp, 'rb') as f:
                            st.download_button("💾 Download", data=f.read(),
                                               file_name=fp.name, use_container_width=True)
                with act_cols[2]:
                    # Simple favorite toggle (optional)
                    pass
                with act_cols[3]:
                    if st.session_state.show_info:
                        st.caption(f"📐 {fp.stat().st_size/(1024*1024):.2f} MB")

                # Info panel
                if st.session_state.show_info:
                    with st.expander("📷 Image Info", expanded=False):
                        try:
                            with Image.open(fp) as img:
                                w, h = img.size
                                st.markdown(f"**Dimensions:** {w} × {h} px")
                                st.markdown(f"**Format:** {img.format}")
                                st.markdown(f"**Modified:** {datetime.datetime.fromtimestamp(fp.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")
                        except Exception:
                            st.markdown("Info not available")

            else:  # video
                fp = Path(current['path'])
                if fp.exists() and fp.stat().st_size < Config.MAX_VIDEO_SIZE:
                    with open(fp, 'rb') as f:
                        vdata = f.read()
                    mt, _ = mimetypes.guess_type(str(fp))
                    st.video(vdata, format=mt or "video/mp4")
                else:
                    st.warning("Video too large or not found.")

        with col_next:
            st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
            if idx < len(files) - 1:
                if st.button("▶", key="next_btn_center", help="Next"):
                    st.session_state.selected_file_index = idx + 1
                    st.session_state.media_nav_index = idx + 1
                    st.rerun()
            else:
                st.markdown("<div class='nav-btn disabled' style='opacity:0.3;'>▶</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Thumbnail strip
        if len(files) > 1:
            st.divider()
            st.markdown("#### 🖼️ Navigate – click any thumbnail")
            # Show first 10 thumbnails in a strip
            thumb_cols = st.columns(min(len(files), 10))
            for i in range(min(len(files), 10)):
                f = files[i]
                fp = Path(f['path'])
                thumb_url = MediaProcessor.get_thumb_strip_url(fp, is_video=(f['type']=='video'))
                if not thumb_url:
                    thumb_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='90' height='68' viewBox='0 0 90 68'%3E%3Crect width='90' height='68' fill='%23333'/%3E%3Ctext x='45' y='38' fill='%23fff' font-size='16' text-anchor='middle'%3E📸%3C/text%3E%3C/svg%3E"
                active = (i == idx)
                with thumb_cols[i]:
                    st.markdown(FrameRenderer.wrap_thumb_strip_item(thumb_url, active, f['type']=='video'),
                                unsafe_allow_html=True)
                    if st.button(f"{i+1}", key=f"strip_btn_{i}", use_container_width=True):
                        st.session_state.selected_file_index = i
                        st.session_state.media_nav_index = i
                        st.rerun()
            if len(files) > 10:
                st.caption(f"... and {len(files)-10} more. Use ◀ ▶ to browse all.")

        # Fullscreen overlay (fixed)
        if st.session_state.fullscreen:
            fs_url = st.session_state.fullscreen
            st.markdown(f"""
            <div class="fs-overlay" id="fsOverlay">
                <img src="{fs_url}" alt="Fullscreen">
                <button class="close-fs-btn" id="closeFsBtn">✖ Close</button>
            </div>
            <script>
            document.getElementById('fsOverlay').addEventListener('click', function(e) {{
                if(e.target === this || e.target.id === 'closeFsBtn') {{
                    this.style.display = 'none';
                }}
            }});
            </script>
            """, unsafe_allow_html=True)
            # Force rerun to clear fullscreen after JS closes it?
            # We'll clear the state when user clicks close button via rerun flag
            if st.button("Exit Fullscreen", key="exit_fs_btn"):
                st.session_state.fullscreen = False
                st.rerun()

    def _render_empty_state(self):
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
            <div style="font-size:80px;">🖼️</div>
            <h2>Welcome to MemoryVault Pro+</h2>
            <p>Select a folder from the sidebar to browse your media.</p>
        </div>
        """, unsafe_allow_html=True)
        total_imgs = sum(info['image_count'] for info in self.tree.values())
        total_vids = sum(info['video_count'] for info in self.tree.values())
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Folders", len(self.tree))
        with c2: st.metric("Images", total_imgs)
        with c3: st.metric("Videos", total_vids)

    # ── MAIN LOOP ─────────────────────────────────────────────────────
    def run(self):
        self.render_sidebar()
        self.render_viewer()
        st.divider()
        st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} v{Config.VERSION}")


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    if not check_password():
        return
    app = PhotoAlbumApp()
    app.run()


if __name__ == "__main__":
    main()
