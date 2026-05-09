"""
COMPREHENSIVE WEB PHOTO & VIDEO ALBUM APPLICATION
Version: 6.0.0 - Directory Sidebar, HD Viewer, Prev/Next Navigation
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFilter
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
import random
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

# Stored as alphabet so the numeric password is never visible in code
# a→1  i→9  h→8  g→7  j→0  e→5  j→0  e→5  decodes to "19870505"
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
                                 placeholder="Enter your numeric access key",
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
                                  fill='#fff', font=None)
                        img.save(sp, 'JPEG', quality=90)
                    except Exception:
                        pass


# ============================================================================
# FRAME RENDERER
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
        return f'<div style="{outer}"><div style="{inner}"><img src="{src}" style="width:100%;display:block;border-radius:4px;"></div></div>'

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
        /* Viewer styles */
        .viewer-container{display:flex;flex-direction:column;align-items:center;gap:12px;}
        .nav-btn{background:rgba(102,126,234,.9);color:#fff;border:none;padding:14px 22px;
                 border-radius:50%;cursor:pointer;font-size:24px;font-weight:700;
                 transition:background .2s,transform .15s,box-shadow .2s;
                 box-shadow:0 4px 16px rgba(0,0,0,.2);width:56px;height:56px;
                 display:flex;align-items:center;justify-content:center;}
        .nav-btn:hover{background:#764ba2;transform:scale(1.1);box-shadow:0 6px 24px rgba(0,0,0,.3);}
        .nav-btn:disabled{opacity:.3;cursor:not-allowed;transform:none;}
        /* Thumb strip */
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
        /* Info bar */
        .info-bar{display:flex;align-items:center;justify-content:space-between;
                  width:100%;padding:8px 16px;background:rgba(0,0,0,.03);border-radius:8px;}
        .counter{color:#888;font-size:14px;font-weight:500;}
        /* Fullscreen */
        .fs-overlay{position:fixed;top:0;left:0;width:100vw;height:100vh;
                    background:rgba(0,0,0,.95);z-index:9999;display:flex;
                    align-items:center;justify-content:center;cursor:zoom-out;}
        .fs-overlay img{max-width:98vw;max-height:96vh;object-fit:contain;border-radius:4px;
                        box-shadow:0 0 80px rgba(0,0,0,.6);}
        </style>
        """, unsafe_allow_html=True)


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
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode in ('RGBA', 'LA'):
                        bg.paste(img, mask=img.split()[-1])
                    else:
                        bg.paste(img)
                    img = bg
                max_w, max_h = Config.HD_SIZE
                if img.width > max_w or img.height > max_h:
                    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return MediaProcessor.get_data_url(fp)

    @staticmethod
    def get_data_url(fp: Path) -> str:
        try:
            if not fp.exists(): return ""
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
    def get_thumb_data_url(fp: Path, size: Tuple[int, int] = None) -> str:
        try:
            if not fp.exists(): return ""
            size = size or Config.THUMB_STRIP_SIZE
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode in ('RGBA', 'LA'):
                        bg.paste(img, mask=img.split()[-1])
                    else:
                        bg.paste(img)
                    img = bg
                img.thumbnail(size, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=80)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except Exception:
            return ""

    @staticmethod
    def get_image_info(fp: Path) -> Dict:
        info = {'filename': fp.name, 'size_bytes': 0, 'width': 0, 'height': 0,
                'format': '', 'date': None, 'exif': {}}
        try:
            if not fp.exists(): return info
            stat = fp.stat()
            info['size_bytes'] = stat.st_size
            info['date'] = datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            with Image.open(fp) as img:
                info['width'] = img.width
                info['height'] = img.height
                info['format'] = img.format or ''
                try:
                    if hasattr(img, '_getexif') and img._getexif():
                        from PIL.ExifTags import TAGS
                        for tid, val in img._getexif().items():
                            tag = TAGS.get(tid, tid)
                            if not isinstance(val, (bytes, np.ndarray)):
                                info['exif'][tag] = str(val)
                except Exception:
                    pass
        except Exception:
            pass
        return info

    @staticmethod
    def create_thumbnail(fp: Path, td: Path = None) -> Optional[Path]:
        td = td or Config.THUMBNAIL_DIR
        os.makedirs(td, exist_ok=True)
        tp = td / f"{fp.stem}_thumb.jpg"
        try:
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode in ('RGBA', 'LA'):
                        bg.paste(img, mask=img.split()[-1])
                    else:
                        bg.paste(img)
                    img = bg
                img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                img.save(tp, 'JPEG', quality=85, optimize=True)
            return tp
        except Exception:
            return None

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
        except Exception:
            pass
        return None


# ============================================================================
# DIRECTORY SCANNER
# ============================================================================
class DirectoryScanner:
    """Scans the data directory and builds a tree of folders and files."""

    @staticmethod
    def scan(data_dir: Path = None) -> Dict[str, List[Dict]]:
        """Returns {folder_name: [{'path': ..., 'name': ..., 'type': 'image'|'video', ...}]}"""
        data_dir = data_dir or Config.DATA_DIR
        tree = {}
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            Config.create_samples()

        skip = {'thumbnails', 'video_thumbnails', 'video_cache', 'database', 'metadata', 'exports'}
        for d in sorted(data_dir.iterdir()):
            if not d.is_dir() or d.name.startswith('.') or d.name in skip:
                continue
            files = []
            for f in sorted(d.iterdir()):
                if not f.is_file() or f.suffix.lower() not in Config.ALLOWED_EXTENSIONS:
                    continue
                mt = 'video' if f.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS else 'image'
                files.append({
                    'path': str(f),
                    'name': f.name,
                    'stem': f.stem,
                    'suffix': f.suffix.lower(),
                    'type': mt,
                    'size': f.stat().st_size,
                })
            if files:
                display = ' '.join(p.capitalize() for p in d.name.replace('-', ' ').replace('_', ' ').split())
                tree[d.name] = {
                    'display_name': display,
                    'folder_path': str(d),
                    'files': files,
                    'image_count': sum(1 for f in files if f['type'] == 'image'),
                    'video_count': sum(1 for f in files if f['type'] == 'video'),
                }
        return tree


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PhotoAlbumApp:
    def __init__(self):
        self.setup_page_config()
        Config.init_directories()
        self._init_session()
        self.tree = {}
        self._refresh_tree()

    def setup_page_config(self):
        st.set_page_config(page_title=Config.APP_NAME, page_icon="🖼️", layout="wide",
                           initial_sidebar_state="expanded",
                           menu_items={'About': f"# {Config.APP_NAME} v{Config.VERSION}"})

    def _init_session(self):
        defaults = {
            'selected_folder': None,
            'selected_file_index': 0,
            'frame_style': Config.DEFAULT_FRAME,
            'fullscreen': False,
            'dir_expanded': {},
            'show_info': True,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _refresh_tree(self):
        self.tree = DirectoryScanner.scan()

    @property
    def fs(self):
        return st.session_state.get('frame_style', Config.DEFAULT_FRAME)

    def _get_current_files(self) -> List[Dict]:
        """Get files in the currently selected folder."""
        folder = st.session_state.get('selected_folder')
        if folder and folder in self.tree:
            return self.tree[folder]['files']
        return []

    def _get_current_file(self) -> Optional[Dict]:
        """Get the currently selected file."""
        files = self._get_current_files()
        idx = st.session_state.get('selected_file_index', 0)
        if files and 0 <= idx < len(files):
            return files[idx]
        return None

    # ── SIDEBAR: DIRECTORY TREE ───────────────────────────────────────
    def render_sidebar(self):
        with st.sidebar:
            st.title("🖼️ MemoryVault")
            st.caption(f"v{Config.VERSION}")
            st.divider()

            # Frame style
            fsi = st.selectbox("🖼️ Frame", Config.FRAME_STYLES,
                               index=Config.FRAME_STYLES.index(self.fs), key="fs_sel")
            if fsi != self.fs:
                st.session_state['frame_style'] = fsi

            st.divider()
            st.subheader("📂 Directories")

            # Refresh button
            if st.button("🔄 Refresh", use_container_width=True):
                self._refresh_tree()
                st.rerun()

            st.markdown("---")

            # Auto-select first folder if none selected
            if st.session_state.selected_folder is None and self.tree:
                first = list(self.tree.keys())[0]
                st.session_state.selected_folder = first
                st.session_state.selected_file_index = 0
                st.session_state.setdefault('dir_expanded', {})[first] = True

            # Render directory tree
            for folder_name, info in self.tree.items():
                is_active = st.session_state.selected_folder == folder_name
                is_expanded = st.session_state.get('dir_expanded', {}).get(folder_name, is_active)
                icon = "📂" if is_expanded else "📁"
                active_cls = " active" if is_active else ""

                # Folder header
                folder_label = f"{icon} {info['display_name']}"
                count_label = f"📸{info['image_count']}" + (f" 🎬{info['video_count']}" if info['video_count'] else "")

                col_f1, col_f2 = st.columns([5, 1])
                with col_f1:
                    if st.button(folder_label, key=f"fld_{folder_name}", use_container_width=True):
                        st.session_state.selected_folder = folder_name
                        st.session_state.selected_file_index = 0
                        current_exp = st.session_state.get('dir_expanded', {}).get(folder_name, False)
                        st.session_state.setdefault('dir_expanded', {})[folder_name] = not current_exp
                        st.rerun()
                with col_f2:
                    st.caption(count_label)

                # Files inside folder (if expanded)
                if st.session_state.get('dir_expanded', {}).get(folder_name, is_active):
                    for fi, f in enumerate(info['files']):
                        ficon = "🎬" if f['type'] == 'video' else "🖼️"
                        fname = f['stem'][:28] + ("…" if len(f['stem']) > 28 else "")
                        is_file_active = (st.session_state.selected_folder == folder_name and
                                          st.session_state.selected_file_index == fi)

                        if st.button(f"  {ficon} {fname}", key=f"fl_{folder_name}_{fi}",
                                     use_container_width=True):
                            st.session_state.selected_folder = folder_name
                            st.session_state.selected_file_index = fi
                            st.rerun()

            # Stats
            st.divider()
            total_imgs = sum(info['image_count'] for info in self.tree.values())
            total_vids = sum(info['video_count'] for info in self.tree.values())
            st.metric("Folders", len(self.tree))
            c1, c2 = st.columns(2)
            with c1: st.metric("Images", total_imgs)
            with c2: st.metric("Videos", total_vids)

            # Toggle info
            st.divider()
            si = st.toggle("ℹ️ Show Info", value=st.session_state.get('show_info', True))
            st.session_state['show_info'] = si

    # ── MAIN VIEWER ───────────────────────────────────────────────────
    def render_viewer(self):
        FrameRenderer.inject_css()

        files = self._get_current_files()
        current = self._get_current_file()

        if not files or not current:
            self._render_empty_state()
            return

        idx = st.session_state.get('selected_file_index', 0)
        folder = st.session_state.get('selected_folder', '')
        folder_info = self.tree.get(folder, {})
        display_name = folder_info.get('display_name', folder)

        # ── TOP BAR ───────────────────────────────────────────────────
        top_c1, top_c2, top_c3 = st.columns([3, 2, 1])
        with top_c1:
            st.markdown(f"### 📂 {display_name}")
        with top_c2:
            st.markdown(f"<div style='text-align:center;padding-top:8px;'>"
                        f"<span style='color:#888;font-size:14px;'>"
                        f"{current['name']}</span></div>", unsafe_allow_html=True)
        with top_c3:
            st.markdown(f"<div style='text-align:right;padding-top:8px;'>"
                        f"<span style='color:#667eea;font-weight:700;font-size:16px;'>"
                        f"{idx + 1}</span>"
                        f"<span style='color:#888;font-size:14px;'> / {len(files)}</span>"
                        f"</div>", unsafe_allow_html=True)

        st.divider()

        # ── IMAGE/VIDEO DISPLAY WITH PREV/NEXT ────────────────────────
        if current['type'] == 'image':
            self._render_image_viewer(current, idx, files)
        else:
            self._render_video_viewer(current, idx, files)

        # ── THUMBNAIL STRIP ───────────────────────────────────────────
        st.divider()
        self._render_thumb_strip(files, idx)

    def _render_image_viewer(self, current: Dict, idx: int, files: List[Dict]):
        """Render HD image with prev/next navigation."""
        fp = Path(current['path'])
        hd_url = MediaProcessor.get_hd_data_url(fp)

        # Navigation + Image layout
        nav_cols = st.columns([1, 16, 1])

        # PREVIOUS button
        with nav_cols[0]:
            st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;min-height:300px;'>",
                        unsafe_allow_html=True)
            if idx > 0:
                if st.button("◀", key="prev_btn", help="Previous image (←)"):
                    st.session_state.selected_file_index = idx - 1
                    st.rerun()
            else:
                st.markdown("<div style='opacity:.2;font-size:28px;text-align:center;'>◀</div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # MAIN IMAGE with frame
        with nav_cols[1]:
            if hd_url:
                st.markdown(FrameRenderer.wrap_detail(hd_url, self.fs), unsafe_allow_html=True)
            else:
                st.error("Unable to load image")

            # Action bar under image
            act_c1, act_c2, act_c3, act_c4 = st.columns([2, 1, 1, 1])
            with act_c1:
                st.caption(f"🖼️ {current['name']}  •  {Path(current['path']).stat().st_size / (1024*1024):.2f} MB")
            with act_c2:
                if st.button("🔍 Full Screen", key="fs_btn", use_container_width=True):
                    st.session_state.fullscreen = True
                    st.rerun()
            with act_c3:
                if fp.exists():
                    with open(fp, 'rb') as f:
                        st.download_button("💾 Save", data=f.read(), file_name=fp.name,
                                           key="dl_btn", use_container_width=True)
            with act_c4:
                pass

        # NEXT button
        with nav_cols[2]:
            st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;min-height:300px;'>",
                        unsafe_allow_html=True)
            if idx < len(files) - 1:
                if st.button("▶", key="next_btn", help="Next image (→)"):
                    st.session_state.selected_file_index = idx + 1
                    st.rerun()
            else:
                st.markdown("<div style='opacity:.2;font-size:28px;text-align:center;'>▶</div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── INFO PANEL ────────────────────────────────────────────────
        if st.session_state.get('show_info', True):
            self._render_info_panel(current, fp)

        # ── FULLSCREEN OVERLAY ────────────────────────────────────────
        if st.session_state.get('fullscreen', False) and hd_url:
            st.markdown(f"""<div class="fs-overlay" onclick="this.style.display='none'">
                <img src="{hd_url}"></div>""", unsafe_allow_html=True)
            if st.button("✖ Close Full Screen", key="close_fs"):
                st.session_state.fullscreen = False
                st.rerun()

    def _render_video_viewer(self, current: Dict, idx: int, files: List[Dict]):
        """Render video player with prev/next navigation."""
        fp = Path(current['path'])

        nav_cols = st.columns([1, 16, 1])

        with nav_cols[0]:
            st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;min-height:200px;'>",
                        unsafe_allow_html=True)
            if idx > 0:
                if st.button("◀", key="vprev_btn"):
                    st.session_state.selected_file_index = idx - 1
                    st.rerun()
            else:
                st.markdown("<div style='opacity:.2;font-size:28px;text-align:center;'>◀</div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with nav_cols[1]:
            st.markdown("### 🎬 Video Player")
            if fp.exists() and fp.stat().st_size < Config.MAX_VIDEO_SIZE:
                with open(fp, 'rb') as f:
                    vdata = f.read()
                mt, _ = mimetypes.guess_type(str(fp))
                st.video(vdata, format=mt or "video/mp4")
            else:
                st.warning("Video too large or not found.")

            # Video info
            if st.session_state.get('show_info', True):
                with st.expander("📊 Video Info", expanded=True):
                    info = {'filename': current['name'], 'size_mb': fp.stat().st_size / (1024*1024) if fp.exists() else 0}
                    if VIDEO_SUPPORT:
                        try:
                            cap = cv2.VideoCapture(str(fp))
                            info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            info['fps'] = cap.get(cv2.CAP_PROP_FPS)
                            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            info['duration'] = fc / info['fps'] if info['fps'] > 0 else 0
                            cap.release()
                        except Exception:
                            pass
                    for k, v in info.items():
                        if k == 'duration' and v:
                            m, s = int(v // 60), int(v % 60)
                            st.markdown(f"**Duration:** {m:02d}:{s:02d}")
                        elif k == 'size_mb':
                            st.markdown(f"**Size:** {v:.1f} MB")
                        else:
                            st.markdown(f"**{k.title()}:** {v}")

        with nav_cols[2]:
            st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:100%;min-height:200px;'>",
                        unsafe_allow_html=True)
            if idx < len(files) - 1:
                if st.button("▶", key="vnext_btn"):
                    st.session_state.selected_file_index = idx + 1
                    st.rerun()
            else:
                st.markdown("<div style='opacity:.2;font-size:28px;text-align:center;'>▶</div>",
                            unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    def _render_info_panel(self, current: Dict, fp: Path):
        """Render image information panel."""
        info = MediaProcessor.get_image_info(fp)

        with st.expander("📊 Image Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Filename:** {info['filename']}")
                st.markdown(f"**Dimensions:** {info['width']} × {info['height']} px")
                st.markdown(f"**Format:** {info['format']}")
                st.markdown(f"**File Size:** {info['size_bytes'] / (1024*1024):.2f} MB")
                if info['date']:
                    st.markdown(f"**Modified:** {info['date']}")
            with col2:
                # Calculate megapixels
                mp = (info['width'] * info['height']) / 1_000_000
                st.markdown(f"**Megapixels:** {mp:.1f} MP")
                # Aspect ratio
                if info['width'] and info['height']:
                    from math import gcd
                    g = gcd(info['width'], info['height'])
                    st.markdown(f"**Aspect Ratio:** {info['width']//g}:{info['height']//g}")
                # DPI estimate (from EXIF if available)
                if info.get('exif'):
                    st.markdown("**EXIF:**")
                    for k, v in list(info['exif'].items())[:8]:
                        st.caption(f"{k}: {v[:50]}")

    def _render_thumb_strip(self, files: List[Dict], current_idx: int):
        """Render clickable thumbnail strip at the bottom."""
        if not files:
            return

        st.markdown("#### 🖼️ Navigate")
        st.caption(f"Click any thumbnail below, or use ◀ ▶ buttons above")

        # Build HTML thumbnail strip
        html = '<div class="thumb-strip">'

        for i, f in enumerate(files):
            fp = Path(f['path'])
            active = " active" if i == current_idx else ""

            if f['type'] == 'image':
                thumb_url = MediaProcessor.get_thumb_data_url(fp, Config.THUMB_STRIP_SIZE)
            else:
                # Try video thumbnail
                vtp = Config.VIDEO_THUMBNAIL_DIR / f"{fp.stem}_vthumb.jpg"
                if vtp.exists():
                    thumb_url = MediaProcessor.get_data_url(vtp)
                else:
                    thumb_url = ""

            if thumb_url:
                overlay = ""
                if f['type'] == 'video':
                    overlay = ('<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
                               'font-size:20px;color:#fff;opacity:.8;text-shadow:1px 1px 3px rgba(0,0,0,.6);">▶</div>')
                html += f'''<div class="thumb-item{active}" title="{f['stem']}">
                    <img src="{thumb_url}" alt="{f['stem']}">{overlay}</div>'''
            else:
                icon = "🎬" if f['type'] == 'video' else "🖼️"
                html += f'''<div class="thumb-item{active}" title="{f['stem']}">
                    <div style="width:90px;height:68px;background:#2a2a3e;display:flex;
                    align-items:center;justify-content:center;font-size:22px;">{icon}</div></div>'''

        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        # Hidden clickable buttons for each thumbnail
        btn_cols = st.columns(min(len(files), 20))
        for i in range(min(len(files), 20)):
            with btn_cols[i]:
                if st.button(f"{i+1}", key=f"ts_{i}",
                             help=files[i]['stem'][:30]):
                    st.session_state.selected_file_index = i
                    st.rerun()

        # If more than 20 files, show a page navigator
        if len(files) > 20:
            page = st.number_input("Page", min_value=1, max_value=max(1, math.ceil(len(files) / 20)),
                                   value=1, key="thumb_page")
            offset = (page - 1) * 20
            btn_cols2 = st.columns(min(len(files) - offset, 20))
            for i in range(min(len(files) - offset, 20)):
                with btn_cols2[i]:
                    ri = offset + i
                    if st.button(f"{ri+1}", key=f"ts2_{ri}",
                                 help=files[ri]['stem'][:30]):
                        st.session_state.selected_file_index = ri
                        st.rerun()

    def _render_empty_state(self):
        """Render when no folder/file is selected."""
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
            <div style="font-size:80px;margin-bottom:20px;">🖼️</div>
            <h2>Welcome to MemoryVault Pro+</h2>
            <p style="color:#888;font-size:16px;max-width:500px;margin:0 auto;">
            Select a folder from the sidebar to browse your photos and videos.<br>
            Use ◀ ▶ buttons to navigate within a directory.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick stats
        st.divider()
        c1, c2, c3 = st.columns(3)
        total_imgs = sum(info['image_count'] for info in self.tree.values())
        total_vids = sum(info['video_count'] for info in self.tree.values())
        with c1: st.metric("Folders", len(self.tree))
        with c2: st.metric("Images", total_imgs)
        with c3: st.metric("Videos", total_vids)

    # ── MAIN ──────────────────────────────────────────────────────────
    def render_main(self):
        self.render_sidebar()
        self.render_viewer()

        st.divider()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} v{Config.VERSION}")


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    if not check_password():
        return
    try:
        app = PhotoAlbumApp()
        if not VIDEO_SUPPORT:
            st.sidebar.caption("⚠️ pip install opencv-python moviepy")
        app.render_main()
    except Exception as e:
        st.error(f"Error: {e}")
        with st.expander("Details"):
            import traceback
            traceback.print_exc()
        if st.button("Retry"):
            st.rerun()


if __name__ == "__main__":
    main()
