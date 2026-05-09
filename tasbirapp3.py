"""
COMPREHENSIVE WEB PHOTO & VIDEO ALBUM APPLICATION
Version: 6.1.0 - Clean Sidebar (Folders Only), Expandable View, Centered Nav
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ExifTags, ImageDraw
import base64
import datetime
import uuid
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
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

# VIDEO SUPPORT (optional)
try:
    import cv2
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False

# ============================================================================
# NUMERIC PASSWORD (8 digits)
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
              margin:40px 0;}
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
# CONFIG
# ============================================================================
class Config:
    APP_NAME = "MemoryVault Pro+"
    VERSION = "6.1.0"
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    THUMBNAIL_DIR = BASE_DIR / "thumbnails"
    VIDEO_THUMBNAIL_DIR = BASE_DIR / "video_thumbnails"
    DB_DIR = BASE_DIR / "database"
    DB_FILE = DB_DIR / "album.db"
    THUMBNAIL_SIZE = (300, 300)
    HD_SIZE = (1920, 1080)
    MAX_VIDEO_SIZE = 100 * 1024 * 1024
    SUPPORTED_VIDEO_FORMATS = ['.mp4','.mov','.avi','.mkv','.webm']
    IMAGE_EXTENSIONS = {'.jpg','.jpeg','.png','.gif','.bmp','.webp'}
    ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | set(SUPPORTED_VIDEO_FORMATS)
    FRAME_STYLES = ["Elegant Gold","Polaroid","Modern Shadow","Dark Museum","Vintage","Gallery White"]
    DEFAULT_FRAME = "Elegant Gold"
    THUMB_STRIP_SIZE = (120, 90)

    @classmethod
    def init_dirs(cls):
        for d in [cls.DATA_DIR, cls.THUMBNAIL_DIR, cls.VIDEO_THUMBNAIL_DIR, cls.DB_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_samples()

    @classmethod
    def create_samples(cls):
        for name in ["john-smith", "sarah-johnson", "michael-brown"]:
            folder = cls.DATA_DIR / name
            folder.mkdir(exist_ok=True)
            for i in range(1,4):
                img = folder / f"photo_{i}.jpg"
                if not img.exists():
                    try:
                        im = Image.new('RGB', (600,400), color=['#667eea','#f56565','#48bb78'][i-1])
                        draw = ImageDraw.Draw(im)
                        draw.text((200,180), f"{name.split('-')[0].title()} {i}", fill='#fff')
                        im.save(img, 'JPEG')
                    except:
                        pass

# ============================================================================
# FRAME RENDERER
# ============================================================================
class FrameRenderer:
    @staticmethod
    def wrap_detail(src: str, style: str, expanded: bool = False) -> str:
        max_h = "85vh" if expanded else "65vh"
        s = {
            "Elegant Gold": ('background:linear-gradient(135deg,#b8860b,#daa520,#ffd700);padding:14px;border-radius:10px;',
                             'background:#fffff5;padding:20px;border-radius:6px;'),
            "Polaroid": ('background:#fff;padding:20px 20px 70px 20px;border-radius:3px;',''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:16px;box-shadow:0 14px 48px rgba(0,0,0,.2);',''),
            "Dark Museum": ('background:linear-gradient(160deg,#0d0d1a,#1a1a30);padding:28px;border-radius:16px;',
                            'background:#fffff8;padding:18px;border-radius:6px;'),
            "Vintage": ('background:linear-gradient(135deg,#d4b896,#e8d5b7);padding:16px;border-radius:6px;border:2px solid #a08050;',
                        'background:#faf5ee;padding:14px;border-radius:4px;'),
            "Gallery White": ('background:#fff;padding:24px;border-radius:4px;border:1px solid #e0e0e0;',''),
        }
        outer, inner = s.get(style, s["Elegant Gold"])
        return f'''<div style="{outer}"><div style="{inner}">
            <img src="{src}" style="width:100%;max-height:{max_h};object-fit:contain;display:block;margin:0 auto;border-radius:4px;">
        </div></div>'''

    @staticmethod
    def wrap_thumb_strip_item(src: str, active: bool = False, is_video: bool = False) -> str:
        active_cls = "active" if active else ""
        play = '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:20px;color:#fff;">▶</div>' if is_video else ''
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
        .nav-btn { background: #667eea; color: white; border: none; border-radius: 50%; width: 56px; height: 56px; font-size: 28px; cursor: pointer; transition: 0.2s; }
        .nav-btn:hover { background: #764ba2; transform: scale(1.05); }
        .nav-btn.disabled { opacity: 0.3; cursor: not-allowed; pointer-events: none; }
        .thumb-strip { display: flex; gap: 6px; overflow-x: auto; padding: 8px 4px; }
        .thumb-item { min-width: 90px; height: 68px; border-radius: 6px; overflow: hidden; cursor: pointer; opacity: 0.5; transition: 0.2s; position: relative; }
        .thumb-item.active { opacity: 1; transform: scale(1.05); box-shadow: 0 0 0 2px #667eea; }
        .thumb-item img { width: 100%; height: 100%; object-fit: cover; }
        .sidebar-folder { padding: 8px 12px; border-radius: 8px; cursor: pointer; margin-bottom: 2px; transition: background 0.1s; }
        .sidebar-folder:hover { background: rgba(102,126,234,0.2); }
        .sidebar-folder.active { background: rgba(102,126,234,0.3); font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

# ============================================================================
# DIRECTORY SCANNER (folders only, no file listing in sidebar)
# ============================================================================
class DirectoryScanner:
    @staticmethod
    def scan() -> Dict[str, Dict]:
        tree = {}
        skip = {'thumbnails','video_thumbnails','database','exports'}
        for folder in sorted(Config.DATA_DIR.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.') or folder.name in skip:
                continue
            files = []
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS:
                    media_type = 'video' if f.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS else 'image'
                    files.append({
                        'path': str(f),
                        'name': f.name,
                        'stem': f.stem,
                        'type': media_type,
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
# MEDIA PROCESSOR
# ============================================================================
class MediaProcessor:
    @staticmethod
    def get_hd_url(path: Path) -> str:
        try:
            with Image.open(path) as img:
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
    def get_thumb_strip_url(path: Path, is_video: bool = False) -> str:
        try:
            if is_video:
                vt = Config.VIDEO_THUMBNAIL_DIR / f"{path.stem}_thumb.jpg"
                if vt.exists():
                    with open(vt, "rb") as f:
                        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
                return ""
            with Image.open(path) as img:
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

# ============================================================================
# MAIN APP
# ============================================================================
class PhotoApp:
    def __init__(self):
        st.set_page_config(page_title=Config.APP_NAME, layout="wide")
        Config.init_dirs()
        self._init_state()
        self.tree = DirectoryScanner.scan()
        self._auto_select_first()

    def _init_state(self):
        if "selected_folder" not in st.session_state:
            st.session_state.selected_folder = None
            st.session_state.selected_index = 0
            st.session_state.frame_style = Config.DEFAULT_FRAME
            st.session_state.expanded_view = False
            st.session_state.dir_expanded = {}

    def _auto_select_first(self):
        if self.tree and st.session_state.selected_folder is None:
            st.session_state.selected_folder = list(self.tree.keys())[0]
            st.session_state.selected_index = 0

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

            # Frame style
            fs = st.selectbox("Frame Style", Config.FRAME_STYLES,
                              index=Config.FRAME_STYLES.index(st.session_state.frame_style))
            if fs != st.session_state.frame_style:
                st.session_state.frame_style = fs

            st.divider()
            st.subheader("Directories")
            if st.button("🔄 Refresh", use_container_width=True):
                self.tree = DirectoryScanner.scan()
                self._auto_select_first()
                st.rerun()

            # Folders only (no file list inside)
            for folder_name, info in self.tree.items():
                active = (st.session_state.selected_folder == folder_name)
                # Show folder name with badge
                col1, col2 = st.columns([4,1])
                with col1:
                    if st.button(f"📁 {info['display_name']}", key=f"folder_{folder_name}",
                                 use_container_width=True):
                        st.session_state.selected_folder = folder_name
                        st.session_state.selected_index = 0
                        st.rerun()
                with col2:
                    st.caption(f"{info['image_count']+info['video_count']}")

            # Stats
            st.divider()
            total_imgs = sum(f['image_count'] for f in self.tree.values())
            total_vids = sum(f['video_count'] for f in self.tree.values())
            st.metric("Total Folders", len(self.tree))
            c1,c2 = st.columns(2)
            c1.metric("Images", total_imgs)
            c2.metric("Videos", total_vids)

            # Expand toggle
            st.divider()
            st.session_state.expanded_view = st.toggle("🔍 Expanded View", value=st.session_state.expanded_view)

    def render_viewer(self):
        FrameRenderer.inject_css()
        files = self._current_files()
        current = self._current_file()
        if not files or not current:
            st.info("No media in this folder. Add some images/videos to the data directory.")
            return

        idx = st.session_state.selected_index
        folder_name = st.session_state.selected_folder
        folder_info = self.tree[folder_name]

        # Header
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f"### {folder_info['display_name']}")
        with col2:
            st.markdown(f"<div style='text-align:right;color:#888;font-size:14px;'>{idx+1} / {len(files)}</div>", unsafe_allow_html=True)
        st.divider()

        # Navigation columns: Prev | Media | Next (vertically centered)
        col_prev, col_mid, col_next = st.columns([1, 8, 1])

        with col_prev:
            st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
            if idx > 0:
                if st.button("◀", key="prev_btn", help="Previous"):
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
                # Info row
                with st.expander("ℹ️ Info", expanded=False):
                    st.write(f"**Name:** {current['name']}")
                    st.write(f"**Size:** {current['size']/(1024*1024):.2f} MB")
            else:  # video
                fp = Path(current['path'])
                if fp.exists() and fp.stat().st_size < Config.MAX_VIDEO_SIZE:
                    with open(fp, 'rb') as f:
                        st.video(f.read())
                else:
                    st.warning("Video too large or missing")

        with col_next:
            st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
            if idx < len(files) - 1:
                if st.button("▶", key="next_btn", help="Next"):
                    st.session_state.selected_index = idx + 1
                    st.rerun()
            else:
                st.markdown("<div class='nav-btn disabled'>▶</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Thumbnail strip (clickable)
        if len(files) > 1:
            st.divider()
            st.markdown("#### Quick navigation")
            # Show first 10 thumbnails
            cols = st.columns(min(len(files), 10))
            for i in range(min(len(files), 10)):
                f = files[i]
                fp = Path(f['path'])
                thumb = MediaProcessor.get_thumb_strip_url(fp, is_video=(f['type']=='video'))
                if not thumb:
                    thumb = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='90' height='68' viewBox='0 0 90 68'%3E%3Crect width='90' height='68' fill='%23333'/%3E%3Ctext x='45' y='38' fill='%23fff' font-size='16' text-anchor='middle'%3E📸%3C/text%3E%3C/svg%3E"
                active = (i == idx)
                with cols[i]:
                    st.markdown(FrameRenderer.wrap_thumb_strip_item(thumb, active, f['type']=='video'),
                                unsafe_allow_html=True)
                    if st.button(str(i+1), key=f"thumb_{i}", use_container_width=True):
                        st.session_state.selected_index = i
                        st.rerun()
            if len(files) > 10:
                st.caption(f"... and {len(files)-10} more")

    def run(self):
        self.render_sidebar()
        self.render_viewer()

# ============================================================================
# MAIN
# ============================================================================
def main():
    if not check_password():
        return
    app = PhotoApp()
    app.run()

if __name__ == "__main__":
    main()
