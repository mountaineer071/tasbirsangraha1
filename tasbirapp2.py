```python
"""
COMPREHENSIVE WEB PHOTO & VIDEO ALBUM APPLICATION
Version: 7.0.0 - Ultimate Edition
Features: Cipher Password, Directory Tree Sidebar, HD Prev/Next Viewer, 
          Luxury Frames, SQLite DB, Comments, Ratings, Favorites, Search, 
          Dashboard, Settings, Fullscreen, Thumbnail Strip
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps, ExifTags, ImageDraw
import base64
import json
import datetime
import uuid
import sqlite3
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

try:
    import cv2
    from moviepy.editor import VideoFileClip
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False

# ============================================================================
# CIPHER-BASED PASSWORD AUTHENTICATION
# ============================================================================
# The numeric password 19870505 is mapped to alphabet using a=1, i=9, h=8, etc.
# When the user types 19870505, it is converted to "aihgjeje" and unlocks.
# If the user types alphabet characters, it is rejected, so the alphabet code 
# never hints at the date of birth.
_CIPHER_MAP = {'1':'a', '2':'b', '3':'c', '4':'d', '5':'e', '6':'f', '7':'g', '8':'h', '9':'i', '0':'j'}
_ACCESS_CIPHER = "aihgjeje"

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
                                 placeholder="Enter numeric access code", label_visibility="collapsed")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔓 Unlock", use_container_width=True, type="primary"):
                # ONLY numeric input is accepted. We convert digits to alphabet internally.
                if password.strip().isdigit():
                    converted = "".join(_CIPHER_MAP.get(c, c) for c in password.strip())
                    if converted == _ACCESS_CIPHER:
                        st.session_state.authenticated = True
                        st.success("✅ Access granted!")
                        time.sleep(0.4)
                        st.rerun()
                # Reject anything else (including typing the alphabet directly)
                st.error("❌ Invalid key. Numeric code required.")
                time.sleep(0.3)
        with col_b:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.authenticated = False
                st.rerun()
        with st.expander("🔑 Hint"):
            st.info("💡 The access key is a **numeric** code.")
            st.caption("Only digits 0‑9 are accepted. Letters will not work.")
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
    DB_DIR = BASE_DIR / "database"
    EXPORT_DIR = BASE_DIR / "exports"
    DB_FILE = DB_DIR / "album.db"
    THUMBNAIL_SIZE = (300, 300)
    HD_SIZE = (1920, 1080)
    MAX_VIDEO_SIZE = 100 * 1024 * 1024
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | set(SUPPORTED_VIDEO_FORMATS)
    CACHE_TTL = 3600
    FRAME_STYLES = ["Elegant Gold", "Polaroid", "Modern Shadow", "Dark Museum", "Vintage", "Gallery White"]
    DEFAULT_FRAME = "Elegant Gold"
    THUMB_STRIP_SIZE = (120, 90)

    @classmethod
    def init_directories(cls):
        for d in [cls.DATA_DIR, cls.THUMBNAIL_DIR, cls.VIDEO_THUMBNAIL_DIR, cls.DB_DIR, cls.EXPORT_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        if not any(cls.DATA_DIR.iterdir()):
            cls.create_samples()

    @classmethod
    def create_samples(cls):
        for name in ["john-smith", "sarah-johnson", "michael-brown"]:
            pd_dir = cls.DATA_DIR / name
            pd_dir.mkdir(exist_ok=True)
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


# ============================================================================
# LUXURY FRAME RENDERER
# ============================================================================
class FrameRenderer:
    @staticmethod
    def wrap_detail(src: str, style: str = "Elegant Gold") -> str:
        s = {
            "Elegant Gold": (
                'background:linear-gradient(135deg,#b8860b,#daa520,#ffd700,#daa520,#b8860b);'
                'padding:18px;border-radius:12px;box-shadow:0 24px 80px rgba(0,0,0,.5),'
                'inset 0 3px 0 rgba(255,255,255,.5),inset 0 -3px 0 rgba(0,0,0,.2);border:2px solid #ffd700;',
                'background:#fffff5;padding:24px;border-radius:8px;box-shadow:inset 0 0 30px rgba(0,0,0,.08);'),
            "Polaroid": ('background:#fff;padding:24px 24px 80px 24px;box-shadow:0 16px 50px rgba(0,0,0,.25);border-radius:4px;', ''),
            "Modern Shadow": ('background:transparent;padding:0;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,.3);overflow:hidden;', ''),
            "Dark Museum": ('background:linear-gradient(160deg,#0d0d1a,#1a1a30,#0d0d1a);padding:32px;border-radius:20px;box-shadow:0 30px 90px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.05);',
                            'background:#fffff8;padding:20px;border-radius:8px;box-shadow:inset 0 0 25px rgba(0,0,0,.05);'),
            "Vintage": ('background:linear-gradient(135deg,#d4b896,#e8d5b7,#c9a96e);padding:20px;border-radius:10px;box-shadow:0 16px 50px rgba(0,0,0,.3),inset 0 0 60px rgba(139,109,63,.15);border:3px solid #a08050;',
                        'background:#faf5ee;padding:16px;border-radius:6px;'),
            "Gallery White": ('background:#fff;padding:30px;border-radius:6px;box-shadow:0 8px 30px rgba(0,0,0,.12);border:1px solid #e8e8e8;', ''),
        }
        outer, inner = s.get(style, s["Elegant Gold"])
        return f'''<div style="{outer}"><div style="{inner}">
            <img src="{src}" style="width:100%;max-height:78vh;object-fit:contain;display:block;margin:0 auto;border-radius:6px;">
        </div></div>'''

    @staticmethod
    def wrap_thumb_strip_item(src: str, active: bool = False, is_video: bool = False) -> str:
        active_class = "active" if active else ""
        play_mark = '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:20px;color:#fff;opacity:.9;">▶</div>' if is_video else ''
        return f'''<div class="thumb-item {active_class}" style="cursor:pointer;">
            <img src="{src}" style="width:100%;height:100%;object-fit:cover;">{play_mark}</div>'''

    @staticmethod
    def inject_css():
        st.markdown("""
        <style>
        .stApp{scroll-behavior:smooth;}
        .dir-folder{padding:8px 12px;border-radius:8px;cursor:pointer;display:flex;
                     align-items:center;gap:8px;font-size:14px;font-weight:500;margin-bottom:2px;}
        .dir-folder:hover{background:rgba(102,126,234,.15);}
        .dir-folder.active{background:rgba(102,126,234,.25);color:#fff;font-weight:700;}
        .dir-file{padding:5px 12px 5px 32px;border-radius:6px;cursor:pointer;display:flex;
                   align-items:center;gap:6px;font-size:12px;color:#a0a0a0;margin-bottom:1px;}
        .dir-file:hover{background:rgba(102,126,234,.1);color:#e0e0e0;}
        .dir-file.active{background:rgba(102,126,234,.2);color:#fff;font-weight:600;}
        .nav-container{display:flex;align-items:center;justify-content:center;height:100%;min-height:60vh;}
        .nav-btn{background:rgba(102,126,234,.9);color:#fff;border:none;border-radius:50%;width:56px;height:56px;
                 font-size:28px;cursor:pointer;transition:all 0.2s;box-shadow:0 4px 12px rgba(0,0,0,.2);
                 display:flex;align-items:center;justify-content:center;}
        .nav-btn:hover{background:#764ba2;transform:scale(1.1);}
        .nav-btn.disabled{opacity:0.3;cursor:not-allowed;pointer-events:none;}
        .thumb-strip{display:flex;gap:8px;overflow-x:auto;padding:8px 4px;scroll-behavior:smooth;}
        .thumb-strip::-webkit-scrollbar{height:6px;}
        .thumb-strip::-webkit-scrollbar-thumb{background:#667eea;border-radius:3px;}
        .thumb-item{min-width:90px;height:68px;border-radius:6px;overflow:hidden;flex-shrink:0;
                    cursor:pointer;transition:transform .2s,opacity .2s;opacity:.5;position:relative;}
        .thumb-item:hover{opacity:.85;transform:scale(1.05);}
        .thumb-item.active{opacity:1;transform:scale(1.08);border-bottom:3px solid #667eea;}
        .thumb-item img{width:100%;height:100%;object-fit:cover;display:block;}
        .fs-overlay{position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,.96);
                    z-index:9999;display:flex;align-items:center;justify-content:center;cursor:zoom-out;}
        .fs-overlay img{max-width:96vw;max-height:96vh;object-fit:contain;border-radius:6px;box-shadow:0 0 80px rgba(0,0,0,.7);}
        .close-fs-btn{position:fixed;top:20px;right:30px;background:rgba(0,0,0,.7);color:#fff;border:none;
                      border-radius:30px;padding:10px 20px;font-size:16px;cursor:pointer;z-index:10000;font-weight:bold;}
        </style>
        """, unsafe_allow_html=True)


# ============================================================================
# MEDIA & DATA MODELS
# ============================================================================
@dataclass
class MediaMetadata:
    media_id: str; filename: str; filepath: str; file_size: int; media_type: str
    dimensions: Tuple[int, int]; format: str; duration: Optional[float]
    created_date: datetime.datetime; checksum: str

@dataclass
class AlbumEntry:
    entry_id: str; media_id: str; person_id: str; caption: str; description: str
    location: str; date_taken: Optional[str]; tags: str; privacy_level: str
    created_by: str; created_at: datetime.datetime

@dataclass
class Comment:
    comment_id: str; entry_id: str; user_id: str; username: str; content: str; created_at: datetime.datetime

@dataclass
class Rating:
    rating_id: str; entry_id: str; user_id: str; rating_value: int; created_at: datetime.datetime


# ============================================================================
# DATABASE MANAGER
# ============================================================================
class DatabaseManager:
    def __init__(self):
        self.db_path = Config.DB_FILE
        self._init_db()

    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try: yield conn
        finally: conn.close()

    def _init_db(self):
        Config.DB_DIR.mkdir(parents=True, exist_ok=True)
        with self.get_conn() as conn:
            conn.executescript('''
            CREATE TABLE IF NOT EXISTS media (media_id TEXT PRIMARY KEY, filename TEXT, filepath TEXT UNIQUE, file_size INTEGER, media_type TEXT, width INTEGER, height INTEGER, duration REAL, created_date TIMESTAMP, thumbnail_path TEXT, checksum TEXT);
            CREATE TABLE IF NOT EXISTS people (person_id TEXT PRIMARY KEY, folder_name TEXT UNIQUE, display_name TEXT);
            CREATE TABLE IF NOT EXISTS album_entries (entry_id TEXT PRIMARY KEY, media_id TEXT, person_id TEXT, caption TEXT, description TEXT, location TEXT, date_taken TIMESTAMP, tags TEXT, privacy_level TEXT, created_by TEXT, created_at TIMESTAMP);
            CREATE TABLE IF NOT EXISTS comments (comment_id TEXT PRIMARY KEY, entry_id TEXT, user_id TEXT, username TEXT, content TEXT, created_at TIMESTAMP);
            CREATE TABLE IF NOT EXISTS ratings (rating_id TEXT PRIMARY KEY, entry_id TEXT, user_id TEXT, rating_value INTEGER, created_at TIMESTAMP);
            CREATE TABLE IF NOT EXISTS user_favorites (user_id TEXT, entry_id TEXT, created_at TIMESTAMP, PRIMARY KEY (user_id, entry_id));
            ''')
            conn.commit()

    def upsert_media(self, m: MediaMetadata, tp: str):
        with self.get_conn() as conn:
            conn.execute('''INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                         (m.media_id, m.filename, m.filepath, m.file_size, m.media_type, m.dimensions[0], m.dimensions[1], m.duration, m.created_date, tp, m.checksum))
            conn.commit()

    def upsert_person(self, pid, folder, display):
        with self.get_conn() as conn:
            conn.execute('''INSERT OR REPLACE INTO people VALUES (?,?,?)''', (pid, folder, display))
            conn.commit()

    def upsert_entry(self, e: AlbumEntry):
        with self.get_conn() as conn:
            conn.execute('''INSERT OR REPLACE INTO album_entries VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                         (e.entry_id, e.media_id, e.person_id, e.caption, e.description, e.location, e.date_taken, e.tags, e.privacy_level, e.created_by, e.created_at))
            conn.commit()

    def get_media_by_path(self, fp: str) -> Optional[Dict]:
        with self.get_conn() as conn:
            r = conn.execute('SELECT * FROM media WHERE filepath=?', (fp,)).fetchone()
            return dict(r) if r else None

    def get_entry_by_media_id(self, mid: str) -> Optional[Dict]:
        with self.get_conn() as conn:
            r = conn.execute('SELECT * FROM album_entries WHERE media_id=?', (mid,)).fetchone()
            return dict(r) if r else None

    def get_entry_comments(self, eid: str) -> List[Dict]:
        with self.get_conn() as conn:
            return [dict(r) for r in conn.execute('SELECT * FROM comments WHERE entry_id=? ORDER BY created_at DESC', (eid,)).fetchall()]

    def add_comment(self, c: Comment):
        with self.get_conn() as conn:
            conn.execute('INSERT INTO comments VALUES (?,?,?,?,?,?)', (c.comment_id, c.entry_id, c.user_id, c.username, c.content, c.created_at))
            conn.commit()

    def get_entry_ratings(self, eid: str) -> Tuple[float, int]:
        with self.get_conn() as conn:
            r = conn.execute('SELECT AVG(rating_value), COUNT(*) FROM ratings WHERE entry_id=?', (eid,)).fetchone()
            return (r[0] or 0.0, r[1] or 0)

    def add_rating(self, r: Rating):
        with self.get_conn() as conn:
            conn.execute('INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?)', (r.rating_id, r.entry_id, r.user_id, r.rating_value, r.created_at))
            conn.commit()

    def toggle_favorite(self, uid, eid):
        with self.get_conn() as conn:
            if conn.execute('SELECT 1 FROM user_favorites WHERE user_id=? AND entry_id=?', (uid, eid)).fetchone():
                conn.execute('DELETE FROM user_favorites WHERE user_id=? AND entry_id=?', (uid, eid))
            else:
                conn.execute('INSERT INTO user_favorites VALUES (?,?,?)', (uid, eid, datetime.datetime.now()))
            conn.commit()

    def is_favorite(self, uid, eid) -> bool:
        with self.get_conn() as conn:
            return conn.execute('SELECT 1 FROM user_favorites WHERE user_id=? AND entry_id=?', (uid, eid)).fetchone() is not None

    def get_favorites(self, uid) -> List[Dict]:
        with self.get_conn() as conn:
            return [dict(r) for r in conn.execute('''SELECT ae.*, m.filename, m.media_type, m.thumbnail_path FROM user_favorites uf 
                         JOIN album_entries ae ON uf.entry_id=ae.entry_id 
                         JOIN media m ON ae.media_id=m.media_id WHERE uf.user_id=?''', (uid,)).fetchall()]

    def search_entries(self, q: str) -> List[Dict]:
        with self.get_conn() as conn:
            sp = f'%{q}%'
            return [dict(r) for r in conn.execute('''SELECT ae.*, p.display_name, m.filename, m.media_type, m.thumbnail_path 
                         FROM album_entries ae JOIN people p ON ae.person_id=p.person_id 
                         JOIN media m ON ae.media_id=m.media_id 
                         WHERE ae.caption LIKE ? OR ae.tags LIKE ?''', (sp, sp)).fetchall()]


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
                    bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA','LA') else None)
                    img = bg
                img.thumbnail(Config.HD_SIZE, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95, optimize=True)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except: return ""

    @staticmethod
    def get_thumb_strip_url(fp: Path, is_video: bool = False) -> str:
        try:
            if is_video:
                vt = Config.VIDEO_THUMBNAIL_DIR / f"{fp.stem}_thumb.jpg"
                if vt.exists(): return MediaProcessor.get_data_url(vt)
                return ""
            if not fp.exists(): return ""
            with Image.open(fp) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA','LA') else None)
                    img = bg
                img.thumbnail(Config.THUMB_STRIP_SIZE, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=80)
                return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        except: return ""

    @staticmethod
    def get_data_url(fp: Path) -> str:
        try:
            if not fp.exists(): return ""
            mt, _ = mimetypes.guess_type(str(fp))
            if not mt: mt = 'image/jpeg'
            with open(fp, "rb") as f:
                return f"data:{mt};base64,{base64.b64encode(f.read()).decode('utf-8')}"
        except: return ""

    @staticmethod
    def create_thumbnail(fp: Path, is_video=False) -> Optional[Path]:
        td = Config.VIDEO_THUMBNAIL_DIR if is_video else Config.THUMBNAIL_DIR
        os.makedirs(td, exist_ok=True)
        tp = td / f"{fp.stem}_thumb.jpg"
        if tp.exists(): return tp
        try:
            if is_video and VIDEO_SUPPORT:
                cap = cv2.VideoCapture(str(fp))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, frame = cap.read(); cap.release()
                if ret and frame is not None:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                    img.save(tp, 'JPEG', quality=85)
                    return tp
            elif not is_video:
                with Image.open(fp) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        bg = Image.new('RGB', img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA','LA') else None)
                        img = bg
                    img.thumbnail(Config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                    img.save(tp, 'JPEG', quality=85)
                    return tp
        except: pass
        return None


# ============================================================================
# DIRECTORY SCANNER & ALBUM MANAGER
# ============================================================================
class AlbumManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.mp = MediaProcessor()
        self._init_ss()

    def _init_ss(self):
        if 'am_init' not in st.session_state:
            st.session_state.update({
                'am_init': True, 'user_id': str(uuid.uuid4()), 'username': 'Guest',
                'selected_folder': None, 'selected_file_index': 0,
                'frame_style': Config.DEFAULT_FRAME, 'fullscreen': False,
                'dir_expanded': {}, 'current_page': 'viewer'
            })

    @property
    def fs(self): return st.session_state.get('frame_style', Config.DEFAULT_FRAME)
    @property
    def uid(self): return st.session_state.get('user_id', '')

    def scan_and_sync(self):
        """Scans directory, creates thumbnails, and syncs with DB"""
        dd = Config.DATA_DIR
        skip = {'thumbnails', 'video_thumbnails', 'database', 'metadata', 'exports'}
        pdirs = [d for d in dd.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name not in skip]
        if not pdirs: Config.create_samples(); pdirs = [d for d in dd.iterdir() if d.is_dir() and d.name not in skip]
        
        pb = st.progress(0, text="Scanning directories...")
        total = sum(1 for pd in pdirs for f in pd.iterdir() if f.is_file() and f.suffix.lower() in Config.ALLOWED_EXTENSIONS)
        proc = 0
        
        for pdir in pdirs:
            dn = ' '.join(p.capitalize() for p in pdir.name.replace('-', ' ').replace('_', ' ').split())
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, pdir.name))
            self.db.upsert_person(pid, pdir.name, dn)
            
            for mf in pdir.iterdir():
                if not mf.is_file() or mf.suffix.lower() not in Config.ALLOWED_EXTENSIONS: continue
                try:
                    proc += 1; pb.progress(proc / max(total,1), text=f"Processing {mf.name}...")
                    rel_path = str(mf.relative_to(Config.DATA_DIR))
                    existing = self.db.get_media_by_path(rel_path)
                    if existing: continue
                    
                    is_vid = mf.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS
                    tp = self.mp.create_thumbnail(mf, is_vid)
                    
                    # Extract basic metadata
                    w, h, dur = 0, 0, 0.0
                    if is_vid and VIDEO_SUPPORT:
                        try:
                            c = VideoFileClip(str(mf)); w, h = c.size; dur = c.duration; c.close()
                        except: pass
                    elif not is_vid:
                        try:
                            with Image.open(mf) as img: w, h = img.size
                        except: pass
                    
                    mid = str(uuid.uuid4())
                    meta = MediaMetadata(mid, mf.name, rel_path, mf.stat().st_size, 
                                         'video' if is_vid else 'image', (w,h), mf.suffix[1:].upper(), 
                                         dur, datetime.datetime.fromtimestamp(mf.stat().st_ctime), "")
                    self.db.upsert_media(meta, str(tp) if tp else "")
                    
                    eid = str(uuid.uuid4())
                    entry = AlbumEntry(eid, mid, pid, mf.stem.replace('_',' ').title(), 
                                       f"Media of {dn}", "", "", dn.lower().replace(' ','-'), 
                                       'public', 'system', datetime.datetime.now())
                    self.db.upsert_entry(entry)
                except Exception: pass
        pb.empty()

    def get_tree(self) -> Dict[str, Dict]:
        tree = {}
        dd = Config.DATA_DIR
        skip = {'thumbnails', 'video_thumbnails', 'database', 'metadata', 'exports'}
        for folder in sorted(dd.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.') or folder.name in skip: continue
            files = []
            for f in sorted(folder.iterdir()):
                if not f.is_file() or f.suffix.lower() not in Config.ALLOWED_EXTENSIONS: continue
                media_type = 'video' if f.suffix.lower() in Config.SUPPORTED_VIDEO_FORMATS else 'image'
                rel_path = str(f.relative_to(Config.DATA_DIR))
                db_media = self.db.get_media_by_path(rel_path)
                entry = self.db.get_entry_by_media_id(db_media['media_id']) if db_media else None
                files.append({
                    'path': str(f), 'name': f.name, 'stem': f.stem, 'suffix': f.suffix,
                    'type': media_type, 'size': f.stat().st_size, 
                    'entry_id': entry['entry_id'] if entry else None,
                    'media_id': db_media['media_id'] if db_media else None,
                })
            if files:
                dn = ' '.join(p.capitalize() for p in folder.name.replace('-', ' ').replace('_', ' ').split())
                tree[folder.name] = {'display_name': dn, 'files': files, 
                                     'image_count': sum(1 for f in files if f['type']=='image'),
                                     'video_count': sum(1 for f in files if f['type']=='video')}
        return tree


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
            return self.mgr.get_tree().get(folder, {}).get('files', [])
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

            pages = [("📂 Viewer", "viewer"), ("📊 Dashboard", "dashboard"), 
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
                with st.spinner("Syncing directory to database..."):
                    self.mgr.scan_and_sync()
                st.success("Done!"); st.rerun()

            tree = self.mgr.get_tree()
            for folder_name, info in tree.items():
                is_active = st.session_state.get('selected_folder') == folder_name
                is_expanded = st.session_state.get('dir_expanded', {}).get(folder_name, is_active)
                icon = "📂" if is_expanded else "📁"

                with st.expander(f"{icon} {info['display_name']} (📸{info['image_count']} 🎬{info['video_count']})", expanded=is_expanded):
                    if st.button(f"Open Folder", key=f"fo_{folder_name}", use_container_width=True):
                        st.session_state.selected_folder = folder_name
                        st.session_state.selected_file_index = 0
                        st.session_state.current_page = 'viewer'
                        st.rerun()
                    
                    for fi, f in enumerate(info['files']):
                        ficon = "🎬" if f['type'] == 'video' else "🖼️"
                        act = " ←" if (is_active and st.session_state.get('selected_file_index',0) == fi) else ""
                        if st.button(f"{ficon} {f['stem'][:25]}{act}", key=f"fl_{folder_name}_{fi}"):
                            st.session_state.selected_folder = folder_name
                            st.session_state.selected_file_index = fi
                            st.session_state.current_page = 'viewer'
                            st.rerun()

    # ── HD VIEWER ─────────────────────────────────────────────────────
    def render_viewer(self):
        files = self._get_files()
        current = self._get_current()
        if not files or not current:
            st.markdown("<div style='text-align:center;padding:80px;'><h2>🖼️ Select a folder from the sidebar</h2></div>", unsafe_allow_html=True)
            return

        idx = st.session_state.get('selected_file_index', 0)
        folder = st.session_state.get('selected_folder')
        display_name = self.mgr.get_tree().get(folder, {}).get('display_name', folder)

        st.markdown(f"### 📂 {display_name}  •  {current['name']}")
        st.markdown(f"<div style='text-align:right;color:#888;'>{idx+1} / {len(files)}</div>", unsafe_allow_html=True)
        st.divider()

        col_prev, col_mid, col_next = st.columns([1, 8, 1])

        with col_prev:
            st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
            if idx > 0:
                if st.button("◀", key="prev_btn"): st.session_state.selected_file_index = idx - 1; st.rerun()
            else: st.markdown("<div class='nav-btn disabled'>◀</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_mid:
            if current['type'] == 'image':
                fp = Path(current['path'])
                hd_url = MediaProcessor.get_hd_data_url(fp)
                if hd_url: st.markdown(FrameRenderer.wrap_detail(hd_url, self.mgr.fs), unsafe_allow_html=True)
                else: st.error("Could not load image")

                act_cols = st.columns([1,1,1,2])
                with act_cols[0]:
                    if st.button("🔍 Full Screen", use_container_width=True):
                        st.session_state.fullscreen = hd_url; st.rerun()
                with act_cols[1]:
                    if fp.exists():
                        with open(fp, 'rb') as f:
                            st.download_button("💾 Download", data=f.read(), file_name=fp.name, use_container_width=True)
                with act_cols[2]:
                    if current.get('entry_id'):
                        is_fav = self.mgr.db.is_favorite(self.mgr.uid, current['entry_id'])
                        if st.button("⭐ Unfav" if is_fav else "☆ Favorite", use_container_width=True):
                            self.mgr.db.toggle_favorite(self.mgr.uid, current['entry_id']); st.rerun()
            else:
                fp = Path(current['path'])
                if fp.exists() and fp.stat().st_size < Config.MAX_VIDEO_SIZE:
                    with open(fp, 'rb') as f: st.video(f.read())
                else: st.warning("Video unavailable")

            # Comments & Ratings
            if current.get('entry_id'):
                self._render_comments_ratings(current['entry_id'])

        with col_next:
            st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
            if idx < len(files) - 1:
                if st.button("▶", key="next_btn"): st.session_state.selected_file_index = idx + 1; st.rerun()
            else: st.markdown("<div class='nav-btn disabled'>▶</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Thumbnail strip
        if len(files) > 1:
            st.divider()
            st.markdown("#### 🖼️ Navigate")
            strip_cols = st.columns(min(len(files), 12))
            for i in range(min(len(files), 12)):
                f = files[i]
                fp = Path(f['path'])
                thumb_url = MediaProcessor.get_thumb_strip_url(fp, f['type']=='video')
                if not thumb_url: thumb_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='90' height='68'%3E%3Crect width='90' height='68' fill='%23333'/%3E%3Ctext x='45' y='38' fill='%23fff' font-size='16' text-anchor='middle'%3E🖼️%3C/text%3E%3C/svg%3E"
                with strip_cols[i]:
                    st.markdown(FrameRenderer.wrap_thumb_strip_item(thumb_url, i == idx, f['type']=='video'), unsafe_allow_html=True)
                    if st.button(f"{i+1}", key=f"ts_{i}", use_container_width=True):
                        st.session_state.selected_file_index = i; st.rerun()

        # Fullscreen overlay
        if st.session_state.get('fullscreen'):
            st.markdown(f"""<div class="fs-overlay" id="fsOverlay">
                <img src="{st.session_state.fullscreen}" alt="Fullscreen">
                <button class="close-fs-btn" id="closeFsBtn">✖ Close</button>
            </div>
            <script>
            document.getElementById('fsOverlay').addEventListener('click', function(e) {{
                if(e.target === this || e.target.id === 'closeFsBtn') {{ this.style.display = 'none'; }}
            }});
            </script>""", unsafe_allow_html=True)
            if st.button("Exit Fullscreen", key="exit_fs"):
                st.session_state.fullscreen = False; st.rerun()

    def _render_comments_ratings(self, eid):
        with st.expander("💬 Comments & ⭐ Ratings", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                avg, cnt = self.mgr.db.get_entry_ratings(eid)
                st.markdown(f"**Rating:** {'⭐'*int(round(avg))} ({avg:.1f}/5 from {cnt} votes)")
                rc = st.columns(5)
                for i in range(1, 6):
                    with rc[i-1]:
                        if st.button(f"{i}⭐", key=f"r_{i}_{eid}"):
                            self.mgr.db.add_rating(Rating(str(uuid.uuid4()), eid, self.mgr.uid, i, datetime.datetime.now()))
                            st.rerun()
            with c2:
                comments = self.mgr.db.get_entry_comments(eid)
                st.markdown(f"**{len(comments)} Comments**")
                with st.form(f"cmt_{eid}"):
                    ct = st.text_input("Add comment...", key=f"ci_{eid}")
                    if st.form_submit_button("Post") and ct.strip():
                        self.mgr.db.add_comment(Comment(str(uuid.uuid4()), eid, self.mgr.uid, st.session_state.username, ct.strip(), datetime.datetime.now()))
                        st.rerun()
                for c in comments[:5]:
                    st.markdown(f"**{c['username']}**: {c['content'][:80]}")

    # ── DASHBOARD ─────────────────────────────────────────────────────
    def render_dashboard(self):
        st.title("📊 Dashboard")
        tree = self.mgr.get_tree()
        total_i = sum(i['image_count'] for i in tree.values())
        total_v = sum(i['video_count'] for i in tree.values())
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Folders", len(tree))
        with c2: st.metric("Images", total_i)
        with c3: st.metric("Videos", total_v)

    # ── FAVORITES ─────────────────────────────────────────────────────
    def render_favorites(self):
        st.title("⭐ Favorites")
        favs = self.mgr.db.get_favorites(self.mgr.uid)
        if not favs: st.info("No favorites yet."); return
        cols = st.columns(4)
        for i, f in enumerate(favs):
            with cols[i % 4]:
                with st.container(border=True):
                    if f.get('thumbnail_path') and Path(f['thumbnail_path']).exists():
                        st.image(MediaProcessor.get_data_url(Path(f['thumbnail_path'])), use_column_width=True)
                    st.markdown(f"**{f.get('caption','')}**")
                    if st.button("👁️ View", key=f"fv_{f['entry_id']}", use_container_width=True):
                        # Find in tree and navigate
                        tree = self.mgr.get_tree()
                        for folder, info in tree.items():
                            for fi, file in enumerate(info['files']):
                                if file.get('entry_id') == f['entry_id']:
                                    st.session_state.selected_folder = folder
                                    st.session_state.selected_file_index = fi
                                    st.session_state.current_page = 'viewer'
                                    st.rerun()

    # ── SEARCH ────────────────────────────────────────────────────────
    def render_search(self):
        st.title("🔍 Search")
        sq = st.text_input("Search captions, tags...")
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

    # ── SETTINGS ──────────────────────────────────────────────────────
    def render_settings(self):
        st.title("⚙️ Settings")
        st.subheader("User Profile")
        st.text_input("Username", value=st.session_state.username, key="set_uname")
        if st.button("Update Name"): st.session_state.username = st.session_state.set_uname; st.success("Updated!")
        
        st.subheader("Database")
        if st.button("Rebuild DB"):
            self.mgr.scan_and_sync(); st.success("Synced!")

    # ── MAIN RUNNER ───────────────────────────────────────────────────
    def run(self):
        self.render_sidebar()
        page = st.session_state.get('current_page', 'viewer')
        if page == 'viewer': self.render_viewer()
        elif page == 'dashboard': self.render_dashboard()
        elif page == 'favorites': self.render_favorites()
        elif page == 'search': self.render_search()
        elif page == 'settings': self.render_settings()
        
        st.divider()
        st.caption(f"© {datetime.datetime.now().year} {Config.APP_NAME} v{Config.VERSION}")


def main():
    if not check_password(): return
    app = PhotoAlbumApp()
    app.run()

if __name__ == "__main__":
    main()
```
