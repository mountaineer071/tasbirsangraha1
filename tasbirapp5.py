"""
MEMORYVAULT CHAT ROOMS v9.0.0
Multi-Room Chat with Working Photos, Replies & Timer
Uses Streamlit native chat elements + BytesIO image display
"""
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps
import base64
import datetime
import uuid
from typing import Dict, List, Optional
import os
import warnings
import time
import io

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    APP_NAME = "MemoryVault Chat"
    VERSION = "9.0.0"
    MAX_MESSAGE_LENGTH = 2000
    MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB for stability
    SUPPORTED_VIDEO_FORMATS = ['.mp4','.mov','.avi','.mkv','.webm','.wmv','.flv','.m4v']
    IMAGE_EXTENSIONS = {'.jpg','.jpeg','.png','.gif','.bmp','.webp','.tiff'}
    ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | set(SUPPORTED_VIDEO_FORMATS)
    THUMBNAIL_SIZE = (600, 600)
    MAX_MESSAGES = 500
    TIMER_OPTIONS = {
        "1 min": 60,
        "3 min": 180,
        "5 min": 300,
        "10 min": 600,
        "30 min": 1800,
        "1 hour": 3600,
    }
    DEFAULT_ROOMS = [
        "🌟 General",
        "📸 Photo Sharing", 
        "🎉 Events",
        "💬 Random",
        "🔒 Private"
    ]


# ============================================================================
# SESSION STORAGE - MULTI-ROOM
# ============================================================================
class SessionStorage:
    def __init__(self):
        defaults = {
            'session_uploads': {},
            'chat_messages': {},  # room_name -> list of messages
            'guest_users': {},
            'active_sessions': set(),
            'message_counter': 0,
            'photo_timers': {},
            'user_upvotes': set(),
            'deleted_messages': set(),
            'current_room': Config.DEFAULT_ROOMS[0],
            'replying_to': None,
            'replying_to_user': None,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

        # Initialize rooms
        for room in Config.DEFAULT_ROOMS:
            if room not in st.session_state.chat_messages:
                st.session_state.chat_messages[room] = []

    def store_upload(self, file_id: str, file_bytes: bytes, filename: str, mime_type: str):
        st.session_state.session_uploads[file_id] = {
            'bytes': file_bytes,
            'filename': filename,
            'mime_type': mime_type,
            'uploaded_at': datetime.datetime.now().isoformat(),
            'size': len(file_bytes)
        }
        return file_id

    def get_upload(self, file_id: str) -> Optional[Dict]:
        return st.session_state.session_uploads.get(file_id)

    def add_chat_message(self, room: str, user_id: str, username: str, content: str, 
                        media_ids: List[str] = None, reply_to: Optional[str] = None,
                        reply_to_user: Optional[str] = None,
                        timer_duration: int = 300):
        st.session_state.message_counter += 1
        msg_id = f"msg_{st.session_state.message_counter}_{uuid.uuid4().hex[:6]}"

        msg = {
            'message_id': msg_id,
            'room': room,
            'user_id': user_id,
            'username': username,
            'content': content,
            'media_ids': media_ids or [],
            'reply_to': reply_to,
            'reply_to_user': reply_to_user,
            'timestamp': datetime.datetime.now().isoformat(),
            'likes': set(),
            'edited': False,
            'deleted': False
        }

        if room not in st.session_state.chat_messages:
            st.session_state.chat_messages[room] = []
        st.session_state.chat_messages[room].append(msg)

        if media_ids:
            st.session_state.photo_timers[msg_id] = {
                'created_at': time.time(),
                'base_duration': timer_duration,
                'upvote_extension': 0,
                'total_duration': timer_duration,
                'expires_at': time.time() + timer_duration,
                'media_ids': media_ids.copy(),
                'timer_choice': timer_duration
            }
        return msg_id

    def get_room_messages(self, room: str, limit: int = 200) -> List[Dict]:
        if room not in st.session_state.chat_messages:
            return []

        messages = []
        now = time.time()

        for msg in st.session_state.chat_messages[room][-limit:]:
            if msg['message_id'] in st.session_state.deleted_messages:
                continue

            msg = msg.copy()
            timer = st.session_state.photo_timers.get(msg['message_id'])

            if timer and now > timer['expires_at']:
                msg['media_expired'] = True
                msg['expired_media_ids'] = msg['media_ids']
                msg['media_ids'] = []
            else:
                msg['media_expired'] = False
                if timer:
                    msg['time_remaining'] = max(0, int(timer['expires_at'] - now))
                    msg['upvote_count'] = timer['upvote_extension'] // 60
                    msg['timer_choice'] = timer.get('timer_choice', 300)

            msg['like_count'] = len(msg['likes'])
            messages.append(msg)

        return messages

    def get_message_by_id(self, msg_id: str) -> Optional[Dict]:
        for room_msgs in st.session_state.chat_messages.values():
            for msg in room_msgs:
                if msg['message_id'] == msg_id:
                    return msg
        return None

    def upvote_photo_timer(self, message_id: str, user_id: str) -> bool:
        vote_key = (user_id, message_id)
        if vote_key in st.session_state.user_upvotes:
            return False

        timer = st.session_state.photo_timers.get(message_id)
        if not timer or time.time() > timer['expires_at']:
            return False

        st.session_state.user_upvotes.add(vote_key)
        extension = 60
        timer['upvote_extension'] += extension
        timer['total_duration'] += extension
        timer['expires_at'] += extension
        return True

    def like_message(self, message_id: str, user_id: str):
        for room_msgs in st.session_state.chat_messages.values():
            for msg in room_msgs:
                if msg['message_id'] == message_id:
                    if user_id in msg['likes']:
                        msg['likes'].remove(user_id)
                        return False
                    else:
                        msg['likes'].add(user_id)
                        return True
        return None

    def delete_message(self, message_id: str, user_id: str) -> bool:
        for room_msgs in st.session_state.chat_messages.values():
            for msg in room_msgs:
                if msg['message_id'] == message_id and msg['user_id'] == user_id:
                    st.session_state.deleted_messages.add(message_id)
                    timer = st.session_state.photo_timers.get(message_id)
                    if timer:
                        timer['expires_at'] = 0
                    return True
        return False

    def register_guest(self, username: str) -> str:
        user_id = f"guest_{uuid.uuid4().hex[:10]}"
        st.session_state.guest_users[user_id] = {
            'user_id': user_id,
            'username': username,
            'joined_at': datetime.datetime.now().isoformat(),
            'last_active': time.time(),
            'status': 'online'
        }
        st.session_state.active_sessions.add(user_id)
        return user_id

    def update_activity(self, user_id: str):
        user = st.session_state.guest_users.get(user_id)
        if user:
            user['last_active'] = time.time()
            user['status'] = 'online'

    def get_online_count(self) -> int:
        now = time.time()
        active = set()
        for uid, user in list(st.session_state.guest_users.items()):
            if now - user.get('last_active', 0) < 1800:
                active.add(uid)
            else:
                user['status'] = 'offline'
        st.session_state.active_sessions = active
        return len(active)

    def get_online_users(self) -> List[Dict]:
        now = time.time()
        users = []
        for uid in st.session_state.active_sessions:
            user = st.session_state.guest_users.get(uid)
            if user and now - user.get('last_active', 0) < 1800:
                users.append(user)
        return sorted(users, key=lambda x: x['last_active'], reverse=True)


# ============================================================================
# MEDIA PROCESSOR - Using BytesIO for reliable display
# ============================================================================
class MediaProcessor:
    @staticmethod
    def process_upload(uploaded_file) -> Optional[Dict]:
        if uploaded_file is None:
            return None

        file_bytes = uploaded_file.getvalue()
        file_size = len(file_bytes)

        if file_size > Config.MAX_FILE_SIZE:
            st.error(f"File too large. Max: {Config.MAX_FILE_SIZE/(1024*1024):.0f}MB")
            return None

        file_ext = Path(uploaded_file.name).suffix.lower()
        file_id = str(uuid.uuid4())

        if file_ext in Config.IMAGE_EXTENSIONS:
            return MediaProcessor._process_image(file_id, file_bytes, uploaded_file.name, file_ext)
        elif file_ext in Config.SUPPORTED_VIDEO_FORMATS:
            return MediaProcessor._process_video(file_id, file_bytes, uploaded_file.name, file_ext)
        else:
            st.error(f"Unsupported: {file_ext}")
            return None

    @staticmethod
    def _process_image(file_id, file_bytes, filename, ext):
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img = ImageOps.exif_transpose(img)

            # Store raw bytes for reliable display
            storage = SessionStorage()
            storage.store_upload(file_id, file_bytes, filename, f"image/{ext.replace('.', '')}")

            return {
                'file_id': file_id,
                'type': 'image',
                'filename': filename,
                'dimensions': img.size,
                'size': len(file_bytes)
            }
        except Exception as e:
            st.error(f"Image error: {e}")
            return None

    @staticmethod
    def _process_video(file_id, file_bytes, filename, ext):
        storage = SessionStorage()
        storage.store_upload(file_id, file_bytes, filename, f"video/{ext.replace('.', '')}")
        return {
            'file_id': file_id,
            'type': 'video',
            'filename': filename,
            'size': len(file_bytes)
        }

    @staticmethod
    def get_image_bytes(file_id: str) -> Optional[bytes]:
        upload = SessionStorage().get_upload(file_id)
        if upload and upload['mime_type'].startswith('image'):
            return upload['bytes']
        return None

    @staticmethod
    def get_video_bytes(file_id: str) -> Optional[bytes]:
        upload = SessionStorage().get_upload(file_id)
        if upload and upload['mime_type'].startswith('video'):
            return upload['bytes']
        return None


# ============================================================================
# CSS
# ============================================================================
def inject_css():
    st.markdown("""
    <style>
    /* Main layout */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1000px;
    }

    /* Chat container */
    .chat-container {
        background: #f0f2f5;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 20px;
    }

    /* Message bubbles */
    .msg-row {
        display: flex;
        margin-bottom: 12px;
        width: 100%;
    }
    .msg-row.own {
        justify-content: flex-end;
    }
    .msg-row.other {
        justify-content: flex-start;
    }

    .msg-bubble {
        max-width: 75%;
        padding: 10px 14px;
        border-radius: 14px;
        position: relative;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .msg-bubble.own {
        background: #d9fdd3;
        border-bottom-right-radius: 4px;
        margin-left: auto;
    }
    .msg-bubble.other {
        background: #ffffff;
        border-bottom-left-radius: 4px;
        margin-right: auto;
    }

    .msg-sender {
        font-size: 12px;
        font-weight: 700;
        color: #1f7a1f;
        margin-bottom: 3px;
    }
    .msg-text {
        font-size: 14px;
        color: #111;
        line-height: 1.4;
        margin-bottom: 4px;
    }
    .msg-time {
        font-size: 10px;
        color: #999;
        text-align: right;
        margin-top: 2px;
    }

    /* Reply reference */
    .msg-reply {
        background: rgba(0,0,0,0.05);
        border-left: 3px solid #667eea;
        padding: 4px 8px;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 12px;
        color: #555;
    }

    /* Media */
    .msg-media {
        margin: 6px 0;
        border-radius: 10px;
        overflow: hidden;
    }
    .msg-media img {
        max-width: 100%;
        max-height: 350px;
        border-radius: 10px;
        display: block;
    }

    /* Timer badge */
    .timer-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(255,152,0,0.12);
        color: #e65100;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        margin: 4px 0;
    }
    .timer-expired {
        background: rgba(244,67,54,0.12);
        color: #c62828;
    }

    /* Actions */
    .msg-actions {
        display: flex;
        gap: 6px;
        margin-top: 4px;
        flex-wrap: wrap;
    }

    /* Online badge */
    .online-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .online-dot {
        width: 8px;
        height: 8px;
        background: #4caf50;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    /* Guest login */
    .guest-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 32px;
        border-radius: 20px;
        text-align: center;
        margin: 40px auto;
        max-width: 500px;
    }

    /* Room selector */
    .room-active {
        background: #667eea !important;
        color: white !important;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }

    /* Hide form borders */
    .stForm {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }

    /* Button styling */
    .stButton button {
        border-radius: 20px !important;
        font-size: 12px !important;
        padding: 2px 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# GUEST AUTH
# ============================================================================
class GuestAuth:
    @staticmethod
    def render_login():
        st.markdown("""
        <div class="guest-card">
            <h1>👋 MemoryVault Chat</h1>
            <p style="font-size:16px; opacity:0.9;">Join chat rooms instantly.<br>No account needed.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            username = st.text_input("Choose display name", 
                                   placeholder="e.g., PhotoFan2024",
                                   max_chars=30,
                                   key="guest_username_input")

            if st.button("🚀 Join Chat", use_container_width=True, type="primary"):
                if username and len(username.strip()) >= 2:
                    storage = SessionStorage()
                    user_id = storage.register_guest(username.strip())
                    st.session_state.current_user_id = user_id
                    st.session_state.current_username = username.strip()
                    st.session_state.authenticated = True
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Name must be 2+ characters")
        return False

    @staticmethod
    def check_auth():
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

        if st.session_state.authenticated and 'current_user_id' in st.session_state:
            storage = SessionStorage()
            storage.update_activity(st.session_state.current_user_id)
            return True
        return GuestAuth.render_login()

    @staticmethod
    def logout():
        st.session_state.authenticated = False
        st.session_state.current_user_id = None
        st.session_state.current_username = None
        st.rerun()


# ============================================================================
# CHAT ROOM COMPONENT
# ============================================================================
class ChatRoom:
    def __init__(self, room_name: str):
        self.room_name = room_name
        self.storage = SessionStorage()
        self.processor = MediaProcessor()

    def render_input(self):
        """Render message input with file upload and timer"""

        # Reply indicator
        if st.session_state.get('replying_to'):
            reply_msg = self.storage.get_message_by_id(st.session_state.replying_to)
            if reply_msg:
                st.info(f"💬 Replying to @{reply_msg['username']}: {reply_msg['content'][:60]}...")
                if st.button("❌ Cancel Reply", key="cancel_reply"):
                    st.session_state.replying_to = None
                    st.session_state.replying_to_user = None
                    st.rerun()

        with st.form(key=f"chat_form_{self.room_name}", clear_on_submit=True):
            col1, col2 = st.columns([4,1])
            with col1:
                message_text = st.text_area("Message", 
                                           placeholder="Type a message...",
                                           max_chars=Config.MAX_MESSAGE_LENGTH,
                                           height=60,
                                           label_visibility="collapsed")
            with col2:
                timer_choice = st.selectbox(
                    "Timer",
                    options=list(Config.TIMER_OPTIONS.keys()),
                    index=2,
                    label_visibility="collapsed"
                )

            uploaded_files = st.file_uploader(
                "📎 Attach photos/videos",
                accept_multiple_files=True,
                type=list(Config.ALLOWED_EXTENSIONS),
                label_visibility="collapsed"
            )

            submitted = st.form_submit_button("📤 Send", use_container_width=True, type="primary")

        if submitted:
            if not message_text.strip() and not uploaded_files:
                st.warning("Please write a message or upload media")
                return

            media_metadata = []
            if uploaded_files:
                for file in uploaded_files:
                    meta = self.processor.process_upload(file)
                    if meta:
                        media_metadata.append(meta)

            user_id = st.session_state.current_user_id
            username = st.session_state.current_username
            media_ids = [m['file_id'] for m in media_metadata]
            timer_seconds = Config.TIMER_OPTIONS[timer_choice]

            reply_to = st.session_state.get('replying_to')
            reply_to_user = st.session_state.get('replying_to_user')

            self.storage.add_chat_message(
                room=self.room_name,
                user_id=user_id,
                username=username,
                content=message_text.strip(),
                media_ids=media_ids,
                reply_to=reply_to,
                reply_to_user=reply_to_user,
                timer_duration=timer_seconds
            )

            # Clear reply state
            st.session_state.replying_to = None
            st.session_state.replying_to_user = None

            st.success("Sent! ✅")
            time.sleep(0.2)
            st.rerun()

    def format_time(self, seconds: int) -> str:
        if seconds <= 0:
            return "Expired"
        mins, secs = divmod(seconds, 60)
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def render_message(self, msg: Dict):
        is_own = msg['user_id'] == st.session_state.get('current_user_id')
        row_class = "own" if is_own else "other"
        bubble_class = "own" if is_own else "other"
        current_uid = st.session_state.get('current_user_id', '')

        try:
            ts = datetime.datetime.fromisoformat(msg['timestamp'])
            time_str = ts.strftime("%H:%M")
        except:
            time_str = "now"

        is_liked = current_uid in msg.get('likes', set())
        like_icon = "❤️" if is_liked else "🤍"

        # Reply reference HTML
        reply_html = ""
        if msg.get('reply_to') and msg.get('reply_to_user'):
            reply_html = f"""
            <div class="msg-reply">
                ↩️ Replying to @{msg['reply_to_user']}
            </div>
            """

        # Timer HTML
        timer_html = ""
        if msg.get('media_ids') and not msg.get('media_expired'):
            remaining = msg.get('time_remaining', 0)
            upvotes = msg.get('upvote_count', 0)
            timer_label = self.format_time(msg.get('timer_choice', 300))

            timer_html = f"""
            <div class="timer-badge">
                ⏱️ {self.format_time(remaining)} / {timer_label}
                {f" • ⬆️ +{upvotes}m" if upvotes > 0 else ""}
            </div>
            """
        elif msg.get('media_expired'):
            timer_html = """
            <div class="timer-badge timer-expired">
                ⏱️ Photos expired
            </div>
            """

        # Media HTML - using placeholder, actual display via st.image below
        media_html = ""
        if msg.get('media_ids') and not msg.get('media_expired'):
            media_html = '<div class="msg-media">[Media attached]</div>'

        # Build bubble HTML
        st.markdown(f"""
        <div class="msg-row {row_class}">
            <div class="msg-bubble {bubble_class}">
                <div class="msg-sender">@{msg['username']}</div>
                {reply_html}
                <div class="msg-text">{msg['content']}</div>
                {timer_html}
                {media_html}
                <div class="msg-time">{time_str} {'✓✓' if is_own else ''}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display actual media using st.image/st.video (reliable method)
        if msg.get('media_ids') and not msg.get('media_expired'):
            media_cols = st.columns(min(len(msg['media_ids']), 3))
            for idx, media_id in enumerate(msg['media_ids']):
                with media_cols[idx % 3]:
                    upload = self.storage.get_upload(media_id)
                    if upload:
                        mime = upload['mime_type']
                        if mime.startswith('image'):
                            # Use BytesIO for reliable display [^32^]
                            img_bytes = io.BytesIO(upload['bytes'])
                            st.image(img_bytes, use_container_width=True)
                        elif mime.startswith('video'):
                            vid_bytes = io.BytesIO(upload['bytes'])
                            st.video(vid_bytes)

        # Action buttons
        cols = st.columns([1,1,1,1,6])

        with cols[0]:
            if st.button(f"{like_icon} {msg.get('like_count', 0)}", 
                        key=f"like_{msg['message_id']}_{self.room_name}",
                        help="Like"):
                self.storage.like_message(msg['message_id'], current_uid)
                st.rerun()

        if msg.get('media_ids') and not msg.get('media_expired') and not is_own:
            with cols[1]:
                if st.button("⬆️ +1m", 
                            key=f"upvote_{msg['message_id']}_{self.room_name}",
                            help="Extend timer"):
                    success = self.storage.upvote_photo_timer(msg['message_id'], current_uid)
                    if success:
                        st.toast("Timer extended! +1 minute ✅")
                        time.sleep(0.2)
                        st.rerun()
                    else:
                        st.toast("Already upvoted or expired")

        with cols[2]:
            if st.button("↩️ Reply", 
                        key=f"reply_{msg['message_id']}_{self.room_name}", 
                        help="Reply to this message"):
                st.session_state.replying_to = msg['message_id']
                st.session_state.replying_to_user = msg['username']
                st.rerun()

        with cols[3]:
            if is_own:
                if st.button("🗑️", 
                            key=f"del_{msg['message_id']}_{self.room_name}", 
                            help="Delete"):
                    self.storage.delete_message(msg['message_id'], current_uid)
                    st.rerun()

    def render_messages(self):
        messages = self.storage.get_room_messages(self.room_name, Config.MAX_MESSAGES)

        if not messages:
            st.info(f"No messages in {self.room_name} yet. Start the conversation! 🎉")
            return

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in messages:
            self.render_message(msg)
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    def render(self):
        st.subheader(f"💬 {self.room_name}")

        # Message count
        msg_count = len(st.session_state.chat_messages.get(self.room_name, []))
        st.caption(f"{msg_count} message(s)")

        self.render_messages()
        st.divider()
        self.render_input()


# ============================================================================
# ONLINE SIDEBAR
# ============================================================================
class OnlineSidebar:
    @staticmethod
    def render():
        storage = SessionStorage()
        count = storage.get_online_count()
        users = storage.get_online_users()

        st.sidebar.markdown(f"""
        <div class="online-badge">
            <div class="online-dot"></div>
            <span>{count} online</span>
        </div>
        """, unsafe_allow_html=True)

        st.sidebar.divider()

        st.sidebar.subheader("👥 Active Guests")
        for user in users[:15]:
            st.sidebar.caption(f"🟢 @{user['username']}")

        if len(users) > 15:
            st.sidebar.caption(f"...and {len(users)-15} more")

        st.sidebar.divider()

        # Room list with message counts
        st.sidebar.subheader("📋 Rooms")
        for room in Config.DEFAULT_ROOMS:
            msg_count = len(st.session_state.chat_messages.get(room, []))
            active = "✅" if room == st.session_state.get('current_room') else ""
            st.sidebar.caption(f"{active} {room} ({msg_count})")

        st.sidebar.divider()

        # Stats
        total_guests = len(st.session_state.guest_users)
        total_msgs = sum(len(msgs) for msgs in st.session_state.chat_messages.values())
        total_uploads = len(st.session_state.session_uploads)

        col1, col2 = st.sidebar.columns(2)
        col1.metric("Guests", total_guests)
        col2.metric("Posts", total_msgs)

        st.sidebar.metric("Media", total_uploads)

        st.sidebar.divider()
        st.sidebar.caption("⏳ Photos auto-delete based on timer")
        st.sidebar.caption("⬆️ Upvotes extend by +1 minute")


# ============================================================================
# GALLERY - FIXED
# ============================================================================
class Gallery:
    @staticmethod
    def render():
        uploads = st.session_state.session_uploads

        if not uploads:
            st.info("No photos shared yet. Post in a chat room! 📸")
            return

        st.subheader("📸 Community Gallery")

        # Get all non-expired media IDs across all rooms
        active_media_ids = set()
        now = time.time()
        for timer in st.session_state.photo_timers.values():
            if now < timer['expires_at']:
                for mid in timer.get('media_ids', []):
                    active_media_ids.add(mid)

        images = []
        videos = []
        for fid, data in uploads.items():
            if fid in active_media_ids:
                if data['mime_type'].startswith('image'):
                    images.append((fid, data))
                elif data['mime_type'].startswith('video'):
                    videos.append((fid, data))

        if not images and not videos:
            st.info("No active media. Check chat rooms for new uploads! ⏳")
            return

        st.caption(f"Showing {len(images)} photo(s) and {len(videos)} video(s)")

        # Images grid
        if images:
            st.markdown("**Photos**")
            cols = st.columns(min(len(images), 4))
            for idx, (fid, data) in enumerate(images):
                with cols[idx % 4]:
                    img_bytes = io.BytesIO(data['bytes'])
                    st.image(img_bytes, use_container_width=True)
                    st.caption(f"{data['filename'][:20]}...")

        # Videos
        if videos:
            st.markdown("**Videos**")
            for fid, data in videos:
                vid_bytes = io.BytesIO(data['bytes'])
                st.video(vid_bytes)
                st.caption(f"{data['filename']}")


# ============================================================================
# MAIN APP
# ============================================================================
class ChatApp:
    def __init__(self):
        st.set_page_config(
            page_title=Config.APP_NAME, 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        inject_css()
        self.storage = SessionStorage()

    def render_sidebar(self):
        with st.sidebar:
            st.title("📁 MemoryVault")
            st.caption(f"v{Config.VERSION}")

            OnlineSidebar.render()

            st.subheader("🏠 Room Selection")
            room = st.radio(
                "Choose a room",
                options=Config.DEFAULT_ROOMS,
                index=Config.DEFAULT_ROOMS.index(st.session_state.get('current_room', Config.DEFAULT_ROOMS[0])),
                key="room_selector"
            )

            if room != st.session_state.get('current_room'):
                st.session_state.current_room = room
                st.session_state.replying_to = None
                st.session_state.replying_to_user = None
                st.rerun()

            st.divider()
            st.subheader("Your Profile")
            st.write(f"👤 @{st.session_state.get('current_username', 'Guest')}")
            st.write(f"🆔 ...{st.session_state.get('current_user_id', '')[-8:]}")

            if st.button("🚪 Leave", use_container_width=True):
                GuestAuth.logout()

            st.divider()
            st.caption("💡 Session-based: data clears on refresh")
            st.caption("⏱️ Choose photo timer at upload")

    def render_chat(self):
        room = st.session_state.get('current_room', Config.DEFAULT_ROOMS[0])
        chat = ChatRoom(room)
        chat.render()

    def render_gallery(self):
        Gallery.render()

    def render_about(self):
        st.title("ℹ️ About MemoryVault Chat")
        st.markdown("""
        ### 🌟 v9.0.0 Features

        **Multi-Room Chat:**
        - Join different themed rooms (General, Photo Sharing, Events, etc.)
        - Each room has its own message history
        - Switch rooms instantly

        **WhatsApp-Style Chat:**
        - Your messages on the right (green), others on the left (white)
        - Reply to specific messages with context
        - Like and upvote system

        **Photo Timer & Upvotes:**
        - Choose timer before uploading (1 min to 1 hour)
        - Community upvotes extend timer by +1 minute
        - Photos auto-hide when expired

        **Reliable Photo Display:**
        - Uses BytesIO for 100% reliable image rendering
        - Works on Streamlit Cloud without issues

        **Live Features:**
        - Real-time online counter
        - Active guest list
        - Gallery shows only active photos
        """)

        st.info(f"📊 Currently online: {self.storage.get_online_count()} guests")

    def run(self):
        if not GuestAuth.check_auth():
            return

        self.render_sidebar()

        # Main content area
        tab1, tab2, tab3 = st.tabs(["💬 Chat Room", "📸 Gallery", "ℹ️ About"])

        with tab1:
            self.render_chat()
        with tab2:
            self.render_gallery()
        with tab3:
            self.render_about()


# ============================================================================
# MAIN
# ============================================================================
def main():
    app = ChatApp()
    app.run()

if __name__ == "__main__":
    main()
