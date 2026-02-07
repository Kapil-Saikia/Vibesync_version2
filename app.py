from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import cv2
from tensorflow import keras
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
import hashlib
import secrets
from functools import wraps
from werkzeug.utils import secure_filename
from deepface import DeepFace
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)
load_dotenv()

print("\n🔄 Loading Custom Emotion Detection Model...")
MODEL_PATH = 'fer2013_best_model.keras'

# File upload configuration
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 50 * 1024 * 1024

os.makedirs(os.path.join(UPLOAD_FOLDER, 'audio'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'covers'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'artists'), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ============================================================
# MongoDB Configuration (Keep as is)
# ============================================================
MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://kapilsaikia029_db_user:MUENYsfv7skWMVB7@cluster0.rtp8aea.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("✓ Connected to MongoDB Atlas")
    db = client['vibesync_db']
    songs_collection = db['songs']
except Exception as e:
    print(f"❌ MongoDB Atlas connection failed: {e}")
    print("Falling back to local MongoDB...")
    client = MongoClient('mongodb://localhost:27017/')
    db = client['vibesync_db']
    songs_collection = db['songs']

# ============================================================
# PostgreSQL Configuration (REPLACES SQLite)
# ============================================================
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in .env file!")
    exit(1)

# Parse the DATABASE_URL
url = urlparse(DATABASE_URL)

POSTGRES_CONFIG = {
    'host': url.hostname,
    'port': url.port or 5432,
    'user': url.username,
    'password': url.password,
    'database': url.path[1:],  # Remove leading '/'
    'sslmode': 'require'
}

def get_db_connection():
    """Get PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_CONFIG['host'],
            port=POSTGRES_CONFIG['port'],
            user=POSTGRES_CONFIG['user'],
            password=POSTGRES_CONFIG['password'],
            database=POSTGRES_CONFIG['database'],
            sslmode=POSTGRES_CONFIG['sslmode'],
            cursor_factory=RealDictCursor  # Returns dict-like rows
        )
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL connection error: {e}")
        raise

# ============================================================
# SECURITY MIDDLEWARE
# ============================================================

@app.after_request
def set_no_cache_headers(response):
    """Set no-cache headers for protected pages to prevent back button access after logout"""
    protected_paths = ['/home', '/admin', '/recently', '/favorites', '/profile', '/playlist']
    
    if request.path in protected_paths:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-Content-Type-Options'] = 'nosniff'
    
    return response

# ============================================================
# DATABASE INITIALIZATION (REPLACES init_sqlite)
# ============================================================

def init_postgres():
    """Initialize PostgreSQL database with users and history tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                first_name VARCHAR(100) NOT NULL,
                last_name VARCHAR(100) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(64) NOT NULL,
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # Create admin user if not exists
        admin_email = 'admin@music.com'
        admin_password = 'admin123'
        admin_hash = hash_password(admin_password)
        
        cursor.execute('SELECT id FROM users WHERE email = %s', (admin_email,))
        admin_user = cursor.fetchone()
        
        if not admin_user:
            cursor.execute('''
                INSERT INTO users (first_name, last_name, email, password_hash, is_admin)
                VALUES (%s, %s, %s, %s, %s)
            ''', ('Admin', 'User', admin_email, admin_hash, True))
            print(f"✓ Created admin user: {admin_email} / {admin_password}")
        else:
            # Update existing user to be admin and update password
            cursor.execute('''
                UPDATE users 
                SET is_admin = TRUE, password_hash = %s
                WHERE email = %s
            ''', (admin_hash, admin_email))
            print(f"✓ Updated admin user: {admin_email} / {admin_password}")
        
        # Emotion history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                email VARCHAR(255) NOT NULL,
                emotion VARCHAR(50) NOT NULL,
                confidence REAL NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_user_id ON emotion_history(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_detected_at ON emotion_history(detected_at)')
        
        # Recently played songs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recently_played (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                email VARCHAR(255) NOT NULL,
                song_id VARCHAR(255) NOT NULL,
                song_title VARCHAR(500) NOT NULL,
                artist VARCHAR(500) NOT NULL,
                played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recent_user_id ON recently_played(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_recent_played_at ON recently_played(played_at)')
        
        # Favorites table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS favorites (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                song_id VARCHAR(255) NOT NULL,
                song_title VARCHAR(500) NOT NULL,
                artist VARCHAR(500) NOT NULL,
                cover_url TEXT,
                audio_url TEXT,
                artist_photo_url TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, song_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_fav_user_id ON favorites(user_id)')
        
        # Playlists table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlists (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_playlist_user_id ON playlists(user_id)')
        
        # Playlist songs junction table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS playlist_songs (
                id SERIAL PRIMARY KEY,
                playlist_id INTEGER NOT NULL,
                song_id VARCHAR(255) NOT NULL,
                song_title VARCHAR(500) NOT NULL,
                artist VARCHAR(500) NOT NULL,
                cover_url TEXT,
                audio_url TEXT,
                artist_photo_url TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE,
                UNIQUE(playlist_id, song_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_playlist_songs_playlist ON playlist_songs(playlist_id)')
        
        conn.commit()
        print("✓ PostgreSQL database initialized")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error initializing PostgreSQL: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# ============================================================
# HELPER FUNCTIONS (Keep these as is)
# ============================================================

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def serialize_song(song):
    """Convert MongoDB document to JSON-serializable dict"""
    song['_id'] = str(song['_id'])
    return song

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Login required'}), 401
            else:
                return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Login required'}), 401
        
        if not session.get('is_admin', False):
            return jsonify({'error': 'Admin privileges required'}), 403
        
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename, file_type='audio'):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'audio':
        return ext in ALLOWED_AUDIO_EXTENSIONS
    elif file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    return False

def save_uploaded_file(file, folder):
    """Save uploaded file and return the URL path"""
    if file and allowed_file(file.filename, 'audio' if folder == 'audio' else 'image'):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], folder, filename)
        file.save(filepath)
        return f'/static/uploads/{folder}/{filename}'
    return None

# ============================================================
# AUTHENTICATION ROUTES (Update all queries)
# ============================================================

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Create new user account"""
    try:
        data = request.get_json()
        
        first_name = data.get('firstName', '').strip()
        last_name = data.get('lastName', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not all([first_name, last_name, email, password]):
            return jsonify({'error': 'All fields are required'}), 400
        
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        if '@' not in email:
            return jsonify({'error': 'Invalid email address'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'error': 'Email already registered'}), 400
        
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (first_name, last_name, email, password_hash, is_admin)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        ''', (first_name, last_name, email, password_hash, False))
        
        user_id = cursor.fetchone()['id']
        conn.commit()
        cursor.close()
        conn.close()
        
        session['user_id'] = user_id
        session['email'] = email
        session['first_name'] = first_name
        session['is_admin'] = False
        
        print(f"✓ New user registered and logged in: {email}")
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'redirect': '/home',
            'user': {
                'id': user_id,
                'firstName': first_name,
                'lastName': last_name,
                'email': email,
                'isAdmin': False
            }
        }), 201
        
    except Exception as e:
        print(f"Error in signup: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user (checks for admin)"""
    try:
        data = request.get_json()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, first_name, last_name, email, password_hash, is_active, is_admin
            FROM users WHERE email = %s
        ''', (email,))
        
        user = cursor.fetchone()
        
        if not user:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Invalid email or password'}), 401
        
        password_hash = hash_password(password)
        if user['password_hash'] != password_hash:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if not user['is_active']:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Account is deactivated'}), 403
        
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s
        ''', (user['id'],))
        conn.commit()
        cursor.close()
        conn.close()
        
        session['user_id'] = user['id']
        session['email'] = user['email']
        session['first_name'] = user['first_name']
        session['is_admin'] = bool(user['is_admin'])
        
        redirect_url = '/admin' if user['is_admin'] else '/home'
        
        print(f"✓ User logged in: {email} (Admin: {bool(user['is_admin'])})")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'redirect': redirect_url,
            'user': {
                'id': user['id'],
                'firstName': user['first_name'],
                'lastName': user['last_name'],
                'email': user['email'],
                'isAdmin': bool(user['is_admin'])
            }
        }), 200
        
    except Exception as e:
        print(f"Error in login: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user - clear all session data"""
    session.clear()
    session.permanent = False
    return jsonify({'success': True, 'message': 'Logged out'}), 200

@app.route('/api/auth/me', methods=['GET'])
@login_required
def get_current_user():
    """Get current logged in user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, first_name, last_name, email, is_admin, created_at, last_login
        FROM users WHERE id = %s
    ''', (session['user_id'],))
    
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user['id'],
        'firstName': user['first_name'],
        'lastName': user['last_name'],
        'email': user['email'],
        'isAdmin': bool(user['is_admin']),
        'createdAt': user['created_at'].isoformat() if user['created_at'] else None,
        'lastLogin': user['last_login'].isoformat() if user['last_login'] else None
    }), 200

# ============================================================
# MAIN ROUTES (Keep as is - no database calls here)
# ============================================================

@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('is_admin', False):
            return redirect('/admin')
        return redirect('/home')
    return redirect('/login')

@app.route('/home')
@login_required
def home():
    return render_template('index.html')

@app.route('/admin')
@login_required
def admin():
    if not session.get('is_admin', False):
        return redirect('/home')
    return render_template('admin.html')

@app.route('/recently')
@login_required
def recently_page():
    return render_template('recently.html')

@app.route('/favorites')
@login_required
def favorites_page():
    return render_template('favorites.html')

@app.route('/profile')
@login_required
def profile_page():
    return render_template('profile.html')

@app.route('/playlist')
@login_required
def playlist_page():
    return render_template('playlist.html')

# ============================================================
# RECENTLY PLAYED (SQLite)
# ============================================================

# ============================================================
# EMOTION DETECTION
# ============================================================

@app.route('/detect_emotion', methods=['POST'])
@login_required
def detect_emotion():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Convert to BGR
        if len(image_np.shape) == 3:
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            else:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Save temp image
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, image_np)
        
        # Detect face first with OpenCV for landmarks
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({
                'success': False,
                'message': '😕 No face detected',
                'showFallback': True
            }), 200
        
        # Get the first (largest) face
        x, y, w, h = faces[0]
        face_region = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        
        # Generate stylish landmark points (68 facial landmarks simulation)
        landmarks = []
        
        # Face outline (17 points - 0 to 16)
        for i in range(17):
            angle = (i / 16.0) * np.pi
            lx = int(x + w/2 + (w/2.2) * np.cos(angle + np.pi))
            ly = int(y + h/2 + (h/1.5) * np.sin(angle + np.pi/6))
            landmarks.append({'x': lx, 'y': ly})
        
        # Left eyebrow (5 points - 17 to 21)
        for i in range(5):
            lx = int(x + w * (0.25 + i * 0.05))
            ly = int(y + h * 0.3)
            landmarks.append({'x': lx, 'y': ly})
        
        # Right eyebrow (5 points - 22 to 26)
        for i in range(5):
            lx = int(x + w * (0.55 + i * 0.05))
            ly = int(y + h * 0.3)
            landmarks.append({'x': lx, 'y': ly})
        
        # Nose bridge (4 points - 27 to 30)
        for i in range(4):
            lx = int(x + w/2)
            ly = int(y + h * (0.35 + i * 0.08))
            landmarks.append({'x': lx, 'y': ly})
        
        # Nose base (5 points - 31 to 35)
        for i in range(5):
            lx = int(x + w * (0.35 + i * 0.075))
            ly = int(y + h * 0.6)
            landmarks.append({'x': lx, 'y': ly})
        
        # Left eye (6 points - 36 to 41)
        eye_center_x = x + int(w * 0.3)
        eye_center_y = y + int(h * 0.4)
        for i in range(6):
            angle = (i / 6.0) * 2 * np.pi
            lx = int(eye_center_x + (w * 0.05) * np.cos(angle))
            ly = int(eye_center_y + (h * 0.03) * np.sin(angle))
            landmarks.append({'x': lx, 'y': ly})
        
        # Right eye (6 points - 42 to 47)
        eye_center_x = x + int(w * 0.7)
        for i in range(6):
            angle = (i / 6.0) * 2 * np.pi
            lx = int(eye_center_x + (w * 0.05) * np.cos(angle))
            ly = int(eye_center_y + (h * 0.03) * np.sin(angle))
            landmarks.append({'x': lx, 'y': ly})
        
        # Outer mouth (12 points - 48 to 59)
        mouth_center_x = x + int(w/2)
        mouth_center_y = y + int(h * 0.75)
        for i in range(12):
            angle = (i / 12.0) * 2 * np.pi
            lx = int(mouth_center_x + (w * 0.15) * np.cos(angle))
            ly = int(mouth_center_y + (h * 0.06) * np.sin(angle))
            landmarks.append({'x': lx, 'y': ly})
        
        # Inner mouth (8 points - 60 to 67)
        for i in range(8):
            angle = (i / 8.0) * 2 * np.pi
            lx = int(mouth_center_x + (w * 0.1) * np.cos(angle))
            ly = int(mouth_center_y + (h * 0.04) * np.sin(angle))
            landmarks.append({'x': lx, 'y': ly})
        
        # Now analyze emotion with DeepFace
        result = DeepFace.analyze(
            img_path=temp_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        if isinstance(result, list):
            result = result[0]
        
        dominant_emotion = result['dominant_emotion']
        emotion_scores = {k: float(v) for k, v in result['emotion'].items()}
        confidence = float(emotion_scores[dominant_emotion])
        
        print(f"🎭 {session['email']} - Detected: {dominant_emotion} ({confidence:.1f}%)")
        
        # Save to history (SQLite)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO emotion_history (user_id, email, emotion, confidence)
            VALUES (?, ?, ?, ?)
        ''', (session['user_id'], session['email'], dominant_emotion, confidence))
        conn.commit()
        conn.close()
        
        # Get songs from MongoDB
        songs = list(songs_collection.find({
            'emotions': {'$in': [dominant_emotion.lower()]}
        }).limit(10))
        songs = [serialize_song(song) for song in songs]
        
        emotion_mapping = {
            'angry': 'Angry', 'disgust': 'Disgust', 'fear': 'Fear',
            'happy': 'Happy', 'sad': 'Sad', 'surprise': 'Surprise', 'neutral': 'Neutral'
        }
        
        display_emotion = emotion_mapping.get(dominant_emotion, 'Neutral')
        
        return jsonify({
            'success': True,
            'emotion': display_emotion,
            'confidence': round(confidence, 2),
            'probabilities': emotion_scores,
            'songs': songs,
            'faceRegion': face_region,
            'landmarks': landmarks,  # ← THIS IS THE KEY!
            'message': f'🎭 Mood: {display_emotion}!'
        }), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'message': '😕 No face detected',
            'showFallback': True
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ============================================================
# RECENTLY PLAYED (SQLite)
# ============================================================

@app.route('/api/recently-played', methods=['POST'])
@login_required
def add_recently_played():
    """Add song to recently played"""
    try:
        data = request.get_json()
        
        song_id = data.get('songId')
        song_title = data.get('title')
        artist = data.get('artist')
        
        if not all([song_id, song_title, artist]):
            return jsonify({'error': 'Missing song data'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recently_played (user_id, email, song_id, song_title, artist)
            VALUES (?, ?, ?, ?, ?)
        ''', (session['user_id'], session['email'], song_id, song_title, artist))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recently-played', methods=['GET'])
@login_required
def get_recently_played():
    """Get user's recently played songs with cover images from MongoDB"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT song_id, song_title, artist, played_at
            FROM recently_played
            WHERE user_id = ?
            ORDER BY played_at DESC
            LIMIT 50
        ''', (session['user_id'],))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            song_id = row['song_id']
            
            # ✅ Get FULL song details from MongoDB
            try:
                from bson import ObjectId
                song_doc = songs_collection.find_one({'_id': ObjectId(song_id)})
                
                if song_doc:
                    # Use real data from database
                    cover_url = song_doc.get('coverUrl', f'https://picsum.photos/400/400?random={song_id}')
                    audio_url = song_doc.get('audioUrl', 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3')
                    artist_photo = song_doc.get('artistPhotoUrl', '')
                else:
                    # Fallback if song not found in MongoDB
                    cover_url = f'https://picsum.photos/400/400?random={song_id}'
                    audio_url = 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3'
                    artist_photo = ''
                    
            except Exception as e:
                print(f"Error fetching song {song_id} from MongoDB: {e}")
                cover_url = f'https://picsum.photos/400/400?random={song_id}'
                audio_url = 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3'
                artist_photo = ''
            
            history.append({
                'songId': song_id,
                'title': row['song_title'],
                'artist': row['artist'],
                'playedAt': row['played_at'],
                'coverUrl': cover_url,  # ✅ Real cover from MongoDB
                'audioUrl': audio_url,  # ✅ Real audio from MongoDB
                'artistPhotoUrl': artist_photo  # ✅ Real artist photo from MongoDB
            })
        
        return jsonify(history), 200
        
    except Exception as e:
        print(f"Error in get_recently_played: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/api/emotion-history', methods=['GET'])
@login_required
def get_emotion_history():
    """Get user's emotion detection history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT emotion, confidence, detected_at
            FROM emotion_history
            WHERE user_id = ?
            ORDER BY detected_at DESC
            LIMIT 50
        ''', (session['user_id'],))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = [{
            'emotion': row['emotion'],
            'confidence': row['confidence'],
            'detectedAt': row['detected_at']
        } for row in rows]
        
        return jsonify(history), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# ADMIN ROUTES - SONG MANAGEMENT (MongoDB)
# ============================================================


@app.route('/api/songs', methods=['GET'])
@login_required
def get_all_songs():
    """Get all songs from MongoDB"""
    try:
        songs = list(songs_collection.find().sort('createdAt', -1))
        songs = [serialize_song(song) for song in songs]
        return jsonify(songs), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/songs/by-emotion', methods=['GET'])
@login_required
def get_songs_by_emotion():
    """Get songs matching a specific emotion"""
    try:
        emotion = request.args.get('emotion', '').strip()
        
        if not emotion:
            return jsonify({'error': 'Emotion parameter required'}), 400
        
        # Map display emotion names to lowercase for database query
        emotion_mapping = {
            'Happy': 'happy',
            'Sad': 'sad',
            'Angry': 'angry',
            'Surprise': 'surprise',
            'Fear': 'fear',
            'Disgust': 'disgust',
            'Neutral': 'neutral'
        }
        
        emotion_lower = emotion_mapping.get(emotion, emotion.lower())
        
        # Get songs from MongoDB that match the emotion
        songs = list(songs_collection.find({
            'emotions': {'$in': [emotion_lower]}
        }).limit(20))
        
        songs = [serialize_song(song) for song in songs]
        
        return jsonify(songs), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/songs/upload', methods=['POST'])
@admin_required
def upload_song():
    """Upload song with files (admin only)"""
    try:
        # Get form data
        title = request.form.get('title', '').strip()
        artist = request.form.get('artist', '').strip()
        emotions_json = request.form.get('emotions', '[]')
        language = request.form.get('language', 'English').strip()

        if not title or not artist:
            return jsonify({'error': 'Title and artist required'}), 400
        
        # Parse emotions
        import json
        try:
            emotions = json.loads(emotions_json)
        except:
            emotions = []
        
        if len(emotions) == 0:
            return jsonify({'error': 'At least one emotion required'}), 400
        
        # Handle audio file or URL
        audio_url = ''
        if 'audioFile' in request.files:
            audio_file = request.files['audioFile']
            if audio_file.filename:
                audio_url = save_uploaded_file(audio_file, 'audio')
                if not audio_url:
                    return jsonify({'error': 'Invalid audio file format'}), 400
        else:
            audio_url = request.form.get('audioUrl', '')
        
        # Handle cover image file or URL
        cover_url = ''
        if 'coverFile' in request.files:
            cover_file = request.files['coverFile']
            if cover_file.filename:
                cover_url = save_uploaded_file(cover_file, 'covers')
                if not cover_url:
                    return jsonify({'error': 'Invalid image file format'}), 400
        else:
            cover_url = request.form.get('coverUrl', f'https://picsum.photos/400/400?random={datetime.now().timestamp()}')
        
        # Handle artist photo file or URL
        artist_photo_url = ''
        if 'artistPhotoFile' in request.files:
            artist_photo_file = request.files['artistPhotoFile']
            if artist_photo_file.filename:
                artist_photo_url = save_uploaded_file(artist_photo_file, 'artists')
        else:
            artist_photo_url = request.form.get('artistPhotoUrl', '')
        
        # Create song document
        song = {
            'title': title,
            'artist': artist,
            'coverUrl': cover_url,
            'audioUrl': audio_url,
            'artistPhotoUrl': artist_photo_url,
            'emotions': [e.lower() for e in emotions],
            'language': language,
            'createdAt': datetime.utcnow(),
            'updatedAt': datetime.utcnow(),
            'uploadedBy': session['email']
        }
        
        result = songs_collection.insert_one(song)
        song['_id'] = str(result.inserted_id)
        
        print(f"✓ Song added by {session['email']}: {song['title']}")
        
        return jsonify({'success': True, 'song': song}), 201
        
    except Exception as e:
        print(f"Error uploading song: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/songs', methods=['POST'])
@admin_required
def add_song():
    """Add new song to MongoDB (admin only) - Legacy route for URL-only"""
    try:
        data = request.get_json()
        
        if not data.get('title') or not data.get('artist'):
            return jsonify({'error': 'Title and artist required'}), 400
        
        if not data.get('emotions') or len(data.get('emotions', [])) == 0:
            return jsonify({'error': 'At least one emotion required'}), 400
        
        song = {
            'title': data['title'],
            'artist': data['artist'],
            'coverUrl': data.get('coverUrl', 'https://via.placeholder.com/400'),
            'audioUrl': data.get('audioUrl', ''),
            'artistPhotoUrl': data.get('artistPhotoUrl', ''),
            'emotions': [e.lower() for e in data['emotions']],
            'language': data.get('language', 'English'),
            'createdAt': datetime.utcnow(),
            'updatedAt': datetime.utcnow(),
            'uploadedBy': session['email']
        }
        
        result = songs_collection.insert_one(song)
        song['_id'] = str(result.inserted_id)
        
        print(f"✓ Song added by {session['email']}: {song['title']}")
        
        return jsonify({'success': True, 'song': song}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/songs/by-language', methods=['GET'])
@login_required
def get_songs_by_language():
    """Get songs by language"""
    try:
        language = request.args.get('language', '').strip()
        
        if not language:
            return jsonify({'error': 'Language parameter required'}), 400
        
        # Get songs from MongoDB that match the language
        songs = list(songs_collection.find({
            'language': language
        }).sort('createdAt', -1))
        
        songs = [serialize_song(song) for song in songs]
        
        return jsonify(songs), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/songs/languages', methods=['GET'])
@login_required
def get_available_languages():
    """Get list of all available languages"""
    try:
        # Get distinct languages from MongoDB
        languages = songs_collection.distinct('language')
        
        # Filter out None/empty and sort
        languages = [lang for lang in languages if lang]
        languages.sort()
        
        return jsonify(languages), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500








@app.route('/api/songs/<song_id>', methods=['DELETE'])
@admin_required
def delete_song(song_id):
    """Delete song from MongoDB (admin only)"""
    try:
        result = songs_collection.delete_one({'_id': ObjectId(song_id)})
        if result.deleted_count == 0:
            return jsonify({'error': 'Song not found'}), 404
        
        print(f"✓ Song deleted by {session['email']}: {song_id}")
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/songs/<song_id>', methods=['PUT'])
@admin_required
def update_song(song_id):
    """Update song in MongoDB (admin only)"""
    try:
        data = request.get_json()
        
        if not data.get('title') or not data.get('artist'):
            return jsonify({'error': 'Title and artist required'}), 400
        
        if not data.get('emotions') or len(data.get('emotions', [])) == 0:
            return jsonify({'error': 'At least one emotion required'}), 400
        
        # Build update document
        update_data = {
            'title': data['title'],
            'artist': data['artist'],
            'emotions': [e.lower() for e in data['emotions']],
            'updatedAt': datetime.utcnow()
        }
        
        # Only update if provided
        if data.get('coverUrl'):
            update_data['coverUrl'] = data['coverUrl']
        if data.get('artistPhotoUrl'):
            update_data['artistPhotoUrl'] = data['artistPhotoUrl']
        if data.get('audioUrl'):
            update_data['audioUrl'] = data['audioUrl']
        
        # Update in MongoDB
        result = songs_collection.update_one(
            {'_id': ObjectId(song_id)},
            {'$set': update_data}
        )
        
        if result.matched_count == 0:
            return jsonify({'error': 'Song not found'}), 404
        
        print(f"✓ Song updated by {session['email']}: {data['title']}")
        
        return jsonify({
            'success': True,
            'message': 'Song updated successfully'
        }), 200
        
    except Exception as e:
        print(f"Error updating song: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================
# ADMIN ROUTES - USER MANAGEMENT (SQLite)
# ============================================================

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_all_users():
    """Get all users (admin only)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, first_name, last_name, email, is_admin, is_active, 
                   created_at, last_login
            FROM users
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        users = [{
            'id': row['id'],
            'firstName': row['first_name'],
            'lastName': row['last_name'],
            'email': row['email'],
            'isAdmin': bool(row['is_admin']),
            'isActive': bool(row['is_active']),
            'createdAt': row['created_at'],
            'lastLogin': row['last_login']
        } for row in rows]
        
        return jsonify(users), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>/toggle-status', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    """Toggle user active status (admin only)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT is_active FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        new_status = 0 if user['is_active'] else 1
        cursor.execute('UPDATE users SET is_active = ? WHERE id = ?', (new_status, user_id))
        conn.commit()
        conn.close()
        
        print(f"✓ User status toggled by {session['email']}: User ID {user_id} -> Active: {bool(new_status)}")
        
        return jsonify({'success': True, 'isActive': bool(new_status)}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_admin_stats():
    """Get admin dashboard statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total users
        cursor.execute('SELECT COUNT(*) as count FROM users')
        total_users = cursor.fetchone()['count']
        
        # Active users
        cursor.execute('SELECT COUNT(*) as count FROM users WHERE is_active = 1')
        active_users = cursor.fetchone()['count']
        
        # Total plays
        cursor.execute('SELECT COUNT(*) as count FROM recently_played')
        total_plays = cursor.fetchone()['count']
        
        # Total emotions detected
        cursor.execute('SELECT COUNT(*) as count FROM emotion_history')
        total_emotions = cursor.fetchone()['count']
        
        conn.close()
        
        # Total songs from MongoDB
        total_songs = songs_collection.count_documents({})
        
        return jsonify({
            'totalUsers': total_users,
            'activeUsers': active_users,
            'totalSongs': total_songs,
            'totalPlays': total_plays,
            'totalEmotions': total_emotions
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>/emotion-history', methods=['GET'])
@admin_required
def get_user_emotion_history(user_id):
    """Get user's emotion detection history (admin only)"""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT emotion, confidence, detected_at
            FROM emotion_history
            WHERE user_id = ?
            ORDER BY detected_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = [{
            'emotion': row['emotion'].capitalize(),
            'confidence': row['confidence'],
            'detectedAt': row['detected_at']
        } for row in rows]
        
        return jsonify(history), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>/recently-played', methods=['GET'])
@admin_required
def get_user_recently_played(user_id):
    """Get user's recently played songs (admin only)"""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT song_id, song_title, artist, played_at
            FROM recently_played
            WHERE user_id = ?
            ORDER BY played_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = [{
            'songId': row['song_id'],
            'songTitle': row['song_title'],
            'artist': row['artist'],
            'playedAt': row['played_at']
        } for row in rows]
        
        return jsonify(history), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>/activity-charts', methods=['GET'])
@admin_required
def get_user_activity_charts(user_id):
    """Get user activity data for charts (admin only)"""
    try:
        period = request.args.get('period', 'weekly')  # 'weekly' or 'monthly'
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        from datetime import datetime, timedelta
        
        # Calculate date range
        if period == 'weekly':
            days = 7
            date_format = '%a'  # Day of week (Mon, Tue, etc)
        else:  # monthly
            days = 30
            date_format = '%b %d'  # Month Day (Jan 01, Jan 02, etc)
        
        # Generate labels (last N days)
        labels = []
        dates = []
        today = datetime.now().date()
        
        for i in range(days - 1, -1, -1):
            date = today - timedelta(days=i)
            dates.append(date.strftime('%Y-%m-%d'))
            if period == 'weekly':
                labels.append(date.strftime('%a'))
            else:
                labels.append(date.strftime('%b %d'))
        
        # Get listening activity (songs played per day)
        listening_data = []
        for date_str in dates:
            cursor.execute('''
                SELECT COUNT(*) as count
                FROM recently_played
                WHERE user_id = ?
                AND DATE(played_at) = ?
            ''', (user_id, date_str))
            result = cursor.fetchone()
            listening_data.append(result['count'] if result else 0)
        
        # Get emotion distribution
        cursor.execute('''
            SELECT emotion, COUNT(*) as count
            FROM emotion_history
            WHERE user_id = ?
            AND DATE(detected_at) >= DATE('now', '-' || ? || ' days')
            GROUP BY emotion
            ORDER BY count DESC
        ''', (user_id, days))
        
        emotion_rows = cursor.fetchall()
        emotion_labels = []
        emotion_data = []
        
        # Capitalize emotion names for display
        emotion_map = {
            'happy': '😊 Happy',
            'sad': '😢 Sad',
            'angry': '😠 Angry',
            'surprise': '😲 Surprise',
            'fear': '😨 Fear',
            'disgust': '🤢 Disgust',
            'neutral': '😐 Neutral'
        }
        
        for row in emotion_rows:
            emotion_name = row['emotion'].lower()
            display_name = emotion_map.get(emotion_name, emotion_name.capitalize())
            emotion_labels.append(display_name)
            emotion_data.append(row['count'])
        
        conn.close()
        
        return jsonify({
            'listeningActivity': {
                'labels': labels,
                'data': listening_data
            },
            'emotionDistribution': {
                'labels': emotion_labels,
                'data': emotion_data
            }
        }), 200
        
    except Exception as e:
        print(f"Error getting activity charts: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/users/<int:user_id>/change-password', methods=['POST'])
@admin_required
def change_user_password(user_id):
    """Change user's password (admin only)"""
    try:
        data = request.get_json()
        new_password = data.get('newPassword', '')
        
        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT id, email FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        # Update password
        password_hash = hash_password(new_password)
        cursor.execute('''
            UPDATE users SET password_hash = ? WHERE id = ?
        ''', (password_hash, user_id))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Password changed by admin {session['email']} for user: {user['email']}")
        
        return jsonify({'success': True, 'message': 'Password changed successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add these routes to your Flask app (after the existing /api/recently-played routes)

@app.route('/api/recently-played/delete', methods=['POST'])
@login_required
def delete_recently_played():
    """Delete a specific song from recently played history"""
    try:
        data = request.get_json()
        
        song_id = data.get('songId')
        played_at = data.get('playedAt')
        
        if not song_id or not played_at:
            return jsonify({'error': 'Missing data'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete the specific entry
        cursor.execute('''
            DELETE FROM recently_played
            WHERE user_id = ? AND song_id = ? AND played_at = ?
            LIMIT 1
        ''', (session['user_id'], song_id, played_at))
        
        conn.commit()
        conn.close()
        
        print(f"✓ History item deleted by {session['email']}")
        
        return jsonify({'success': True, 'message': 'Deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting history item: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recently-played/clear', methods=['POST'])
@login_required
def clear_recently_played():
    """Clear all recently played history for current user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete all history for this user
        cursor.execute('''
            DELETE FROM recently_played
            WHERE user_id = ?
        ''', (session['user_id'],))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✓ All history cleared by {session['email']} ({deleted_count} items)")
        
        return jsonify({
            'success': True, 
            'message': 'History cleared',
            'deletedCount': deleted_count
        }), 200
        
    except Exception as e:
        print(f"Error clearing history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================
# FAVORITES (SQLite)
# ============================================================

@app.route('/api/favorites', methods=['GET'])
@login_required
def get_favorites():
    """Get user's favorite songs"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT song_id, song_title, artist, cover_url, audio_url, artist_photo_url, added_at
            FROM favorites
            WHERE user_id = ?
            ORDER BY added_at DESC
        ''', (session['user_id'],))
        
        rows = cursor.fetchall()
        conn.close()
        
        favorites = [{
            'id': row['song_id'],
            'title': row['song_title'],
            'artist': row['artist'],
            'img': row['cover_url'] or f'https://picsum.photos/400/400?random={row["song_id"]}',
            'coverUrl': row['cover_url'] or f'https://picsum.photos/400/400?random={row["song_id"]}',
            'audioUrl': row['audio_url'] or 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
            'artistPhotoUrl': row['artist_photo_url'] or '',
            'addedAt': row['added_at']
        } for row in rows]
        
        return jsonify(favorites), 200
        
    except Exception as e:
        print(f"Error getting favorites: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/favorites', methods=['POST'])
@login_required
def add_favorite():
    """Add song to favorites"""
    try:
        data = request.get_json()
        
        song_id = data.get('songId') or data.get('id')
        song_title = data.get('title')
        artist = data.get('artist')
        cover_url = data.get('coverUrl') or data.get('img')
        audio_url = data.get('audioUrl')
        artist_photo_url = data.get('artistPhotoUrl')
        
        if not all([song_id, song_title, artist]):
            return jsonify({'error': 'Missing song data'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if already favorited
        cursor.execute('SELECT id FROM favorites WHERE user_id = ? AND song_id = ?', 
                      (session['user_id'], song_id))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': True, 'message': 'Already in favorites'}), 200
        
        # Add to favorites
        cursor.execute('''
            INSERT INTO favorites (user_id, song_id, song_title, artist, cover_url, audio_url, artist_photo_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], song_id, song_title, artist, cover_url, audio_url, artist_photo_url))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Favorite added by {session['email']}: {song_title}")
        return jsonify({'success': True, 'message': 'Added to favorites'}), 201
        
    except Exception as e:
        print(f"Error adding favorite: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/favorites/<song_id>', methods=['DELETE'])
@login_required
def remove_favorite(song_id):
    """Remove song from favorites"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM favorites
            WHERE user_id = ? AND song_id = ?
        ''', (session['user_id'], song_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Favorite not found'}), 404
        
        conn.commit()
        conn.close()
        
        print(f"✓ Favorite removed by {session['email']}: {song_id}")
        return jsonify({'success': True, 'message': 'Removed from favorites'}), 200
        
    except Exception as e:
        print(f"Error removing favorite: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/favorites/check/<song_id>', methods=['GET'])
@login_required
def check_favorite(song_id):
    """Check if song is favorited"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM favorites WHERE user_id = ? AND song_id = ?',
                      (session['user_id'], song_id))
        is_favorited = cursor.fetchone() is not None
        
        conn.close()
        return jsonify({'isFavorited': is_favorited}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# PROFILE MANAGEMENT
# ============================================================

@app.route('/api/profile', methods=['PUT'])
@login_required
def update_profile():
    """Update user profile"""
    try:
        data = request.get_json()
        
        first_name = data.get('firstName', '').strip()
        last_name = data.get('lastName', '').strip()
        
        if not first_name or not last_name:
            return jsonify({'error': 'First name and last name required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users
            SET first_name = ?, last_name = ?
            WHERE id = ?
        ''', (first_name, last_name, session['user_id']))
        
        conn.commit()
        conn.close()
        
        # Update session
        session['first_name'] = first_name
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'user': {
                'id': session['user_id'],
                'firstName': first_name,
                'lastName': last_name,
                'email': session['email']
            }
        }), 200
        
    except Exception as e:
        print(f"Error updating profile: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile/password', methods=['POST'])
@login_required
def change_password():
    """Change user password"""
    try:
        data = request.get_json()
        
        current_password = data.get('currentPassword', '')
        new_password = data.get('newPassword', '')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Current and new password required'}), 400
        
        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Verify current password
        cursor.execute('''
            SELECT password_hash FROM users WHERE id = ?
        ''', (session['user_id'],))
        
        user = cursor.fetchone()
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        
        current_hash = hash_password(current_password)
        if user['password_hash'] != current_hash:
            conn.close()
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Update password
        new_hash = hash_password(new_password)
        cursor.execute('''
            UPDATE users SET password_hash = ? WHERE id = ?
        ''', (new_hash, session['user_id']))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Password changed successfully'}), 200
        
    except Exception as e:
        print(f"Error changing password: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================
# PLAYLISTS (SQLite)
# ============================================================

@app.route('/api/playlists', methods=['GET'])
@login_required
def get_playlists():
    """Get user's playlists with songs"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get playlists
        cursor.execute('''
            SELECT id, name, description, created_at, updated_at
            FROM playlists
            WHERE user_id = ?
            ORDER BY updated_at DESC
        ''', (session['user_id'],))
        
        playlists_rows = cursor.fetchall()
        playlists = []
        
        for playlist_row in playlists_rows:
            playlist_id = playlist_row['id']
            
            # Get songs for this playlist
            cursor.execute('''
                SELECT song_id, song_title, artist, cover_url, audio_url, artist_photo_url
                FROM playlist_songs
                WHERE playlist_id = ?
                ORDER BY added_at ASC
            ''', (playlist_id,))
            
            songs_rows = cursor.fetchall()
            songs = [{
                'id': row['song_id'],
                'title': row['song_title'],
                'artist': row['artist'],
                'img': row['cover_url'] or f'https://picsum.photos/400/400?random={row["song_id"]}',
                'coverUrl': row['cover_url'] or f'https://picsum.photos/400/400?random={row["song_id"]}',
                'audioUrl': row['audio_url'] or 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
                'artistPhotoUrl': row['artist_photo_url'] or ''
            } for row in songs_rows]
            
            playlists.append({
                'id': playlist_id,
                'name': playlist_row['name'],
                'description': playlist_row['description'],
                'songs': songs,
                'createdAt': playlist_row['created_at'],
                'updatedAt': playlist_row['updated_at']
            })
        
        conn.close()
        return jsonify(playlists), 200
        
    except Exception as e:
        print(f"Error getting playlists: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists', methods=['POST'])
@login_required
def create_playlist():
    """Create new playlist"""
    try:
        data = request.get_json()
        
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        
        if not name:
            return jsonify({'error': 'Playlist name required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO playlists (user_id, name, description)
            VALUES (?, ?, ?)
        ''', (session['user_id'], name, description))
        
        playlist_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✓ Playlist created by {session['email']}: {name}")
        return jsonify({
            'success': True,
            'playlist': {
                'id': playlist_id,
                'name': name,
                'description': description,
                'songs': [],
                'createdAt': datetime.now().isoformat(),
                'updatedAt': datetime.now().isoformat()
            }
        }), 201
        
    except Exception as e:
        print(f"Error creating playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<int:playlist_id>', methods=['PUT'])
@login_required
def update_playlist(playlist_id):
    """Update playlist"""
    try:
        data = request.get_json()
        
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        
        if not name:
            return jsonify({'error': 'Playlist name required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check ownership
        cursor.execute('SELECT user_id FROM playlists WHERE id = ?', (playlist_id,))
        playlist = cursor.fetchone()
        
        if not playlist:
            conn.close()
            return jsonify({'error': 'Playlist not found'}), 404
        
        if playlist['user_id'] != session['user_id']:
            conn.close()
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Update playlist
        cursor.execute('''
            UPDATE playlists
            SET name = ?, description = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (name, description, playlist_id))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Playlist updated by {session['email']}: {name}")
        return jsonify({'success': True, 'message': 'Playlist updated'}), 200
        
    except Exception as e:
        print(f"Error updating playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<int:playlist_id>', methods=['DELETE'])
@login_required
def delete_playlist(playlist_id):
    """Delete playlist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check ownership
        cursor.execute('SELECT user_id, name FROM playlists WHERE id = ?', (playlist_id,))
        playlist = cursor.fetchone()
        
        if not playlist:
            conn.close()
            return jsonify({'error': 'Playlist not found'}), 404
        
        if playlist['user_id'] != session['user_id']:
            conn.close()
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Delete playlist (cascade will delete songs)
        cursor.execute('DELETE FROM playlists WHERE id = ?', (playlist_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Playlist deleted by {session['email']}: {playlist['name']}")
        return jsonify({'success': True, 'message': 'Playlist deleted'}), 200
        
    except Exception as e:
        print(f"Error deleting playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<int:playlist_id>/songs', methods=['POST'])
@login_required
def add_song_to_playlist(playlist_id):
    """Add song to playlist"""
    try:
        data = request.get_json()
        
        song_id = data.get('songId') or data.get('id')
        song_title = data.get('title')
        artist = data.get('artist')
        cover_url = data.get('coverUrl') or data.get('img')
        audio_url = data.get('audioUrl')
        artist_photo_url = data.get('artistPhotoUrl')
        
        if not all([song_id, song_title, artist]):
            return jsonify({'error': 'Missing song data'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check ownership
        cursor.execute('SELECT user_id FROM playlists WHERE id = ?', (playlist_id,))
        playlist = cursor.fetchone()
        
        if not playlist:
            conn.close()
            return jsonify({'error': 'Playlist not found'}), 404
        
        if playlist['user_id'] != session['user_id']:
            conn.close()
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Check if song already in playlist
        cursor.execute('SELECT id FROM playlist_songs WHERE playlist_id = ? AND song_id = ?',
                      (playlist_id, song_id))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': True, 'message': 'Song already in playlist'}), 200
        
        # Add song to playlist
        cursor.execute('''
            INSERT INTO playlist_songs (playlist_id, song_id, song_title, artist, cover_url, audio_url, artist_photo_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (playlist_id, song_id, song_title, artist, cover_url, audio_url, artist_photo_url))
        
        # Update playlist updated_at
        cursor.execute('''
            UPDATE playlists SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
        ''', (playlist_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Song added to playlist by {session['email']}: {song_title}")
        return jsonify({'success': True, 'message': 'Song added to playlist'}), 201
        
    except Exception as e:
        print(f"Error adding song to playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<int:playlist_id>/songs/<song_id>', methods=['DELETE'])
@login_required
def remove_song_from_playlist(playlist_id, song_id):
    """Remove song from playlist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check ownership
        cursor.execute('SELECT user_id FROM playlists WHERE id = ?', (playlist_id,))
        playlist = cursor.fetchone()
        
        if not playlist:
            conn.close()
            return jsonify({'error': 'Playlist not found'}), 404
        
        if playlist['user_id'] != session['user_id']:
            conn.close()
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Remove song
        cursor.execute('''
            DELETE FROM playlist_songs
            WHERE playlist_id = ? AND song_id = ?
        ''', (playlist_id, song_id))
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Song not found in playlist'}), 404
        
        # Update playlist updated_at
        cursor.execute('''
            UPDATE playlists SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
        ''', (playlist_id,))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Song removed from playlist by {session['email']}")
        return jsonify({'success': True, 'message': 'Song removed from playlist'}), 200
        
    except Exception as e:
        print(f"Error removing song from playlist: {str(e)}")
        return jsonify({'error': str(e)}), 500
# ============================================================
# MAIN
# ============================================================

# ============================================================
# AUTO-INITIALIZE DATABASE (runs on startup)
# ============================================================



def initialize_app():
    """Initialize database on app startup"""
    try:
        init_postgres()  # Changed from init_sqlite()
        songs_collection.create_index('emotions')
        print("\n" + "="*60)
        print("🎵 VIBESYNC - DATABASE INITIALIZED")
        print("="*60)
        print(f"\n🐘 PostgreSQL: Connected to Neon")
        print(f"📦 MongoDB: {MONGO_URI[:50]}...")
        print(f"📊 Songs in DB: {songs_collection.count_documents({})}")
        print("\n👤 Admin Credentials:")
        print("   Email: admin@music.com")
        print("   Pass:  admin123")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error initializing database: {e}")

initialize_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

    
    print("\n🌐 URLs:")
    print("   Main:   http://localhost:5000")
    print("   Login:  http://localhost:5000/login")
    print("   Signup: http://localhost:5000/signup")
    print("   Home:   http://localhost:5000/home")
    print("   Admin:  http://localhost:5000/admin")
    print("="*60 + "\n")



