from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_session import Session
import os
import re
import threading
import queue
import time
import random
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import yt_dlp
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from dotenv import load_dotenv
import requests
from urllib.parse import urlencode
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TRCK, TCON, APIC
import base64
import shutil
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['DOWNLOAD_FOLDER'] = 'downloads'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

Session(app)

# Ensure folders exist
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
os.makedirs('./.spotify_cache/', exist_ok=True)

# Configuration
SPOTIFY_SCOPE = "user-library-read playlist-read-private user-read-private user-read-email user-library-read"
CACHE_FOLDER = './.spotify_cache/'

# Password protection (change this to your desired password)
APP_PASSWORD_HASH = generate_password_hash(os.environ.get('APP_PASSWORD'))

def password_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'authenticated' not in session or not session['authenticated']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if check_password_hash(APP_PASSWORD_HASH, password):
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid password")

    downloader = SpotifyDownloader()
    if hasattr(downloader, 'auth_url') and downloader.auth_url:
        return render_template('login.html', spotify_auth_url=downloader.auth_url)
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    # Also remove the cache file
    cache_path = os.path.join(CACHE_FOLDER, '.spotify_cache')
    if os.path.exists(cache_path):
        os.remove(cache_path)
    return redirect(url_for('login'))

@app.route('/')
@password_required
def index():
    return render_template('index.html')

class SpotifyDownloader:
    def __init__(self):
        self.sp = None
        self.auth_url = None
        self.error = None
        self.initialize_spotify()
    
    def initialize_spotify(self):
        """Initialize Spotify client with OAuth"""
        try:
            auth_manager = SpotifyOAuth(
                client_id=os.environ.get('SPOTIFY_CLIENT_ID'),
                client_secret=os.environ.get('SPOTIFY_CLIENT_SECRET'),
                redirect_uri=os.environ.get('SPOTIFY_REDIRECT_URI', 'http://localhost:5005/callback'),
                scope=SPOTIFY_SCOPE,
                cache_path=os.path.join(CACHE_FOLDER, '.spotify_cache'),
                show_dialog=True
            )
            
            # Try to get a valid token
            token_info = auth_manager.get_cached_token()
            
            if not token_info:
                # No valid token found, need to authenticate
                self.auth_url = auth_manager.get_authorize_url()
                return {'status': 'auth_required', 'auth_url': self.auth_url}
            
            # Check if token is expired
            if auth_manager.is_token_expired(token_info):
                # Try to refresh the token
                token_info = auth_manager.refresh_access_token(token_info['refresh_token'])
                
                if not token_info:
                    # Refresh failed, need to reauthenticate
                    self.auth_url = auth_manager.get_authorize_url()
                    return {'status': 'auth_required', 'auth_url': self.auth_url}
            
            # We have a valid token, create Spotify client
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Verify authentication works
            try:
                user = self.sp.current_user()
                session['spotify_user'] = user['display_name'] or user['id']
                session['spotify_authenticated'] = True
                return {'status': 'success'}
            except Exception as e:
                # Token might be invalid, clear it and require reauthentication
                if os.path.exists(auth_manager.cache_path):
                    os.remove(auth_manager.cache_path)
                self.auth_url = auth_manager.get_authorize_url()
                return {'status': 'auth_required', 'auth_url': self.auth_url}
                
        except Exception as e:
            self.error = str(e)
            logger.error(f"Spotify initialization error: {e}")
            return {'status': 'error', 'message': self.error}
    
    def extract_content_id(self, url):
        """Extract content ID from Spotify URL"""
        try:
            url = url.strip()
            
            # Handle Spotify URLs
            if 'open.spotify.com' in url:
                patterns = [
                    r'open\.spotify\.com/(playlist|album|track|episode|show)/([a-zA-Z0-9]+)',
                    r'spotify:(playlist|album|track|episode|show):([a-zA-Z0-9]+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, url)
                    if match:
                        return {"type": match.group(1), "id": match.group(2)}
            
            return None
        except Exception as e:
            logger.error(f"Error extracting content ID: {e}")
            return None
    
    def get_content_tracks(self, content_url):
        """Get all tracks from Spotify content"""
        # Check if Spotify client is initialized
        if not self.sp:
            if self.auth_url:
                return "Please authenticate with Spotify first", None
            elif self.error:
                return f"Spotify authentication error: {self.error}", None
            else:
                return "Spotify client not initialized", None

        content_info = self.extract_content_id(content_url)

        if not content_info:
            return "Invalid Spotify URL", None

        try:
            content_type = content_info["type"]
            content_id = content_info["id"]
            content_data = None
            tracks = []

            if content_type == "playlist":
                content_data = self.sp.playlist(content_id)
                results = self.sp.playlist_tracks(content_id)
                
                while results:
                    for item in results['items']:
                        track = item.get('track')
                        if track and track.get('name') and not track.get('is_local', False):
                            tracks.append(self._create_track_info(track, len(tracks) + 1))
                    
                    if results['next']:
                        results = self.sp.next(results)
                    else:
                        break

            elif content_type == "album":
                content_data = self.sp.album(content_id)
                results = self.sp.album_tracks(content_id)
                
                while results:
                    for item in results['items']:
                        if item and item.get('name'):
                            tracks.append(self._create_track_info(item, len(tracks) + 1, content_data))
                    
                    if results['next']:
                        results = self.sp.next(results)
                    else:
                        break

            elif content_type == "track":
                content_data = self.sp.track(content_id)
                tracks.append(self._create_track_info(content_data, 1))
                
            elif content_type == "episode":
                content_data = self.sp.episode(content_id)
                tracks.append(self._create_episode_info(content_data, 1))
                
            elif content_type == "show":
                content_data = self.sp.show(content_id)
                results = self.sp.show_episodes(content_id)
                
                while results:
                    for item in results['items']:
                        if item and item.get('name'):
                            tracks.append(self._create_episode_info(item, len(tracks) + 1, content_data))
                    
                    if results['next']:
                        results = self.sp.next(results)
                    else:
                        break

            else:
                return f"Content type '{content_type}' not supported yet", None

            if not tracks:
                return "No downloadable tracks found", None

            return tracks, content_data

        except Exception as e:
            error_msg = f"Error fetching content: {str(e)}"
            logger.error(error_msg)
            return error_msg, None
    
    def _create_track_info(self, track_data, track_number, album_data=None):
        """Create a standardized track info dictionary"""
        track_name = track_data.get('name', 'Unknown Track')
        artists = [artist.get('name', 'Unknown Artist') for artist in track_data.get('artists', [])]
        
        if album_data:
            album_name = album_data.get('name', 'Unknown Album')
            cover_url = album_data.get('images', [{}])[0].get('url') if album_data.get('images') else None
        else:
            album_name = track_data.get('album', {}).get('name', 'Unknown Album')
            cover_url = track_data.get('album', {}).get('images', [{}])[0].get('url') if track_data.get('album', {}).get('images') else None
        
        safe_filename = self.sanitize_filename(f"{track_name} - {', '.join(artists)}")
        
        return {
            'name': f"{track_name} - {', '.join(artists)}",
            'title': track_name,
            'artists': artists,
            'album': album_name,
            'filename': safe_filename,
            'search_query': f"{track_name} {', '.join(artists)} official audio",
            'track_number': track_number,
            'duration_ms': track_data.get('duration_ms', 0),
            'spotify_id': track_data.get('id', ''),
            'type': 'track',
            'cover_url': cover_url
        }
    
    def _create_episode_info(self, episode_data, episode_number, show_data=None):
        """Create a standardized episode info dictionary"""
        episode_name = episode_data.get('name', 'Unknown Episode')
        
        if show_data:
            show_name = show_data.get('name', 'Unknown Show')
            cover_url = show_data.get('images', [{}])[0].get('url') if show_data.get('images') else None
        else:
            show_name = episode_data.get('show', {}).get('name', 'Unknown Show')
            cover_url = episode_data.get('images', [{}])[0].get('url') if episode_data.get('images') else None
        
        safe_filename = self.sanitize_filename(f"{episode_name} - {show_name}")
        
        return {
            'name': f"{episode_name} - {show_name}",
            'title': episode_name,
            'artists': [show_name],
            'album': show_name,
            'filename': safe_filename,
            'search_query': f"{episode_name} {show_name} official",
            'track_number': episode_number,
            'duration_ms': episode_data.get('duration_ms', 0),
            'spotify_id': episode_data.get('id', ''),
            'type': 'episode',
            'cover_url': cover_url
        }
    
    def sanitize_filename(self, filename):
        """Clean filename for safe saving"""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace multiple spaces with single space
        filename = re.sub(r'\s+', ' ', filename)
        # Limit length
        return filename.strip()[:200]
    
    def add_metadata(self, file_path, track_info):
        """Add metadata to downloaded file"""
        try:
            audio = MP3(file_path, ID3=ID3)
            
            # Add ID3 tag if it doesn't exist
            try:
                audio.add_tags()
            except Exception:
                pass  # Tags already exist
                
            # Set basic metadata
            audio['TIT2'] = TIT2(encoding=3, text=track_info['title'])
            audio['TPE1'] = TPE1(encoding=3, text="/".join(track_info['artists']))
            
            # Add album if available
            if 'album' in track_info and track_info['album']:
                audio['TALB'] = TALB(encoding=3, text=track_info['album'])
            
            # Add cover art if available
            if 'cover_url' in track_info and track_info['cover_url']:
                try:
                    response = requests.get(track_info['cover_url'], timeout=10)
                    if response.status_code == 200:
                        audio['APIC'] = APIC(
                            encoding=3,
                            mime='image/jpeg',
                            type=3,  # Cover image
                            desc='Cover',
                            data=response.content
                        )
                except Exception as e:
                    logger.warning(f"Error adding cover art: {e}")
                    
            audio.save()
        except Exception as e:
            logger.error(f"Error adding metadata: {e}")
    
    def download_track(self, track_info, output_folder, status_queue, retry_count=0, quality='192'):
        """Download a single track"""
        # Define filename early to ensure it's available in error handling
        filename = track_info.get('filename', 'unknown_track')
        
        try:
            search_query = track_info.get('search_query', '')
            
            # If search_query is missing, create one
            if not search_query:
                title = track_info.get('title', '')
                artists = track_info.get('artists', [])
                artists_str = ', '.join(artists) if isinstance(artists, list) else str(artists)
                search_query = f"{title} {artists_str} official audio"
                track_info['search_query'] = search_query
            
            status_queue.put({
                'track_number': track_info['track_number'],
                'status': 'searching',
                'message': 'Searching...', 
                'percent': 0
            })
            
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            ]
            
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": os.path.join(output_folder, f"{filename}.%(ext)s"),
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": quality,
                    }
                ],
                "quiet": True,
                "no_warnings": False,
                "noplaylist": True,
                "socket_timeout": 30,
                "retries": 3,
                "fragment_retries": 3,
                "http_headers": {
                    "User-Agent": random.choice(user_agents)
                },
                "geo_bypass": True,
                "nocheckcertificate": True,
                "ignoreerrors": False,
                "no_color": True,
                "prefer_ffmpeg": True,
                "keepvideo": False,
                "extract_flat": False,
                "verbose": False
            }
            
            # Check if ffmpeg is available, if not, skip postprocessing
            if not shutil.which('ffmpeg'):
                ydl_opts['postprocessors'] = []
                logger.warning("FFmpeg not found. Downloading without conversion.")
            
            def progress_hook(d):
                if d['status'] == 'downloading':
                    if 'total_bytes' in d and d['total_bytes']:
                        percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
                        speed = d.get('speed', 0)
                        speed_str = f"{speed / 1024 / 1024:.1f} MB/s" if speed else "N/A"
                        downloaded = f"{d['downloaded_bytes'] / 1024 / 1024:.1f}MB"
                        total_size = f"{d['total_bytes'] / 1024 / 1024:.1f}MB" if d.get('total_bytes') else "Unknown"
                        status_queue.put({
                            'track_number': track_info['track_number'],
                            'status': 'downloading',
                            'message': f'{percent:.1f}%',
                            'details': f'{downloaded} of {total_size} @ {speed_str}',
                            'percent': percent
                        })
                elif d['status'] == 'finished':
                    status_queue.put({
                        'track_number': track_info['track_number'],
                        'status': 'converting',
                        'message': 'Converting...', 
                        'percent': 75
                    })
            
            ydl_opts['progress_hooks'] = [progress_hook]
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Try multiple search queries for better results
                search_queries = [
                    f"{track_info['title']} {', '.join(track_info['artists'])} official audio",
                    f"{track_info['title']} {', '.join(track_info['artists'])} official",
                    f"{track_info['title']} {', '.join(track_info['artists'])}",
                    f"{track_info['title']} by {', '.join(track_info['artists'])}"
                ]
                
                video_url = None
                for query in search_queries:
                    try:
                        search_result = ydl.extract_info(f"ytsearch1:{query}", download=False)
                        if search_result and 'entries' in search_result and len(search_result['entries']) > 0:
                            video_info = search_result['entries'][0]
                            video_url = video_info.get('webpage_url')
                            if video_url:
                                break
                    except Exception as e:
                        logger.warning(f"Search query failed: {query}, error: {e}")
                        continue
                
                if not video_url:
                    status_queue.put({
                        'track_number': track_info['track_number'],
                        'status': 'not_found',
                        'message': 'Not found on YouTube',
                        'percent': 100
                    })
                    return {
                        "status": "not_found", 
                        "track": track_info['name'], 
                        "filename": filename
                    }
                
                status_queue.put({
                    'track_number': track_info['track_number'],
                    'status': 'downloading',
                    'message': 'Downloading...', 
                    'percent': 25
                })
                
                ydl.download([video_url])
                
                # Get file size after download
                file_path = os.path.join(output_folder, f"{filename}.mp3")
                
                if not os.path.exists(file_path):
                    status_queue.put({
                        'track_number': track_info['track_number'],
                        'status': 'error',
                        'message': 'File not found after download',
                        'percent': 100
                    })
                    return {
                        "status": "error", 
                        "track": track_info['name'], 
                        "filename": filename, 
                        "error": "File not found after download"
                    }
                
                # Add metadata to the downloaded file
                try:
                    self.add_metadata(file_path, track_info)
                except Exception as e:
                    logger.error(f"Error adding metadata: {e}")
                
                file_size = os.path.getsize(file_path)
                file_size_mb = f"{file_size / 1024 / 1024:.2f} MB"
                
                status_queue.put({
                    'track_number': track_info['track_number'],
                    'status': 'complete',
                    'message': 'Complete',
                    'details': f'Size: {file_size_mb}',
                    'percent': 100
                })
                
                return {
                    "status": "success", 
                    "track": track_info['name'], 
                    "filename": f"{filename}.mp3",
                    "size": file_size_mb
                }
                    
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Download error: {error_msg}")
            
            if any(err in error_msg.lower() for err in ["403", "429", "connection", "timeout"]) and retry_count < 3:
                wait_time = (2 ** retry_count) + random.uniform(0, 1)
                status_queue.put({
                    'track_number': track_info['track_number'],
                    'status': 'retrying',
                    'message': f'Retry {retry_count+1}/3',
                    'percent': 0
                })
                time.sleep(wait_time)
                return self.download_track(track_info, output_folder, status_queue, retry_count + 1, quality)
            
            status_queue.put({
                'track_number': track_info['track_number'],
                'status': 'error',
                'message': f'Error: {error_msg[:50]}...',
                'percent': 100
            })
            return {
                "status": "error", 
                "track": track_info['name'], 
                "filename": filename, 
                "error": error_msg
            }

# Global download manager
download_manager = {
    'active': False,
    'progress': {},
    'status_queue': queue.Queue(),
    'results': {},
    'files': {}  # Track files for each download session
}

def download_worker(tracks, download_id, quality='192', content_name='download'):
    """Background worker for downloading tracks"""
    global download_manager
    
    download_manager['active'] = True
    download_manager['progress'][download_id] = {}
    download_manager['results'][download_id] = {
        'success': 0, 'failed': 0, 'not_found': 0, 'cancelled': 0
    }
    download_manager['files'][download_id] = []  # Store filenames for this session
    
    # Create a folder for this download session
    session_folder = os.path.join(app.config['DOWNLOAD_FOLDER'], str(download_id))
    os.makedirs(session_folder, exist_ok=True)
    
    # Initialize progress
    for track in tracks:
        download_manager['progress'][download_id][track['track_number']] = {
            'status': 'waiting',
            'message': 'Waiting...', 
            'details': '',
            'percent': 0
        }
    
    downloader = SpotifyDownloader()
    status_queue = download_manager['status_queue']
    
    def update_progress_from_queue():
        """Continuously update progress from the queue"""
        while download_manager['active']:
            try:
                update = status_queue.get(timeout=1)
                track_num = update['track_number']
                if download_id in download_manager['progress'] and track_num in download_manager['progress'][download_id]:
                    download_manager['progress'][download_id][track_num].update(update)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in progress updater: {e}")
                break
    
    # Start progress updater thread
    progress_thread = threading.Thread(target=update_progress_from_queue)
    progress_thread.daemon = True
    progress_thread.start()
    
    def download_with_tracking(track_info):
        result = downloader.download_track(track_info, session_folder, status_queue, 0, quality)
        
        # Update results and track files
        if result["status"] == "success":
            download_manager['results'][download_id]['success'] += 1
            download_manager['files'][download_id].append(result["filename"])
        elif result["status"] == "not_found":
            download_manager['results'][download_id]['not_found'] += 1
        elif result["status"] == "cancelled":
            download_manager['results'][download_id]['cancelled'] += 1
        else:
            download_manager['results'][download_id]['failed'] += 1
            
        return result
    
    # Start downloading with more workers for faster downloads
    with ThreadPoolExecutor(max_workers=min(4, len(tracks))) as executor:
        futures = [executor.submit(download_with_tracking, track) for track in tracks]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in download: {e}")

    # Create a zip archive of the downloaded files
    if download_manager['results'][download_id]['success'] > 0:
        zip_filename_base = downloader.sanitize_filename(content_name)
        zip_filepath = os.path.join(app.config['DOWNLOAD_FOLDER'], f"{zip_filename_base}_{download_id}")
        
        try:
            shutil.make_archive(zip_filepath, 'zip', session_folder)
            zip_filename = f"{zip_filename_base}_{download_id}.zip"
            download_manager['results'][download_id]['zip_filename'] = zip_filename
        except Exception as e:
            logger.error(f"Error creating zip archive: {e}")

    download_manager['active'] = False

@app.route('/api/analyze', methods=['POST'])
@password_required
def analyze_url():
    """Analyze Spotify URL and return track list"""
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    downloader = SpotifyDownloader()
    result = downloader.get_content_tracks(url)
    
    if isinstance(result[0], str):
        return jsonify({'error': result[0]}), 400
    
    tracks, content_info = result
    
    # Prepare response
    response = {
        'content_info': {
            'name': content_info.get('name', 'Unknown'),
            'type': content_info.get('type', 'unknown'),
            'total_tracks': len(tracks),
            'image': content_info.get('images', [{}])[0].get('url') if content_info.get('images') else None
        },
        'tracks': tracks
    }
    
    if 'owner' in content_info:
        response['content_info']['owner'] = content_info['owner'].get('display_name', 'Unknown')
    elif 'artists' in content_info:
        response['content_info']['artists'] = [artist.get('name', 'Unknown') for artist in content_info['artists']]
    
    return jsonify(response)

@app.route('/api/download', methods=['POST'])
@password_required
def start_download():
    """Start download process"""
    global download_manager
    
    if download_manager['active']:
        return jsonify({'error': 'Download already in progress'}), 400
    
    data = request.get_json()
    tracks = data.get('tracks', [])
    quality = data.get('quality', '192')
    
    if not tracks:
        return jsonify({'error': 'No tracks selected'}), 400
    
    download_id = int(time.time() * 1000)  # Unique ID for this download
    
    # Start download in background thread
    thread = threading.Thread(target=download_worker, args=(tracks, download_id, quality, "download"))
    thread.daemon = True
    thread.start()
    
    return jsonify({'download_id': download_id, 'message': 'Download started'})

@app.route('/api/progress')
@password_required
def get_progress():
    """Get download progress updates"""
    global download_manager
    
    # Get latest status updates
    updates = []
    while not download_manager['status_queue'].empty():
        try:
            update = download_manager['status_queue'].get_nowait()
            updates.append(update)
        except queue.Empty:
            break
        except Exception as e:
            logger.error(f"Error getting status update: {e}")
    
    # Get current progress for all active downloads
    progress = {}
    for download_id, track_progress in download_manager['progress'].items():
        progress[download_id] = track_progress
    
    return jsonify({
        'active': download_manager['active'],
        'updates': updates,
        'progress': progress,
        'results': download_manager['results']
    })

@app.route('/api/status')
@password_required
def get_status():
    """Get overall download status"""
    global download_manager
    return jsonify({
        'active': download_manager['active'],
        'progress': download_manager['progress'],
        'results': download_manager['results']
    })

@app.route('/callback')
def callback():
    """Spotify OAuth callback"""
    try:
        # Get the authorization code from the query parameters
        code = request.args.get('code')
        
        if not code:
            return "Authentication failed: No authorization code received", 400
        
        # Create the auth manager to exchange the code for a token
        auth_manager = SpotifyOAuth(
            client_id=os.environ.get('SPOTIFY_CLIENT_ID'),
            client_secret=os.environ.get('SPOTIFY_CLIENT_SECRET'),
            redirect_uri=os.environ.get('SPOTIFY_REDIRECT_URI', 'http://localhost:5005/callback'),
            scope=SPOTIFY_SCOPE,
            cache_path=os.path.join(CACHE_FOLDER, '.spotify_cache')
        )
        
        # Exchange the code for an access token
        token_info = auth_manager.get_access_token(code)
        
        if not token_info:
            return "Authentication failed: Could not get access token", 400
        
        # Store Spotify authentication status in session
        session['spotify_authenticated'] = True
        
        # Redirect to the main page
        return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return f"Authentication error: {str(e)}", 400

@app.route('/download_zip/<path:filename>')
@password_required
def download_zip_file(filename):
    """Serve the zip file"""
    file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename
    )

@app.route('/download_zip/<download_id>')
@password_required
def download_zip(download_id):
    """Serve the zip file for a given download ID."""
    global download_manager
    result = download_manager['results'].get(int(download_id))
    if not result or 'zip_filename' not in result:
        return "Download not found or not complete.", 404
    
    zip_filename = result['zip_filename']
    file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], zip_filename)
    
    if not os.path.exists(file_path):
        return "File not found", 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=zip_filename
    )

@app.route('/downloads/<download_id>/<path:filename>')
@password_required
def download_file(download_id, filename):
    """Serve downloaded files from a specific session"""
    file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], download_id, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename
    )

@app.route('/api/downloads')
@password_required
def list_downloads():
    """List all downloaded files"""
    files = []
    download_folder = app.config['DOWNLOAD_FOLDER']
    
    try:
        # List all session folders
        for session_id in os.listdir(download_folder):
            session_path = os.path.join(download_folder, session_id)
            if os.path.isdir(session_path):
                for file in os.listdir(session_path):
                    if file.endswith('.mp3'):
                        file_path = os.path.join(session_path, file)
                        file_size = os.path.getsize(file_path)
                        files.append({
                            'name': file,
                            'session_id': session_id,
                            'size': file_size,
                            'size_formatted': f"{file_size / 1024 / 1024:.2f} MB",
                            'date': time.ctime(os.path.getctime(file_path))
                        })
        
        # Sort by date (newest first)
        files.sort(key=lambda x: os.path.getctime(os.path.join(download_folder, x['session_id'], x['name'])), reverse=True)
        
    except Exception as e:
        logger.error(f"Error listing downloads: {e}")
    
    return jsonify(files)

@app.route('/api/my_playlists')
@password_required
def get_my_playlists():
    """Get user's playlists from Spotify"""
    try:
        downloader = SpotifyDownloader()
        if not downloader.sp:
            return jsonify({'error': 'Not authenticated with Spotify'}), 401
            
        playlists = []
        results = downloader.sp.current_user_playlists()
        
        while results:
            for playlist in results['items']:
                playlists.append({
                    'name': playlist['name'],
                    'id': playlist['id'],
                    'tracks': playlist['tracks']['total'],
                    'owner': playlist['owner']['display_name'],
                    'image': playlist['images'][0]['url'] if playlist['images'] else None
                })
            
            if results['next']:
                results = downloader.sp.next(results)
            else:
                break
                
        return jsonify(playlists)
    except Exception as e:
        logger.error(f"Error getting playlists: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/my_saved_tracks')
@password_required
def get_my_saved_tracks():
    """Get user's saved tracks from Spotify"""
    try:
        downloader = SpotifyDownloader()
        if not downloader.sp:
            return jsonify({'error': 'Not authenticated with Spotify'}), 401
            
        tracks = []
        results = downloader.sp.current_user_saved_tracks()
        
        while results:
            for item in results['items']:
                track = item['track']
                if track and track['name'] and not track['is_local']:
                    track_name = track['name']
                    artists = ", ".join([artist['name'] for artist in track['artists']])
                    album_name = track['album']['name'] if track['album'] else "Unknown Album"
                    
                    safe_filename = downloader.sanitize_filename(f"{track_name} - {artists}")
                    
                    track_info = {
                        'name': f"{track_name} - {artists}",
                        'title': track_name,
                        'artists': artists,
                        'album': album_name,
                        'filename': safe_filename,
                        'search_query': f"{track_name} {artists} official audio",
                        'track_number': len(tracks) + 1,
                        'duration_ms': track.get('duration_ms', 0),
                        'spotify_id': track['id'],
                        'type': 'track',
                        'cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                    }
                    
                    tracks.append(track_info)
            
            if results['next']:
                results = downloader.sp.next(results)
            else:
                break
                
        return jsonify(tracks)
    except Exception as e:
        logger.error(f"Error getting saved tracks: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
@password_required
def search_spotify():
    """Search Spotify for tracks, albums, artists, playlists"""
    query = request.args.get('q', '')
    limit = request.args.get('limit', 10, type=int)
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
        
    try:
        downloader = SpotifyDownloader()
        if not downloader.sp:
            return jsonify({'error': 'Not authenticated with Spotify'}), 401
            
        # Search for multiple types at once
        results = downloader.sp.search(
            q=query, 
            type='track,album,playlist,artist', 
            limit=limit
        )
        
        # Format the response to make it easier to use on the frontend
        formatted_results = {
            'tracks': [],
            'albums': [],
            'playlists': [],
            'artists': []
        }
        
        # Process tracks
        if 'tracks' in results and 'items' in results['tracks']:
            for track in results['tracks']['items']:
                if track and track.get('name') and not track.get('is_local', False):
                    # Safely get album name and image
                    album_name = track['album']['name'] if track.get('album') else 'Unknown Album'
                    album_image = track['album']['images'][0]['url'] if track.get('album') and track['album'].get('images') else None
                    
                    formatted_results['tracks'].append({
                        'name': track['name'],
                        'artists': [artist['name'] for artist in track['artists']] if track.get('artists') else [],
                        'album': album_name,
                        'image': album_image,
                        'url': track['external_urls']['spotify'] if track.get('external_urls') else '',
                        'duration_ms': track.get('duration_ms', 0),
                        'type': 'track'
                    })
        
        # Process albums
        if 'albums' in results and 'items' in results['albums']:
            for album in results['albums']['items']:
                if album and album.get('name'):
                    formatted_results['albums'].append({
                        'name': album['name'],
                        'artists': [artist['name'] for artist in album['artists']] if album.get('artists') else [],
                        'image': album['images'][0]['url'] if album.get('images') else None,
                        'url': album['external_urls']['spotify'] if album.get('external_urls') else '',
                        'release_date': album.get('release_date', ''),
                        'type': 'album'
                    })
        
        # Process playlists
        if 'playlists' in results and 'items' in results['playlists']:
            for playlist in results['playlists']['items']:
                if playlist and playlist.get('name'):
                    formatted_results['playlists'].append({
                        'name': playlist['name'],
                        'owner': playlist['owner']['display_name'] if playlist.get('owner') else 'Unknown',
                        'image': playlist['images'][0]['url'] if playlist.get('images') else None,
                        'url': playlist['external_urls']['spotify'] if playlist.get('external_urls') else '',
                        'tracks_count': playlist['tracks']['total'] if playlist.get('tracks') else 0,
                        'type': 'playlist'
                    })
        
        # Process artists
        if 'artists' in results and 'items' in results['artists']:
            for artist in results['artists']['items']:
                if artist and artist.get('name'):
                    formatted_results['artists'].append({
                        'name': artist['name'],
                        'image': artist['images'][0]['url'] if artist.get('images') else None,
                        'url': artist['external_urls']['spotify'] if artist.get('external_urls') else '',
                        'id': artist['id'],
                        'type': 'artist'
                    })
                
        return jsonify(formatted_results)
    except Exception as e:
        logger.error(f"Error searching Spotify: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/artist/<artist_id>/top-tracks')
@password_required
def get_artist_top_tracks(artist_id):
    """Get an artist's top tracks"""
    try:
        downloader = SpotifyDownloader()
        if not downloader.sp:
            return jsonify({'error': 'Not authenticated with Spotify'}), 401
        
        results = downloader.sp.artist_top_tracks(artist_id)
        
        tracks = []
        for track in results['tracks']:
            if track and track.get('name') and not track.get('is_local', False):
                album_name = track['album']['name'] if track.get('album') else 'Unknown Album'
                album_image = track['album']['images'][0]['url'] if track.get('album') and track['album'].get('images') else None
                
                tracks.append({
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']] if track.get('artists') else [],
                    'album': album_name,
                    'image': album_image,
                    'url': track['external_urls']['spotify'] if track.get('external_urls') else '',
                    'duration_ms': track.get('duration_ms', 0),
                    'type': 'track'
                })
        return jsonify(tracks)
    except Exception as e:
        logger.error(f"Error getting artist top tracks: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/retry-download', methods=['POST'])
@password_required
def retry_download():
    """Retry a failed download"""
    data = request.get_json()
    track = data.get('track')
    download_id = data.get('download_id')
    
    if not track or not download_id:
        return jsonify({'error': 'Missing track or download_id'}), 400
    
    # Add the track back to the download queue
    if download_id in download_manager['progress']:
        download_manager['progress'][download_id][track['track_number']] = {
            'status': 'waiting',
            'message': 'Waiting to retry...', 
            'details': '',
            'percent': 0
        }
    
    return jsonify({'message': 'Download queued for retry'})

if __name__ == '__main__':
    # Set these environment variables or replace with your actual credentials
    if not os.environ.get('SPOTIFY_CLIENT_ID'):
        os.environ['SPOTIFY_CLIENT_ID'] = 'your_spotify_client_id'
    if not os.environ.get('SPOTIFY_CLIENT_SECRET'):
        os.environ['SPOTIFY_CLIENT_SECRET'] = 'your_spotify_client_secret'
    if not os.environ.get('SPOTIFY_REDIRECT_URI'):
        os.environ['SPOTIFY_REDIRECT_URI'] = 'http://localhost:5005/callback'
    
    # Change the port to 5005
    try:
        app.run(debug=True, host='0.0.0.0', port=5005)
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print("Port 5005 is also in use. Trying port 5006...")
            app.run(debug=True, host='0.0.0.0', port=5006)
        else:
            raise