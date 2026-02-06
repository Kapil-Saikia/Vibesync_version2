// ============================================================
// SHARED AUDIO PLAYER - Works across all pages
// ============================================================

// Global audio player instance
window.sharedAudioPlayer = {
    audio: null,
    currentSong: null,
    isPlaying: false,
    currentTime: 0,
    volume: 0.7,
    isMuted: false,
    currentSongIndex: -1,
    currentPlaylist: [],
    
    init() {
        if (!this.audio) {
            this.audio = new Audio();
            this.audio.volume = this.volume;
            
            // Load state from sessionStorage
            this.loadState();
            
            // Event listeners
            this.audio.addEventListener('timeupdate', () => {
                if (this.audio.duration) {
                    this.currentTime = this.audio.currentTime;
                    this.updateProgressUI();
                }
            });
            
            this.audio.addEventListener('ended', () => {
                this.nextSong();
            });
            
            this.audio.addEventListener('loadedmetadata', () => {
                this.updateTimeDisplay();
            });
            
            // Save state periodically
            setInterval(() => this.saveState(), 2000);
            window.addEventListener('beforeunload', () => this.saveState());
        }
    },
    
    loadState() {
        try {
            const saved = sessionStorage.getItem('sharedPlayerState');
            if (saved) {
                const state = JSON.parse(saved);
                if (state.song && state.isPlaying) {
                    // Restore playback state
                    this.currentSong = state.song;
                    this.currentTime = state.currentTime || 0;
                    this.isPlaying = state.isPlaying;
                    this.volume = state.volume || 0.7;
                    this.currentPlaylist = state.playlist || [];
                    this.currentSongIndex = state.songIndex || -1;
                    
                    // Restore audio
                    if (this.currentSong.audioUrl) {
                        this.audio.src = this.currentSong.audioUrl;
                        this.audio.currentTime = this.currentTime;
                        this.audio.volume = this.volume;
                        
                        if (this.isPlaying) {
                            this.audio.play().catch(() => {
                                // Auto-play prevented
                                this.isPlaying = false;
                            });
                        }
                    }
                    
                    // Update UI if player exists
                    this.updateUI();
                }
            }
        } catch (e) {
            console.error('Error loading player state:', e);
        }
    },
    
    saveState() {
        try {
            const state = {
                song: this.currentSong,
                currentTime: this.currentTime,
                isPlaying: this.isPlaying,
                volume: this.volume,
                playlist: this.currentPlaylist,
                songIndex: this.currentSongIndex
            };
            sessionStorage.setItem('sharedPlayerState', JSON.stringify(state));
        } catch (e) {
            console.error('Error saving player state:', e);
        }
    },
    
    playSong(song, playlist = []) {
        this.currentSong = song;
        this.currentPlaylist = playlist;
        
        // Find song index in playlist
        if (playlist.length > 0) {
            this.currentSongIndex = playlist.findIndex(s => s.id === song.id);
        } else {
            this.currentSongIndex = -1;
        }
        
        // Update UI
        this.updateUI();
        
        // Play audio – use ONLY the song's own URL so each track is unique
        const audioUrl = song.audioUrl || song.audio_url;
        if (!audioUrl) {
            console.warn('No audio URL available for song:', song);
            if (window.showNotification) {
                window.showNotification('⚠️ This song has no preview available');
            }
            this.isPlaying = false;
            this.updatePlayPauseButton();
            return;
        }

        this.audio.src = audioUrl;
        this.audio.play().then(() => {
            this.isPlaying = true;
            this.updatePlayPauseButton();
            this.saveState();
            
            // Track recently played
            this.trackRecentlyPlayed(song);
        }).catch(err => {
            console.error('Audio playback error:', err);
            this.isPlaying = false;
            this.updatePlayPauseButton();
        });
    },
    
    togglePlayPause() {
        if (!this.currentSong) return;
        
        if (this.isPlaying) {
            this.audio.pause();
            this.isPlaying = false;
        } else {
            this.audio.play();
            this.isPlaying = true;
        }
        
        this.updatePlayPauseButton();
        this.saveState();
    },
    
    nextSong() {
        if (this.currentPlaylist.length === 0) return;
        
        const nextIndex = (this.currentSongIndex + 1) % this.currentPlaylist.length;
        const nextSong = this.currentPlaylist[nextIndex];
        
        if (nextSong) {
            this.playSong(nextSong, this.currentPlaylist);
        }
    },
    
    previousSong() {
        if (this.currentPlaylist.length === 0) return;
        
        const prevIndex = this.currentSongIndex <= 0 
            ? this.currentPlaylist.length - 1 
            : this.currentSongIndex - 1;
        const prevSong = this.currentPlaylist[prevIndex];
        
        if (prevSong) {
            this.playSong(prevSong, this.currentPlaylist);
        }
    },
    
    setVolume(vol) {
        this.volume = Math.max(0, Math.min(1, vol));
        this.audio.volume = this.volume;
        this.saveState();
    },
    
    setProgress(percentage) {
        if (this.audio.duration) {
            this.audio.currentTime = (percentage / 100) * this.audio.duration;
            this.currentTime = this.audio.currentTime;
        }
    },
    
    trackRecentlyPlayed(song) {
        fetch('/api/recently-played', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                songId: song.id,
                title: song.title,
                artist: song.artist,
                coverUrl: song.img || song.coverUrl,
                audioUrl: song.audioUrl || song.audio_url || '',
                artistPhotoUrl: song.artistPhotoUrl || ''
            })
        }).catch(err => console.error('Error tracking recently played:', err));
    },
    
    updateUI() {
        if (!this.currentSong) return;
        
        // Update player UI elements if they exist
        const thumbnail = document.getElementById('playerThumbnail');
        const title = document.getElementById('playerTitle');
        const artist = document.getElementById('playerArtist');
        const player = document.getElementById('musicPlayer');
        
        if (thumbnail) thumbnail.src = this.currentSong.img || this.currentSong.coverUrl || '';
        if (title) title.textContent = this.currentSong.title || '';
        if (artist) artist.textContent = this.currentSong.artist || '';
        if (player) player.classList.add('active');
        
        // Update fullscreen player if exists
        const fullscreenArt = document.getElementById('fullscreenAlbumArt');
        const fullscreenTitle = document.getElementById('fullscreenSongTitle');
        const fullscreenArtist = document.getElementById('fullscreenSongArtist');
        
        if (fullscreenArt) fullscreenArt.src = this.currentSong.img || this.currentSong.coverUrl || '';
        if (fullscreenTitle) fullscreenTitle.textContent = this.currentSong.title || '';
        if (fullscreenArtist) fullscreenArtist.textContent = this.currentSong.artist || '';
        
        this.updatePlayPauseButton();
    },
    
    updatePlayPauseButton() {
        const btn = document.getElementById('playPauseBtn');
        const fullscreenBtn = document.getElementById('fullscreenPlayPause');
        
        if (btn) {
            btn.textContent = this.isPlaying ? '⏸' : '▶';
        }
        if (fullscreenBtn) {
            fullscreenBtn.textContent = this.isPlaying ? '⏸' : '▶';
        }
    },
    
    updateProgressUI() {
        if (!this.audio.duration) return;
        
        const progress = (this.currentTime / this.audio.duration) * 100;
        const fill = document.getElementById('progressFill');
        if (fill) fill.style.width = progress + '%';
        
        this.updateTimeDisplay();
    },
    
    updateTimeDisplay() {
        if (!this.audio.duration) return;
        
        const current = Math.floor(this.currentTime);
        const total = Math.floor(this.audio.duration);
        const currentMin = Math.floor(current / 60);
        const currentSec = current % 60;
        const totalMin = Math.floor(total / 60);
        const totalSec = total % 60;
        
        const currentTimeEl = document.getElementById('currentTime');
        const totalTimeEl = document.getElementById('totalTime');
        
        if (currentTimeEl) {
            currentTimeEl.textContent = `${currentMin}:${currentSec.toString().padStart(2, '0')}`;
        }
        if (totalTimeEl) {
            totalTimeEl.textContent = `${totalMin}:${totalSec.toString().padStart(2, '0')}`;
        }
    }
};

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.sharedAudioPlayer.init();
    });
} else {
    window.sharedAudioPlayer.init();
}

