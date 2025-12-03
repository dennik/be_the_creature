# soundplayer.py
# Version: 1.5
# Change Log:
# v1.5 - Added os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' at top to suppress pygame welcome message (per user request for console cleanup). Retained v1.4 debug comment-outs and v1.3 CLI; no changes to logic. Certainty: 100% (direct env var from pygame docs; hides "Hello from the pygame community" prompt).
# v1.4 - Commented out debug prints in play_on_source (crossfading/playing messages) and __init__/quit to reduce I/O in potentially frequent calls (e.g., sound triggers in preview). Retained error prints for diagnostics; no changes to logic. Certainty: 85% (prints not in tight loops but conservative for performance).
# v1.3 - Added CLI support: Included if __name__ == "__main__" block to parse sys.argv[1] as MP3 file path, instantiate SoundPlayer, play on 'voice' source (default for simple clips), and block with while loop checking is_source_busy until playback finishes. This mimics the user's working script, preventing immediate exit. Retained v1.2 pydub MP3 handling; no changes to class logic. Certainty: 95% (adds explicit entry point for CLI use; tested conceptually via similar patterns in Pygame docs).

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Suppress pygame welcome message

import pygame  # For mixer and sound playback (handles multiple channels, fadein/fadeout)
import os  # For file path validation
import io  # For in-memory buffer (used in MP3 loading)
from pydub import AudioSegment  # For loading and converting MP3 to WAV buffer (enables MP3 support)
import sys  # For CLI argument parsing (v1.3)
import time  # For sleep in blocking loop (v1.3)

class SoundPlayer:
    """
    Manages audio playback with multiple sources (e.g., music, effects, voice) with crossfading support.
    Each source uses two pygame channels for crossfading: when a new sound starts on a busy source,
    the old sound fades out while the new one fades in over the specified crossfade_ms.
    Initialize pygame.mixer with enough channels (2 per source).
    Usage:
        player = SoundPlayer(sources=['music', 'effects', 'voice'])
        player.play_on_source('music', 'background.mp3', crossfade_ms=500)
    Modularity suggestion: Easily extend by adding more sources in init (auto-allocates channels).
    If needed, add volume control per source or global mute.
    v1.2: Added MP3 handling via pydub for compatibility, as pygame.Sound does not natively support MP3.
    """

    def __init__(self, sources=['music', 'effects', 'voice'], channels_per_source=2, default_crossfade_ms=500):
        """
        Initializes the sound player with specified sources.
        Args:
            sources (list[str]): List of source names (e.g., ['music', 'effects', 'voice']).
            channels_per_source (int): Channels allocated per source (2 for crossfade support).
            default_crossfade_ms (int): Default crossfade duration in milliseconds.
        """
        # Initialize pygame mixer if not already (frequency=44100, size=-16, channels=2, buffer=512 default)
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        # Calculate total channels needed and set (extra for safety)
        total_channels = len(sources) * channels_per_source + 2  # +2 buffer
        pygame.mixer.set_num_channels(total_channels)

        # Assign channel pairs to each source (dict: source -> [ch1, ch2, current])
        self.source_channels = {}
        channel_id = 0
        for source in sources:
            ch1 = pygame.mixer.Channel(channel_id)
            ch2 = pygame.mixer.Channel(channel_id + 1)
            self.source_channels[source] = {
                'ch1': ch1,
                'ch2': ch2,
                'current': ch1,  # Starts with ch1 as current
            }
            channel_id += channels_per_source

        self.default_crossfade_ms = default_crossfade_ms
        # Print init info for debug
        # print(f"SoundPlayer initialized with {len(sources)} sources, {total_channels} channels.", flush=True)

    def _load_sound(self, file_path):
        """
        Internal helper: Loads sound from file, handling MP3 via pydub conversion to WAV buffer.
        Returns pygame.mixer.Sound object.
        """
        try:
            if file_path.lower().endswith('.mp3'):
                # Load MP3 with pydub and convert to in-memory WAV
                segment = AudioSegment.from_mp3(file_path)
                buf = io.BytesIO()
                segment.export(buf, format="wav")
                buf.seek(0)
                return pygame.mixer.Sound(buf)
            else:
                # Direct load for other formats (e.g., WAV, OGG)
                return pygame.mixer.Sound(file_path)
        except Exception as e:
            print(f"Error loading sound '{file_path}': {e}", flush=True)
            raise  # Re-raise for caller handling

    def play_on_source(self, source, file_path, loops=0, maxtime=0, fadein_ms=0, crossfade_ms=None):
        """
        Plays a sound file on the specified source.
        If the source is busy and crossfade_ms > 0, crossfades by fading out the old sound
        and fading in the new one using the alternate channel.
        Args:
            source (str): Source name (e.g., 'music').
            file_path (str): Path to sound file (WAV, MP3, OGG).
            loops (int): Number of loops (-1 infinite, default 0).
            maxtime (int): Max play time in ms (0 unlimited).
            fadein_ms (int): Fade-in time for new sound if no crossfade (default 0).
            crossfade_ms (int or None): Crossfade duration if source busy (default to self.default_crossfade_ms).
        Returns:
            bool: True if played successfully, False on error.
        """
        if source not in self.source_channels:
            print(f"Error: Unknown source '{source}'.", flush=True)
            return False

        if not os.path.exists(file_path):
            print(f"Error: File not found '{file_path}'.", flush=True)
            return False

        try:
            sound = self._load_sound(file_path)  # Use helper to handle MP3
        except Exception:
            return False

        source_data = self.source_channels[source]
        current_ch = source_data['current']
        crossfade_ms = crossfade_ms if crossfade_ms is not None else self.default_crossfade_ms

        if current_ch.get_busy() and crossfade_ms > 0:
            # Crossfade: Use alternate channel for new sound
            old_ch = current_ch
            new_ch = source_data['ch2'] if old_ch == source_data['ch1'] else source_data['ch1']

            # Start new sound on new channel with fade-in
            new_ch.play(sound, loops=loops, maxtime=maxtime, fade_ms=crossfade_ms)

            # Fade out old channel
            old_ch.fadeout(crossfade_ms)

            # Update current to new channel
            source_data['current'] = new_ch
            # print(f"Crossfading on '{source}' from old to '{file_path}' over {crossfade_ms}ms.", flush=True)
        else:
            # Normal play: On current channel (stops old if playing)
            current_ch.play(sound, loops=loops, maxtime=maxtime, fade_ms=fadein_ms)
            # print(f"Playing '{file_path}' on '{source}' with fade-in {fadein_ms}ms.", flush=True)

        return True

    def stop_source(self, source, fadeout_ms=0):
        """
        Stops playback on the specified source.
        Args:
            source (str): Source name.
            fadeout_ms (int): Fade-out time (default 0 for immediate stop).
        """
        if source not in self.source_channels:
            return

        current_ch = self.source_channels[source]['current']
        if fadeout_ms > 0:
            current_ch.fadeout(fadeout_ms)
        else:
            current_ch.stop()

    def is_source_busy(self, source):
        """
        Checks if the source is currently playing.
        Args:
            source (str): Source name.
        Returns:
            bool: True if busy.
        """
        if source not in self.source_channels:
            return False
        return self.source_channels[source]['current'].get_busy()

    def quit(self):
        """
        Cleans up: Stops all sounds and quits mixer.
        Call on app shutdown.
        """
        pygame.mixer.stop()
        pygame.mixer.quit()
        # print("SoundPlayer quit.", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sound_player.py <path_to_mp3_file>")
        sys.exit(1)

    mp3_file = sys.argv[1]
    player = SoundPlayer()  # Instantiate with default sources
    if player.play_on_source('voice', mp3_file):  # Play on 'voice' source (suitable for clips)
        while player.is_source_busy('voice'):  # Block until playback finishes
            time.sleep(0.1)  # Light sleep to avoid CPU spin
    player.quit()