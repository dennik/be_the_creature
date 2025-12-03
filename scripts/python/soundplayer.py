import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import PATHS
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import os
import io
from pydub import AudioSegment
import sys
import time

class SoundPlayer:
    def __init__(self, sources=['music', 'effects', 'voice'], channels_per_source=2, default_crossfade_ms=500):
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        total_channels = len(sources) * channels_per_source + 2
        pygame.mixer.set_num_channels(total_channels)

        self.source_channels = {}
        channel_id = 0
        for source in sources:
            ch1 = pygame.mixer.Channel(channel_id)
            ch2 = pygame.mixer.Channel(channel_id + 1)
            self.source_channels[source] = {
                'ch1': ch1,
                'ch2': ch2,
                'current': ch1,
            }
            channel_id += channels_per_source

        self.default_crossfade_ms = default_crossfade_ms

    def _load_sound(self, file_path):
        try:
            if file_path.lower().endswith('.mp3'):
                segment = AudioSegment.from_mp3(file_path)
                buf = io.BytesIO()
                segment.export(buf, format="wav")
                buf.seek(0)
                return pygame.mixer.Sound(buf)
            else:
                return pygame.mixer.Sound(file_path)
        except Exception as e:
            print(f"Error loading sound '{file_path}': {e}", flush=True)
            raise

    def play_on_source(self, source, file_path, loops=0, maxtime=0, fadein_ms=0, crossfade_ms=None):
        if source not in self.source_channels:
            print(f"Error: Unknown source '{source}'.", flush=True)
            return False

        if not os.path.exists(file_path):
            print(f"Error: File not found '{file_path}'.", flush=True)
            return False

        try:
            sound = self._load_sound(file_path)
        except Exception:
            return False

        source_data = self.source_channels[source]
        current_ch = source_data['current']
        crossfade_ms = crossfade_ms if crossfade_ms is not None else self.default_crossfade_ms

        if current_ch.get_busy() and crossfade_ms > 0:
            old_ch = current_ch
            new_ch = source_data['ch2'] if old_ch == source_data['ch1'] else source_data['ch1']

            new_ch.play(sound, loops=loops, maxtime=maxtime, fade_ms=crossfade_ms)

            old_ch.fadeout(crossfade_ms)

            source_data['current'] = new_ch
        else:
            current_ch.play(sound, loops=loops, maxtime=maxtime, fade_ms=fadein_ms)

        return True

    def stop_source(self, source, fadeout_ms=0):
        if source not in self.source_channels:
            return

        current_ch = self.source_channels[source]['current']
        if fadeout_ms > 0:
            current_ch.fadeout(fadeout_ms)
        else:
            current_ch.stop()

    def is_source_busy(self, source):
        if source not in self.source_channels:
            return False
        return self.source_channels[source]['current'].get_busy()

    def quit(self):
        pygame.mixer.stop()
        pygame.mixer.quit()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sound_player.py <path_to_mp3_file>")
        sys.exit(1)

    mp3_file = sys.argv[1]
    player = SoundPlayer()
    if player.play_on_source('voice', mp3_file):
        while player.is_source_busy('voice'):
            time.sleep(0.1)
    player.quit()