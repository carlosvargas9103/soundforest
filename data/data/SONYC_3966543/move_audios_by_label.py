import csv
import os
import shutil
from collections import defaultdict

# =======================
# CONFIG
# =======================
CSV_FILE = "annotations.csv"
AUDIO_BASE_DIR = "./audio/"          # base directory where audio files are stored
OUTPUT_BASE_DIR = "./"    # where folders will be created

TARGET_COLUMNS = [
    "1_engine_presence",
    "2_machinery-impact_presence",
    "3_non-machinery-impact_presence",
    "4_powered-saw_presence",
    "5_alert-signal_presence",
    "6_music_presence",
    "7_human-voice_presence",
    "8_dog_presence",
]

# Clean names (remove *_presence and numeric prefix)
CLEAN_NAMES = {
    "1_engine_presence": "engine",
    "2_machinery-impact_presence": "machinery-impact",
    "3_non-machinery-impact_presence": "non-machinery-impact",
    "4_powered-saw_presence": "powered-saw",
    "5_alert-signal_presence": "alert-signal",
    "6_music_presence": "music",
    "7_human-voice_presence": "human-voice",
    "8_dog_presence": "dog",
}

# =======================
# PREP
# =======================
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

for folder in CLEAN_NAMES.values():
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, folder), exist_ok=True)

moved_counter = defaultdict(int)
stayed_counter = 0

# =======================
# PROCESS CSV
# =======================
with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        audio_file = row["audio_filename"]
        audio_path = os.path.join(AUDIO_BASE_DIR, audio_file)

        # Read one-hot values
        active_labels = [
            col for col in TARGET_COLUMNS
            if int(row[col]) == 1
        ]

        # Case 1: exactly one active label
        if len(active_labels) == 1:
            col = active_labels[0]
            target_folder = CLEAN_NAMES[col]
            target_path = os.path.join(OUTPUT_BASE_DIR, target_folder, os.path.basename(audio_file))

            if not os.path.exists(audio_path):
                # print(f"[MISSING FILE] {audio_file}")
                stayed_counter += 1
                continue

            shutil.move(audio_path, target_path)
            moved_counter[target_folder] += 1

        # Case 2: zero or multiple active labels
        else:
            # print(
            #     f"[NOT UNIQUE] {audio_file} -> "
            #     f"{[CLEAN_NAMES[c] for c in active_labels]}"
            # )
            stayed_counter += 1

# =======================
# SUMMARY
# =======================
print("\n===== SUMMARY =====")
total_moved = 0

for label, count in moved_counter.items():
    print(f"{label}: {count}")
    total_moved += count

print(f"Stayed (not moved): {stayed_counter}")
print(f"Total moved: {total_moved}")

'''
===== SUMMARY =====
machinery-impact: 1767
engine: 5106
non-machinery-impact: 675
alert-signal: 2213
human-voice: 3882
music: 777
powered-saw: 1099
dog: 643
Stayed (not moved): 45860
Total moved: 16162
'''

