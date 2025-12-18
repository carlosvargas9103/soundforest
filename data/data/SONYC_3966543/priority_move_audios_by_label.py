import csv
import os
import shutil
from collections import defaultdict

# =======================
# CONFIG
# =======================
CSV_FILE = "annotations.csv"
AUDIO_BASE_DIR = "./audio"
OUTPUT_BASE_DIR = "./"

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

# Priority: smaller number = higher priority
PRIORITY_MAP = {
    "engine": 1,
    "machinery-impact": 2,
    "non-machinery-impact": 3,
    "powered-saw": 4,
    "alert-signal": 5,
    "music": 6,
    "human-voice": 7,
    "dog": 8,
}

# =======================
# PREP
# =======================
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

for folder in list(PRIORITY_MAP.keys()) + ["none"]:
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

        if not os.path.exists(audio_path):
            print(f"[MISSING FILE] {audio_file}")
            stayed_counter += 1
            continue

        active_labels = []

        for col in TARGET_COLUMNS:
            if int(row[col]) == 1:
                label = col.replace("_presence", "").split("_", 1)[1]
                active_labels.append(label)

        # =======================
        # NO LABEL → highest priority (0)
        # =======================
        if not active_labels:
            target_dir = os.path.join(OUTPUT_BASE_DIR, "none")
            target_path = os.path.join(target_dir, os.path.basename(audio_file))

            shutil.move(audio_path, target_path)
            moved_counter["none"] += 1

            print(f"[AUTO] {audio_file} → none/")
            continue

        # =======================
        # PRIORITY RESOLUTION
        # =======================
        chosen_label = min(
            active_labels,
            key=lambda lbl: PRIORITY_MAP[lbl]
        )

        target_dir = os.path.join(OUTPUT_BASE_DIR, chosen_label)
        target_path = os.path.join(target_dir, os.path.basename(audio_file))

        shutil.move(audio_path, target_path)
        moved_counter[chosen_label] += 1

        print(
            f"[AUTO-PRIORITY] {audio_file} → {chosen_label} "
            f"(priority {PRIORITY_MAP[chosen_label]})"
        )

# =======================
# SUMMARY
# =======================
print("\n===== FINAL SUMMARY =====")

for label, count in moved_counter.items():
    print(f"{label}: {count}")

print(f"Stayed (not moved): {stayed_counter}")
print(f"Total moved: {sum(moved_counter.values())}")
