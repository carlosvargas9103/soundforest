import csv
import os
import shutil
import simpleaudio as sa
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

CATEGORIES = {
    1: "engine",
    2: "machinery-impact",
    3: "non-machinery-impact",
    4: "powered-saw",
    5: "alert-signal",
    6: "music",
    7: "human-voice",
    8: "dog",
}

# =======================
# PREP
# =======================
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

for name in list(CATEGORIES.values()) + ["none"]:
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, name), exist_ok=True)

moved_counter = defaultdict(int)
stayed_counter = 0
terminate_program = False

# =======================
# AUDIO PLAYBACK
# =======================
def play_audio(path):
    try:
        wave = sa.WaveObject.from_wave_file(path)
        play = wave.play()
        play.wait_done()
    except Exception as e:
        print(f"[AUDIO ERROR] {e}")

# =======================
# SUBCATEGORY PROMPT
# =======================
def prompt_subcategory():
    sub = input("Sub-category (press Enter for 'default'): ").strip()
    return sub if sub else "default"

# =======================
# USER PROMPT
# =======================
def prompt_user(audio_file, active_labels):
    print("\n==============================")
    print(f"File: {audio_file}")
    print("Detected labels:", active_labels)
    print("\nChoose action:")

    for k, v in CATEGORIES.items():
        print(f"{k}: move to '{v}' (with sub-category)")

    print("9: create combined folder")
    print("0: leave file unmoved")
    print("100: finish program and report")

    while True:
        try:
            choice = int(input("Your choice: ").strip())
            if (choice in list(CATEGORIES.keys()) + [0, 9]) or (choice == 100):
                return choice
        except ValueError:
            pass
        print("Invalid input. Try again.")

# =======================
# PROCESS CSV
# =======================
with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        audio_file = row["audio_filename"]
        audio_path = os.path.join(AUDIO_BASE_DIR, audio_file)

        active_labels = [
            col.replace("_presence", "").split("_", 1)[1]
            for col in TARGET_COLUMNS
            if int(row[col]) == 1
        ]

        if not os.path.exists(audio_path):
            print(f"[MISSING FILE] {audio_file}")
            stayed_counter += 1
            continue

        # =======================
        # NO LABEL → AUTO MOVE
        # =======================
        if len(active_labels) == 0:
            target_path = os.path.join(
                OUTPUT_BASE_DIR,
                "none",
                os.path.basename(audio_file)
            )
            shutil.move(audio_path, target_path)
            moved_counter["none"] += 1
            print(f"[AUTO] {audio_file} → none/")
            continue

        # Skip uniquely labeled files (already handled elsewhere)
        if len(active_labels) == 1:
            continue

        # =======================
        # INTERACTIVE CASE
        # =======================
        play_audio(audio_path)
        choice = prompt_user(audio_file, active_labels)

        if choice == 100:
            print("\n[TERMINATE] Finishing program and reporting results.")
            terminate_program = True
            break

        if choice == 0:
            stayed_counter += 1
            print("[LEFT] File not moved")

        elif choice in CATEGORIES:
            main_cat = CATEGORIES[choice]
            sub_cat = prompt_subcategory()

            target_dir = os.path.join(
                OUTPUT_BASE_DIR,
                main_cat,
                sub_cat
            )
            os.makedirs(target_dir, exist_ok=True)

            target_path = os.path.join(target_dir, os.path.basename(audio_file))
            shutil.move(audio_path, target_path)

            moved_counter[f"{main_cat}/{sub_cat}"] += 1
            print(f"[MOVED] → {main_cat}/{sub_cat}")

        elif choice == 9:
            folder_name = "+".join(sorted(active_labels))
            target_dir = os.path.join(OUTPUT_BASE_DIR, folder_name)
            os.makedirs(target_dir, exist_ok=True)

            target_path = os.path.join(target_dir, os.path.basename(audio_file))
            shutil.move(audio_path, target_path)

            moved_counter[folder_name] += 1
            print(f"[MOVED] → {folder_name}")

        if terminate_program:
            break

# =======================
# SUMMARY
# =======================
print("\n===== FINAL SUMMARY =====")

for k, v in moved_counter.items():
    print(f"{k}: {v}")

print(f"Stayed (not moved): {stayed_counter}")
print(f"Total moved: {sum(moved_counter.values())}")