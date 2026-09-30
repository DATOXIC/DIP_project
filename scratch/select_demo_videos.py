"""
Script tạo manifest video demo ASL Citizen cho từng từ vựng.
Chọn video có duration ~2-3s (không quá nhanh, không quá chậm).
"""
import cv2
import os
import glob
import json

BASE = r'C:\DIP\DIP_project\ASL_Citizen_videos'
OUT = r'C:\DIP\DIP_project\asl_demo_videos.json'

def get_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return round(total / fps, 3), round(fps, 1)

result = {}
words = sorted(os.listdir(BASE))
for word in words:
    folder = os.path.join(BASE, word)
    if not os.path.isdir(folder):
        continue
    files = sorted(glob.glob(os.path.join(folder, '*.mp4')))
    if not files:
        result[word] = None
        print(f"{word:14s}: NO VIDEO")
        continue

    candidates = []
    for f in files:
        dur, fps = get_duration(f)
        candidates.append({'path': f, 'duration': dur, 'fps': fps})

    # Prefer range 1.5-4.0s (comfortable demo pace)
    in_range = [c for c in candidates if 1.5 <= c['duration'] <= 4.0]
    if not in_range:
        in_range = sorted(candidates, key=lambda x: abs(x['duration'] - 2.5))

    # Among valid, pick closest to 2.5s (comfortable speed)
    in_range.sort(key=lambda x: abs(x['duration'] - 2.5))
    chosen = in_range[0]

    # Store relative path for portability
    rel_path = os.path.relpath(chosen['path'], os.path.dirname(OUT))
    result[word] = {
        'path': rel_path.replace('\\', '/'),
        'abs_path': chosen['path'],
        'duration': chosen['duration'],
        'fps': chosen['fps'],
    }
    name = os.path.basename(chosen['path'])
    print(f"{word:14s}: {chosen['duration']:.2f}s | {chosen['fps']:.0f}fps | {name}")

print(f"\nTotal: {len(result)} words")
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"Saved -> {OUT}")
