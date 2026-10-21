import ast, glob, os, shutil, sys, re
ROOT = os.getcwd()
OUT = os.path.join(ROOT, "submission", "code")
core = """backup_utils camera_health check_volunteer_data collect_continuous collect_phrases dip dip_app dl_fusion dynamic_dtw_engine
dynamic_features fingerspell hand_pose_features handshape_fusion handshape_gate improved_dip_pipeline lead_trim
letter_landmark_classifier online_negatives people pipeline_common predict_sentence_webcam sentence_composer session_enroll
speech_service spotter_dl spotter_features spotter_gate train train_dynamic_dl tts_queue ui_render ui_side_panel web_app
pipeline collect_dynamic_templates camera_diag train_spotter eval_rule_spotter eval_negative_session
scoreboard eval_baselines eval_learning_curve eval_hybrid eval_open_set eval_citizen eval_confirm eval_adapt separate_volunteer_raw""".split()
allmods = {os.path.splitext(f)[0] for f in glob.glob("*.py")}
def imports(path):
    t = ast.parse(open(path, encoding="utf-8").read()); s = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names: s.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module: s.add(n.module.split(".")[0])
    return s & allmods
# closure
chosen, st = set(), list(core)
while st:
    m = st.pop()
    if m in chosen: continue
    chosen.add(m); st.extend(imports(m + ".py"))
extra = sorted(chosen - set(core))
print("them do phu thuoc:", extra)
files = [m + ".py" for m in sorted(chosen)]
data = """webcam_templates.json demo_templates.json dynamic_templates.json negative_strokes.json phrases_plan.json asl_demo_videos.json
hand_landmarker.task blaze_face_short_range.tflite spotter_model.pth letter_landmark_model.pt letter_landmark_model_image.pt
handshape_gate_table.json best_gesture_model_none.pth requirements.txt dip.bat HUONG_DAN_SU_DUNG.md PROJECT.md KIEM_TRA_TRUC_TIEP.md
ASL_ALEX_2.0/sign_weights_asllex.json vsl_dataset/models/best_model_v2.pth vsl_dataset/models/best_model_v3.pth
asl_citizen_templates_24d.json docs/DE_CUONG_DO_AN_DIP_ASL.md docs/HUONG_DAN_DIP.md AGENTS.md""".split()
dirs = ["web", "audio_cache", "configs", "profiles"]
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)
missing = []
def cp(rel):
    src = os.path.join(ROOT, rel)
    if not os.path.exists(src): missing.append(rel); return
    dst = os.path.join(OUT, rel); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst)
for f in files + data: cp(f)
for d in dirs:
    for dp, dn, fn in os.walk(d):
        if "__pycache__" in dp: continue
        for f in fn: cp(os.path.join(dp, f))
print("thieu:", missing)
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(OUT) for f in fn)
print("py:", len(files), "tong MB:", round(total / 1e6, 1))

# ---- tests: chi lay file test chi import module co trong goi
import ast as _ast
os.makedirs(os.path.join(OUT, "tests"), exist_ok=True)
kept, skipped = [], []
for tf in sorted(glob.glob("tests/*.py")) + sorted(glob.glob("tests/fixtures/*")):
    rel = tf.replace("\\", "/")
    if tf.endswith(".py") and os.path.basename(tf) != "conftest.py":
        need = imports(tf) if True else set()
        txt = open(tf, encoding="utf-8").read()
        lazy = {m for m in allmods if re.search(r"^\s*(import|from)\s+" + m + r"", txt, re.M)}
        if (need | lazy) - chosen:
            skipped.append(os.path.basename(tf)); continue
    cp(rel); kept.append(rel)
print("tests giu:", len([k for k in kept if k.endswith('.py')]), "bo:", skipped)
# ---- demo clip
import cv2
out = os.path.join(OUT, "demo", "demo_clip.mp4"); os.makedirs(os.path.dirname(out), exist_ok=True)
cap = cv2.VideoCapture("continuous_data/Dat/20261003_103331/session.mp4"); fps = cap.get(5)
w, h = int(cap.get(3)), int(cap.get(4)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(25 * fps))
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i in range(int(45 * fps)):
    ok, f = cap.read()
    if not ok: break
    vw.write(f)
vw.release()
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(OUT) for f in fn)
print("tong MB:", round(total / 1e6, 1))

shutil.copy2("scratch/chay.bat", os.path.join(OUT, "chay.bat"))
shutil.copy2("scratch/README_nop.md", os.path.join(OUT, "README.md"))
shutil.copy2("ui_dip_inspector.py", os.path.join(OUT, "ui_dip_inspector.py"))
z = shutil.make_archive(os.path.join(ROOT, "submission", "code"), "zip", os.path.join(ROOT, "submission"), "code")
print("zip:", z, round(os.path.getsize(z) / 1e6, 1), "MB")
