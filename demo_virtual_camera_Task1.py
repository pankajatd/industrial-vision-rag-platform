# demo_virtual_camera_fixed.py
from pathlib import Path
import cv2
import sys

try:
    from src.task1_synthetic.virtual_camera import VirtualCameraCapture
except Exception as e:
    print("❌ Import error:", e)
    sys.exit(1)

data_dir = Path("data/synthetic_dataset")
if not data_dir.is_dir():
    print(f"❌ Dataset folder not found: {data_dir.resolve()}")
    sys.exit(1)

# Folder mode is auto‑detected because we give a valid source_dir
cap = VirtualCameraCapture(source_dir=str(data_dir), fps=30, loop=False)

print("✅ VirtualCameraCapture created")
print(f"   mode (auto‑detected) = {cap.mode}")
print(f"   total frames (estimated) = {len(cap.image_files) if hasattr(cap, 'image_files') else 'unknown'}")

first_frame = None          # will hold the first successfully read frame
frame_idx = 0
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("⚠️  No more frames (or read failed) at index", frame_idx)
        break

    frame_idx += 1
    print(f"🔹 Frame {frame_idx} read – shape {frame.shape}")

    # Remember the very first frame (you could also keep the last one if you prefer)
    if first_frame is None:
        first_frame = frame.copy()

    cv2.imshow("Virtual Camera (folder mode)", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("🔴  User aborted with 'q'")
        break

cap.release()
cv2.destroyAllWindows()
print(f"🏁 Demo finished after {frame_idx} frame(s)")

# ---------------------------------------------------------
# Save the stored frame (now we are sure it is a real image)
# ---------------------------------------------------------
if first_frame is not None:
    out_path = Path("first_frame.png")
    cv2.imwrite(str(out_path), first_frame)
    print(f"🖼️  First frame saved as {out_path.resolve()}")
else:
    print("❌ No frame was captured – nothing to save.")