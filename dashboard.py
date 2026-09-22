import cv2
import numpy as np
import time
import json
from pipeline import IndustrialVisionPipeline
from src.task1_synthetic.generator import SyntheticIndustrialGenerator, DefectType

def draw_hud(frame, alert, work_order):
    """Draws a diagnostic Heads-Up Display overlay on the camera frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Choose color based on severity
    color_map = {
        "PASS": (0, 255, 0),       # Green
        "LOW": (0, 255, 255),      # Yellow
        "MEDIUM": (0, 165, 255),   # Orange
        "CRITICAL": (0, 0, 255)    # Red
    }
    box_color = color_map.get(alert["severity_level"], (255, 255, 255))
    
    # Draw transparent sidebar
    sidebar_w = 350
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    
    # HUD Text
    x_offset = w - sidebar_w + 15
    y_offset = 35
    
    cv2.putText(frame, "DIAGNOSTIC DASHBOARD", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (x_offset, y_offset + 10), (w - 15, y_offset + 10), (100, 100, 100), 1)
    
    y_offset += 40
    cv2.putText(frame, f"Frame : {alert['frame_index']}", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 30
    cv2.putText(frame, f"Defect: {alert['defect_type'].upper()}", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    y_offset += 30
    cv2.putText(frame, f"Score : {alert['severity_score']}/10.0", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1)
    y_offset += 30
    cv2.putText(frame, f"Status: {alert['severity_level']}", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    if work_order:
        y_offset += 50
        cv2.putText(frame, "--- RAG WORK ORDER ---", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, f"ID: {work_order.get('id')}", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_offset += 25
        cv2.putText(frame, "Action: Check Terminal", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw border around the main image feed based on severity
    cv2.rectangle(frame, (0, 0), (w - sidebar_w, h), box_color, 4)
    
    return frame

def run_dashboard():
    # 1. Spin up the Master Pipeline
    pipeline = IndustrialVisionPipeline()
    
    # 2. Spin up the Virtual Camera (Task 1)
    print("[System] Starting Virtual Camera Feed...")
    generator = SyntheticIndustrialGenerator(img_size=(640, 640))
    
    # We will simulate a sequence of different defects coming down the assembly line
    test_sequence = [
        DefectType.NORMAL,
        DefectType.SCRATCH,
        DefectType.NORMAL,
        DefectType.CRACK,
        DefectType.CORROSION,
        DefectType.DIMENSIONAL
    ]
    
    frame_idx = 100
    for defect in test_sequence:
        print(f"\n=============================================")
        print(f"[CAMERA] CAPTURING FRAME {frame_idx}...")
        
        # Generate synthetic frame
        frame, true_meta, _ = generator.generate_sample(defect_type=defect, severity=8.0)
        
        # Run through Master Pipeline
        results = pipeline.process_frame(frame, frame_idx)
        
        alert = results["alert"]
        work_order = results["work_order"]
        mask = results["mask"]
        
        # Terminal Output
        print(f"  Predicted Defect : {alert['defect_type'].upper()}")
        print(f"  Severity Score   : {alert['severity_score']} ({alert['severity_level']})")
        
        if work_order:
            print(f"\n[WORK ORDER] GENERATED: {work_order['id']}")
            print(f"   Safety: {work_order['safety_directive']}")
            print(f"   Steps : {len(work_order['repair_steps'])} steps extracted.")
        
        # Render visual dashboard overlay
        # First, highlight the defect on the image using the mask from Task 2
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = (0, 0, 255) # Red highlight for defect
        display_frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
        
        # Add the dashboard HUD
        dashboard_img = draw_hud(display_frame, alert, work_order)
        
        # Display via OpenCV window
        cv2.imshow("Task 5: Industrial Vision & Diagnostic Dashboard", dashboard_img)
        
        print("Press any key in the image window to process the next item...")
        cv2.waitKey(0) # Wait for user to press a key before next frame
        
        frame_idx += 1

    cv2.destroyAllWindows()
    print("\n[System] Inspection sequence complete. Shutting down.")

if __name__ == "__main__":
    run_dashboard()
