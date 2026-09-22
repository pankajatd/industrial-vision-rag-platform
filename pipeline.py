import os
import cv2
import pandas as pd
import numpy as np

# Import Task 1: Synthetic Data
from src.task1_synthetic.virtual_camera import VirtualCameraCapture
from src.task1_synthetic.generator import SyntheticIndustrialGenerator

# Import Task 2: Computer Vision
from src.task2_opencv.preprocessor import preprocess
from src.task2_opencv.segmenter import segment
from src.task2_opencv.feature_extractor import extract_features

# Import Task 3: Machine Learning
from src.task3_classifier.api import get_model

# Import Task 4: RAG Agent
from src.task4_rag.manual_generator import ManualGenerator
from src.task4_rag.vector_store import VectorStore
from src.task4_rag.langgraph_agent import DiagnosticRAGAgent

class IndustrialVisionPipeline:
    """Master Orchestrator connecting Tasks 1, 2, 3, and 4."""

    def __init__(self):
        print("[System] Initializing Industrial Vision RAG Pipeline...")
        
        # 1. Initialize Task 4 RAG (Manuals & Vector Store)
        print("[System] Loading Manuals & Vector Database...")
        manual_dir = os.path.join(os.path.dirname(__file__), "data", "manuals")
        generator = ManualGenerator(output_dir=manual_dir)
        manual_paths = generator.generate_all_manuals()
        
        self.vector_store = VectorStore()
        self.vector_store.ingest_files(manual_paths)
        self.rag_agent = DiagnosticRAGAgent(self.vector_store)
        
        # 2. Load Task 3 ML Model
        print("[System] Loading Scikit-Learn Classifier...")
        self.model = get_model()

        print("[System] Pipeline Ready.\n")

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> dict:
        """Processes a single image frame through the entire pipeline."""
        
        # --- TASK 2: OPENCV PIPELINE ---
        pre = preprocess(frame)
        mask = segment(pre)
        feats = extract_features(pre, mask)
        
        # --- TASK 3: ML CLASSIFICATION ---
        X = pd.DataFrame([feats])
        
        # Filter out features that might not have been in the training data if needed
        # (Assuming the model was trained on the exact output of extract_features)
        try:
            pred_label = str(self.model.predict(X)[0])
        except Exception as e:
            # Fallback if feature columns don't match exactly
            pred_label = "unknown"
            print(f"ML Prediction Error: {e}")

        # Calculate dynamic severity based on defect size (Task 3 Regressor simulation)
        defect_ratio = feats.get("total_area", 0) / (frame.shape[0] * frame.shape[1])
        if pred_label == "normal" or defect_ratio == 0:
            severity_score = 0.0
            severity_level = "PASS"
        else:
            # Scale ratio to a 1.0 - 10.0 score
            severity_score = min(10.0, round(defect_ratio * 1500 + 2.0, 1))
            if severity_score > 7.5:
                severity_level = "CRITICAL"
            elif severity_score > 4.0:
                severity_level = "MEDIUM"
            else:
                severity_level = "LOW"
                
        alert = {
            "frame_index": frame_idx,
            "defect_type": pred_label,
            "severity_score": severity_score,
            "severity_level": severity_level
        }
        
        # --- TASK 4: RAG WORK ORDER GENERATION ---
        work_order = None
        if severity_level in ["MEDIUM", "CRITICAL"]:
            # Only trigger LangGraph RAG for severe defects to save compute
            work_order = self.rag_agent.run(alert)
        
        # Return all compiled data
        return {
            "alert": alert,
            "work_order": work_order,
            "mask": mask,
            "features": feats
        }
