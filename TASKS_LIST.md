# Industrial Vision & Diagnostic RAG Platform
## Complete Tasks List & Architecture Blueprint

---

### Project Overview
The objective of this platform is to build an integrated, automated industrial vision and diagnostic intelligence system that detects component surface defects and generates actionable maintenance Standard Operating Procedures (SOPs).

Instead of disconnected scripts, the system forms a continuous 5-stage automated pipeline:
```
[ 1. Synthetic Camera ] ──► [ 2. OpenCV Features ] ──► [ 3. Scikit-Learn ML ] ──► [ 4. LangGraph RAG ] ──► [ 5. Dashboard ]
```

---

### End-to-End Tasks Roadmap

| Task | Module | Core Function | Primary Output |
| :--- | :--- | :--- | :--- |
| **Task 1** | **Synthetic Image Generator & Virtual Camera** | Generates realistic metal component images with parameterizable defects (scratches, cracks, corrosion, dimensional flaws) and streams them like an industrial camera. | Synthetic Image Dataset & `VirtualCameraCapture` class |
| **Task 2** | **OpenCV Preprocessing & Feature Extraction** | Denoises frames, equalizes lighting (CLAHE), detects edges (Canny/Sobel), isolates defect contours, and extracts 15 numerical features. | Numerical Feature Vector & Visual HUD Overlay Image |
| **Task 3** | **Scikit-Learn ML Models** | Uses a Random Forest Classifier to identify the defect type and a Regressor to predict continuous severity (0.0 to 10.0). | Trained `.joblib` models & Diagnostic JSON payload |
| **Task 4** | **PDF Manual Vector DB & LangGraph RAG Agent** | Indexes industrial PDF manuals into ChromaDB and uses a LangGraph state machine to issue actionable repair work orders. | Formatted Technician Maintenance Dispatch Ticket |
| **Task 5** | **End-to-End Pipeline & Diagnostic Dashboard** | Integrates all 4 modules into a single automated controller and visual operator dashboard with automated batch testing. | Master `pipeline.py`, Diagnostic UI & Audit Report |

---

### Detailed Task Specifications

#### Task 1: Synthetic Industrial Image Generator & Virtual Camera Feed
* **Subtask 1.1: Component Surface Generator**
  * Generate realistic base metal texture with stochastic grain, gradient lighting, and specular reflectance.
* **Subtask 1.2: Programmatic Defect Injection**
  * **Normal**: Defect-free surface with standard manufacturing grain.
  * **Scratch**: Linear and curved thin surface fissures with variable orientation, width, and depth.
  * **Crack**: Multi-segment branching fractures with jagged, low-solidity boundaries.
  * **Pit / Corrosion**: Irregular clustered dark voids and surface oxidation patches.
  * **Dimensional Flaw**: Deformed edges, chipped corners, or warped perimeter profiles.
* **Subtask 1.3: Severity Scaling & Metadata Export**
  * Parameterize defects on a continuous scale (1.0 to 10.0 / Low, Medium, Critical).
  * Automatically generate ground-truth `metadata.json` with defect labels, severity scores, and bounding boxes `[x, y, w, h]`.
* **Subtask 1.4: Virtual Camera Streamer (`VirtualCameraCapture`)**
  * Emulate OpenCV's `cv2.VideoCapture` interface (`read()`, `isOpened()`, `release()`) to stream synthetic frames seamlessly into downstream pipelines.

---

#### Task 2: OpenCV Preprocessing & Feature Extraction Pipeline
* **Subtask 2.1: Image Preprocessing & Contrast Normalization**
  * Convert incoming frames to Grayscale and LAB/HSV color spaces.
  * Apply Bilateral Filtering / Gaussian Blur to suppress surface grain noise.
  * Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to balance factory lighting.
* **Subtask 2.2: Multi-Scale Edge & Defect Segmentation**
  * Multi-threshold Canny edge detection and Sobel gradient operators.
  * Morphological Top-Hat, Black-Hat, and Closing transforms to weld fragmented edges.
  * Otsu's adaptive binarization to produce clean binary defect masks (0 = background, 255 = defect).
* **Subtask 2.3: Feature Engineering (15-Dimensional Vector)**
  * **Geometric**: Contour count, defect area ratio, aspect ratio, solidity, Hu moments.
  * **Texture**: Gray-Level Co-occurrence Matrix (GLCM) contrast, dissimilarity, homogeneity, energy.
  * **Statistical & Gradient**: Mean intensity, standard deviation, skewness, edge pixel density.
* **Subtask 2.4: Feature Serialization & Visual Inspection Overlay**
  * Output clean tabular `pandas.DataFrame` / `numpy.ndarray` rows ready for ML inference.
  * Generate annotated visual overlays with color-coded bounding boxes (Green=Pass, Yellow=Low, Red=Critical), defect contours, and HUD status banners.

---

#### Task 3: Scikit-Learn Anomaly Classifier & Severity Model
* **Subtask 3.1: Dataset Partitioning & Feature Pipeline**
  * Stratified 70/15/15 train/validation/test split across all defect classes.
  * Feature normalization using `StandardScaler` within an `sklearn.pipeline.Pipeline`.
  * Feature importance analysis to identify top distinguishing metrics per defect type.
* **Subtask 3.2: Multi-Class Anomaly Classification Model**
  * Train and optimize Random Forest Classifier (`RandomForestClassifier`) using 5-fold `GridSearchCV`.
  * Output predicted class (`Normal`, `Scratch`, `Crack`, `Corrosion`, `Dimensional`) and probability distribution (`predict_proba()`).
* **Subtask 3.3: Severity Score Regression Model**
  * Train Gradient Boosting / Random Forest Regressor predicting continuous severity (`0.0` to `10.0`).
  * Map scores to actionable industrial tiers:
    * `0.0 - 2.0`: PASS (Normal)
    * `2.1 - 5.0`: LOW (Cosmetic / Log for routine maintenance)
    * `5.1 - 7.5`: MEDIUM (Functional degradation / Schedule service)
    * `7.6 - 10.0`: CRITICAL (Failure risk / Trigger emergency SOP & halt line)
* **Subtask 3.4: Model Serialization & Inference Interface**
  * Export trained weights (`anomaly_classifier.joblib`, `severity_regressor.joblib`, `scaler.joblib`).
  * Package Python wrapper function returning structured JSON payloads for live inspection.

---

#### Task 4: PDF Manual Vector DB & LangGraph RAG Agent
* **Subtask 4.1: Industrial Maintenance PDF Manuals**
  * Ingest comprehensive equipment maintenance manuals (SOPs for polishing, welding/replacing cracked parts, acid passivation for corrosion, calibration, and torque specs).
* **Subtask 4.2: Document Chunking & Vector Database (ChromaDB / FAISS)**
  * Recursive character chunking with section metadata.
  * Generate dense vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
  * Persist in local ChromaDB for sub-millisecond semantic retrieval.
* **Subtask 4.3: LangGraph State Machine Workflow**
  * **Gatekeeper Node**: Checks severity. If `< 2.0`, bypasses RAG and logs routine pass.
  * **Retriever Node**: Executes targeted semantic search against ChromaDB.
  * **Document Relevance Grader**: Filters out irrelevant pages to prevent hallucinations.
  * **Synthesizer Node**: Uses LLM to draft a structured maintenance work order.
* **Subtask 4.4: Actionable Work Order Generation**
  * Generates formatted maintenance ticket containing:
    1. Safety Precautions (OSHA Lock-Out / Tag-Out, required PPE).
    2. Required Tools & Replacement Part Numbers.
    3. Step-by-Step SOP Procedure with verbatim manual page citations.
    4. Post-Repair Re-commissioning Checklist.

---

#### Task 5: End-to-End System Integration & Diagnostic Dashboard
* **Subtask 5.1: Master Pipeline Orchestrator (`pipeline.py`)**
  * Single controller tying together Frame Capture ➔ OpenCV ➔ Scikit-Learn ➔ LangGraph RAG.
  * Stage-by-stage latency profiling and error handling.
* **Subtask 5.2: Diagnostic Operator Dashboard**
  * Interactive interface featuring:
    * Visual Inspection Frame (with bounding boxes, contours, HUD badge).
    * Real-Time Telemetry (defect name, confidence %, continuous severity score).
    * Remediation Work Order (step-by-step SOP checklist from manual).
* **Subtask 5.3: Automated Batch Validation & Audit Logging**
  * End-to-end test suite evaluating sample frames from all defect classes.
  * Exports compliance audit log (`audit_report.json` and summary table).

---

### Recommended Project Directory Structure

```
industrial_vision_rag/
│
├── TASKS_LIST.md                     <-- This Roadmap Document
├── requirements.txt                  <-- Project Dependencies
├── pipeline.py                       <-- Master Orchestrator (Task 5)
│
├── src/
│   ├── task1_synthetic/
│   │   ├── generator.py              <-- Synthetic image and defect creation
│   │   └── virtual_camera.py         <-- cv2.VideoCapture emulator
│   │
│   ├── task2_opencv/
│   │   ├── preprocessor.py           <-- Denoising, CLAHE, color conversion
│   │   ├── segmenter.py              <-- Canny edge detection & morphology
│   │   └── feature_extractor.py      <-- GLCM, shape, contour features
│   │
│   ├── task3_ml/
│   │   ├── train.py                  <-- Model training & evaluation script
│   │   ├── inference.py              <-- Predicts class and severity score
│   │   └── models/                   <-- Serialized .joblib artifacts
│   │
│   ├── task4_rag/
│   │   ├── manuals/                  <-- Industrial PDF SOP documents
│   │   ├── vector_store.py           <-- ChromaDB ingestion and retriever
│   │   └── langgraph_agent.py        <-- LangGraph decision graph & work order
│   │
│   └── dashboard/
│       └── app.py                    <-- Operator visual dashboard
│
└── data/
    ├── synthetic_dataset/            <-- Generated train/val/test images
    └── audit_reports/                <-- Inspection JSON logs and reports
```
