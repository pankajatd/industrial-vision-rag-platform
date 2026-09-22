import os
import json
from src.task4_rag.manual_generator import ManualGenerator
from src.task4_rag.vector_store import VectorStore
from src.task4_rag.langgraph_agent import DiagnosticRAGAgent

def run_demo():
    print("--- Starting Task 4 RAG Demo ---")
    
    # 1. Generate the synthetic manuals
    print("\n1. Generating synthetic maintenance manuals...")
    manual_dir = os.path.join(os.path.dirname(__file__), "data", "manuals")
    generator = ManualGenerator(output_dir=manual_dir)
    manual_paths = generator.generate_all_manuals()
    print(f"   Created {len(manual_paths)} manuals in {manual_dir}")

    # 2. Spin up the Vector Store and ingest manuals
    print("\n2. Ingesting manuals into local Vector Store...")
    vector_store = VectorStore()
    vector_store.ingest_files(manual_paths)
    print("   Ingestion complete. TF-IDF index is ready.")

    # 3. Initialize the LangGraph RAG Agent
    print("\n3. Initializing LangGraph Diagnostic Agent...")
    agent = DiagnosticRAGAgent(vector_store=vector_store)

    # 4. Simulate an alert coming in from Task 3 (ML Classifier)
    mock_alert = {
        "frame_index": 1042,
        "defect_type": "corrosion",
        "severity_score": 8.7,
        "severity_level": "CRITICAL"
    }
    
    print("\n--- INCOMING ALERT FROM TASK 3 ---")
    print(json.dumps(mock_alert, indent=2))

    # 5. Run the Agent
    print("\n4. Agent is searching manuals and synthesizing work order...")
    work_order = agent.run(mock_alert)

    # 6. Display the final Output
    print("\n==================================================")
    print("      FINAL MAINTENANCE WORK ORDER TICKET           ")
    print("==================================================")
    print(f"Work Order ID : {work_order.get('id')}")
    print(f"Status        : {work_order.get('status')}")
    print(f"Priority      : {work_order.get('priority')}")
    print("--------------------------------------------------")
    print("SAFETY DIRECTIVE:")
    print(f"  {work_order.get('safety_directive')}")
    print("--------------------------------------------------")
    print("REPAIR STEPS:")
    for step in work_order.get('repair_steps', []):
        # Splitting steps by newlines to make it print cleanly
        for line in step.split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
    print("--------------------------------------------------")
    print("SOURCES REFERENCED:")
    for source in work_order.get('sources', []):
        print(f"  - {source}")
    print("==================================================\n")

if __name__ == "__main__":
    run_demo()
