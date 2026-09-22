from typing import Dict, Any, List, TypedDict, Union
from .vector_store import VectorStore

# Define State Structure
class AgenticDiagnosticState(TypedDict):
    alert: Dict[str, Any]
    search_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_relevance_score: float
    iteration: int
    work_order: Dict[str, Any]
    human_approved: bool

# Fallback StateGraph implementation in case langgraph is not installed
class StateGraph:
    def __init__(self, state_schema):
        self.nodes = {}
        self.edges = {}
        self.conditional_edges = {}
        self.entry_point = None

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, from_node, to_node):
        self.edges[from_node] = to_node

    def add_conditional_edges(self, from_node, condition_func, route_map):
        self.conditional_edges[from_node] = (condition_func, route_map)

    def set_entry_point(self, node):
        self.entry_point = node

    def compile(self):
        return CompiledGraph(self)

class CompiledGraph:
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, state: dict) -> dict:
        current_node = self.graph.entry_point
        while current_node != "END":
            state = self.graph.nodes[current_node](state)
            
            if current_node in self.graph.conditional_edges:
                cond_func, route_map = self.graph.conditional_edges[current_node]
                route_key = cond_func(state)
                current_node = route_map.get(route_key, "END")
            elif current_node in self.graph.edges:
                current_node = self.graph.edges[current_node]
            else:
                break
        return state

try:
    from langgraph.graph import StateGraph as LGStateGraph, END
    GraphClass = LGStateGraph
    END_NODE = END
except ImportError:
    GraphClass = StateGraph
    END_NODE = "END"


class DiagnosticRAGAgent:
    """LangGraph State Machine for Diagnostic Orchestration."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = GraphClass(AgenticDiagnosticState)

        # Add Nodes
        workflow.add_node("ingest_alert", self.ingest_alert)
        workflow.add_node("formulate_queries", self.formulate_queries)
        workflow.add_node("retrieve_manuals", self.retrieve_manuals)
        workflow.add_node("synthesize_report", self.synthesize_report)
        workflow.add_node("signoff_gate", self.human_signoff)

        # Build Edges
        workflow.set_entry_point("ingest_alert")
        workflow.add_edge("ingest_alert", "formulate_queries")
        workflow.add_edge("formulate_queries", "retrieve_manuals")

        workflow.add_conditional_edges(
            "retrieve_manuals",
            self.self_rag_check,
            {
                "sufficient": "synthesize_report",
                "rewrite": "formulate_queries",
                "fallback": "synthesize_report"
            }
        )

        workflow.add_conditional_edges(
            "synthesize_report",
            self.route_human_gate,
            {
                "signoff_gate": "signoff_gate",
                "complete": END_NODE
            }
        )
        
        workflow.add_edge("signoff_gate", END_NODE)

        return workflow.compile()

    # --- Node Functions ---

    def ingest_alert(self, state: AgenticDiagnosticState) -> AgenticDiagnosticState:
        state["iteration"] = 0
        state["search_queries"] = []
        state["retrieved_chunks"] = []
        state["human_approved"] = False
        return state

    def formulate_queries(self, state: AgenticDiagnosticState) -> AgenticDiagnosticState:
        defect_type = state["alert"].get("defect_type", "unknown")
        
        # Create a targeted query based on the defect type
        if state["iteration"] == 0:
            query = f"{defect_type} repair procedure SOP"
        else:
            query = f"{defect_type} safety tools steps" # Re-written query on iteration

        state["search_queries"] = [query]
        return state

    def retrieve_manuals(self, state: AgenticDiagnosticState) -> AgenticDiagnosticState:
        query = state["search_queries"][-1]
        results = self.vector_store.retrieve(query, top_k=3)
        
        state["retrieved_chunks"] = results
        state["iteration"] += 1
        
        # Simple mock grading: If we found results, score is high. Else low.
        if results and results[0].get("score", 0) > 0.1:
            state["retrieval_relevance_score"] = 0.90
        else:
            state["retrieval_relevance_score"] = 0.40
            
        return state

    def self_rag_check(self, state: AgenticDiagnosticState) -> str:
        """Determines if retrieval was successful or if query needs rewriting."""
        if state["retrieval_relevance_score"] >= 0.75:
            return "sufficient"
        elif state["iteration"] >= 3:
            return "fallback"
        else:
            return "rewrite"

    def synthesize_report(self, state: AgenticDiagnosticState) -> AgenticDiagnosticState:
        """Generates the final work order from the retrieved chunks."""
        severity_level = state["alert"].get("severity_level", "LOW")
        chunks = state["retrieved_chunks"]
        
        # Extract content from chunks to build report
        combined_text = "\n".join([c["content"] for c in chunks])
        
        # Simple extraction logic (mocking LLM synthesis for local offline setup)
        safety_directives = "OSHA LOTO, PPE required." if "LOTO" in combined_text else "Standard safety PPE."
        repair_steps = [c["content"] for c in chunks if "Step" in c["content"] or "1." in c["content"]]
        if not repair_steps:
            repair_steps = ["Follow standard procedures for " + state["alert"].get("defect_type", "issue")]
            
        state["work_order"] = {
            "id": f"WO-{state['alert'].get('frame_index', '000')}",
            "priority": severity_level,
            "safety_directive": safety_directives,
            "repair_steps": repair_steps,
            "sources": list(set([c["metadata"]["source"] for c in chunks])),
            "status": "PENDING_APPROVAL" if severity_level == "CRITICAL" else "AUTO_APPROVED"
        }
        return state

    def route_human_gate(self, state: AgenticDiagnosticState) -> str:
        """Routes to human signoff if priority is CRITICAL."""
        if state["work_order"]["priority"] == "CRITICAL" and not state["human_approved"]:
            return "signoff_gate"
        return "complete"

    def human_signoff(self, state: AgenticDiagnosticState) -> AgenticDiagnosticState:
        """Mock node representing a human review gateway."""
        state["human_approved"] = True
        state["work_order"]["status"] = "APPROVED"
        return state

    def run(self, alert_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point to run the RAG agent for an incoming alert."""
        initial_state = AgenticDiagnosticState(
            alert=alert_payload,
            search_queries=[],
            retrieved_chunks=[],
            retrieval_relevance_score=0.0,
            iteration=0,
            work_order={},
            human_approved=False
        )
        final_state = self.graph.invoke(initial_state)
        return final_state["work_order"]
