
import networkx as nx
from dataclasses import dataclass
from typing import List

@dataclass
class CausalNode:
    name: str
    type: str
    value: float
    parents: List[str]

class CausalGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        # base physics nodes
        for node in ["gravity", "mass", "friction", "normal_force", "com_x", "support_polygon", "ankle_torque", "fell"]:
            self.graph.add_node(node, type="physics")

    def add_contact_event(self, t, body_a, body_b, force, friction, cause_chain):
        node_id = f"contact_{t}_{body_a}_{body_b}"
        self.graph.add_node(node_id, force=force, friction=friction, type="contact", t=t)
        for parent in cause_chain:
            if parent not in self.graph:
                self.graph.add_node(parent, type="cause")
            self.graph.add_edge(parent, node_id)
        # contact -> stability
        self.graph.add_edge(node_id, "support_polygon")
        return node_id

    def do(self, var_name, value):
        """Pearl do-operator"""
        mutilated = self.graph.copy()
        for parent in list(mutilated.predecessors(var_name)):
            mutilated.remove_edge(parent, var_name)
        if var_name in mutilated:
            mutilated.nodes[var_name]["value"] = value
            mutilated.nodes[var_name]["intervened"] = True
        else:
            mutilated.add_node(var_name, value=value, intervened=True)
        return mutilated

    def explain_failure(self, failure_type="fell"):
        try:
            # find shortest path to failure
            if failure_type not in self.graph:
                return {"cause_chain": ["no data yet"], "minimal_fix": "collect more contacts"}
            ancestors = list(nx.ancestors(self.graph, failure_type))
            # last 4 ancestors as explanation
            chain = ancestors[-4:] if len(ancestors) >= 4 else ancestors
            # heuristic minimal fix
            if "friction" in chain or "normal_force" in chain:
                fix = "increase normal force or shift COM forward 3cm"
            elif "ankle_torque" in chain:
                fix = "ankle torque saturated -> pre-shift pelvis 2cm"
            else:
                fix = "COM outside support polygon -> reduce pelvis velocity"
            return {"cause_chain": chain, "minimal_fix": fix, "full_graph": self.export()}
        except Exception as e:
            return {"cause_chain": [str(e)], "minimal_fix": "error"}

    def export(self):
        return nx.node_link_data(self.graph, edges="links")
