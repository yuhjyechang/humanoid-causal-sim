"""
Structural Causal Model for humanoid physics
"""
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CausalNode:
    name: str
    type: str  # 'force', 'state', 'contact', 'action'
    value: float
    parents: List[str]

class CausalGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.history = []

    def add_contact_event(self, t, body_a, body_b, force, friction, cause_chain):
        """Every contact creates causal edges"""
        node_id = f"contact_{t}_{body_a}_{body_b}"
        self.graph.add_node(node_id, force=force, friction=friction, type='contact')
        for parent in cause_chain:
            self.graph.add_edge(parent, node_id)
        self.history.append((t, node_id, cause_chain))

    def do(self, var_name, value):
        """Pearl's do-operator: clamp variable and cut incoming edges"""
        # Returns mutilated graph
        mutilated = self.graph.copy()
        # cut parents
        for parent in list(mutilated.predecessors(var_name)):
            mutilated.remove_edge(parent, var_name)
        mutilated.nodes[var_name]['value'] = value
        mutilated.nodes[var_name]['intervened'] = True
        return mutilated

    def explain_failure(self, failure_type="fell"):
        """Find minimal causal chain to failure"""
        # 1. Find failure node
        # 2. Backtrack via ancestors
        # 3. Find minimal fix via counterfactual search
        # Returns: chain + counterfactual suggestion
        chain = nx.ancestors(self.graph, failure_type)
        return {
            'cause_chain': list(chain)[-5:], # last 5 causes
            'minimal_fix': 'TBD: run counterfactual search',
            'graph': self.graph
        }

    def export(self):
        return nx.node_link_data(self.graph, edges="links")


def find_minimal_fix(self, failure_node):
    # Binary search over pelvis x: would shifting 1cm, 2cm, 3cm have prevented fall?
    for delta in [0.01, 0.02, 0.03, 0.05]:
        cf_state = self.do(pelvis_x=f"current+{delta}")
        if not self.predicts_failure(cf_state):
            return f"shift pelvis {delta*100:.0f}cm"
