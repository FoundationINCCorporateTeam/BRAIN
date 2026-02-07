"""
Neuron Conversation Engine: orchestrates all subsystems.
"""
import random
import time
from typing import Dict, Optional

from .graph.brain import BrainGraph
from .loaders.lexicon_loader import load_lexicon, Lexicon
from .loaders.graph_loader import load_graph
from .perception.input_processor import InputProcessor, PerceptionResult
from .cognition.dynamics import run_dynamics, DynamicsConfig, DynamicsResult
from .cognition.goals import select_goal
from .memory.memory import Memory
from .language.motor import generate_response, MotorResult
from .trace.tracer import Trace


class ConversationEngine:
    def __init__(self, lexicon_path: str, graph_path: str, seed: int = 42):
        self.rng = random.Random(seed)
        self.seed = seed

        self.lexicon: Lexicon = load_lexicon(lexicon_path)
        self.graph: BrainGraph = load_graph(graph_path)

        self.perception = InputProcessor(self.lexicon)
        self.memory = Memory()
        self.dynamics_config = DynamicsConfig(steps=20)

        self.modulators: Dict[str, float] = {
            'curiosity': 0.5,
            'calm': 0.6,
            'urgency': 0.3,
        }
        self.debug_mode: bool = False
        self.recent_words: list = []
        self.turn_count: int = 0

    def startup_summary(self) -> str:
        lines = [
            "Neuron Conversation Engine",
            "Mode: CPU-only | Deterministic",
            f"Brain loaded: {self.graph.summary()}",
            f"Lexicon loaded: {self.lexicon.summary()}",
            f"Seed: {self.seed}",
            "Type 'exit' to quit.",
        ]
        return '\n'.join(lines)

    def process_input(self, user_input: str) -> tuple:
        """Process user input and generate response with trace."""
        self.turn_count += 1
        start_time = time.time()
        trace = Trace()

        # 1. Perception
        perception_result = self.perception.process(user_input)

        for word, concepts in perception_result.matched_words:
            trace.input_mapping.append((word, concepts))
        for phrase, concepts in perception_result.matched_phrases:
            trace.input_mapping.append((phrase, concepts))

        trace.initial_activations = dict(perception_result.activated_concepts)
        trace.modulators = dict(self.modulators)

        # 2. Memory retrieval boost
        current_concepts = list(perception_result.activated_concepts.keys())
        memory_boost = self.memory.get_memory_boost(current_concepts)
        trace.memory_effects = dict(memory_boost)

        combined_activations = dict(perception_result.activated_concepts)
        for cid, boost in memory_boost.items():
            combined_activations[cid] = combined_activations.get(cid, 0.0) + boost

        # 3. Dynamics
        dynamics_result = run_dynamics(
            self.graph, combined_activations,
            self.dynamics_config, self.modulators
        )

        trace.step_records = dynamics_result.steps
        trace.top_edges = dynamics_result.top_contributing_edges

        # 4. Goal selection
        goal_result = select_goal(self.graph)
        trace.selected_goal = goal_result.selected_goal or "goal_inform"
        trace.goal_candidates = goal_result.candidates

        # 5. Language generation
        motor_result = generate_response(
            self.graph, self.lexicon,
            trace.selected_goal,
            self.rng,
            self.recent_words
        )

        trace.language_candidates = motor_result.candidates_considered[:15]
        trace.language_selected = motor_result.selected_words
        trace.final_words = [w.word for w in motor_result.selected_words]

        response_text = motor_result.final_text

        self.recent_words.extend(trace.final_words)
        if len(self.recent_words) > 30:
            self.recent_words = self.recent_words[-30:]

        # 6. Store in memory
        self.memory.store_turn(
            user_input, response_text,
            current_concepts,
            trace.selected_goal
        )

        self._update_modulators(perception_result)

        elapsed = time.time() - start_time

        return response_text, trace, elapsed

    def _update_modulators(self, perception: PerceptionResult):
        """Slightly adjust modulators based on input."""
        if '?' in perception.raw_input:
            self.modulators['curiosity'] = min(1.0, self.modulators['curiosity'] + 0.1)
        else:
            self.modulators['curiosity'] = max(0.2, self.modulators['curiosity'] - 0.05)

        self.modulators['urgency'] = max(0.1, self.modulators['urgency'] - 0.02)

    def show_brain(self) -> str:
        """Return brain statistics."""
        lines = [
            f"Brain: {self.graph.summary()}",
            f"Node types:",
        ]
        for ntype in ('concept', 'topic', 'emotion', 'goal', 'motor', 'lexeme'):
            nodes = self.graph.get_nodes_by_type(ntype)
            lines.append(f"  {ntype}: {len(nodes)}")

        edge_types = {}
        for e in self.graph.edges:
            edge_types[e.type] = edge_types.get(e.type, 0) + 1
        lines.append("Edge types:")
        for etype, count in edge_types.items():
            lines.append(f"  {etype}: {count}")

        return '\n'.join(lines)

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
