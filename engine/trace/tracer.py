"""
Trace system: records and formats the thought trace for each response.
"""
from typing import Dict, List, Tuple, Optional


class Trace:
    def __init__(self):
        self.input_mapping: List[Tuple[str, List[str]]] = []
        self.initial_activations: Dict[str, float] = {}
        self.modulators: Dict[str, float] = {}
        self.step_records: list = []
        self.top_edges: List[Tuple[str, str, str, float]] = []
        self.memory_effects: Dict[str, float] = {}
        self.selected_goal: str = ""
        self.goal_candidates: List[Tuple[str, float]] = []
        self.language_candidates: list = []
        self.language_selected: list = []
        self.final_words: List[str] = []

    def format_compact(self) -> str:
        """Format trace for compact display."""
        lines = []
        lines.append("─── THOUGHT TRACE ───")

        if self.input_mapping:
            lines.append("  Input → Concepts:")
            for token, concepts in self.input_mapping:
                lines.append(f"    '{token}' → {concepts}")

        if self.initial_activations:
            lines.append("  Initial Activations:")
            sorted_act = sorted(self.initial_activations.items(), key=lambda x: x[1], reverse=True)[:5]
            for nid, val in sorted_act:
                lines.append(f"    {nid}: {val:.3f}")

        if self.modulators:
            mods = ', '.join(f"{k}={v:.2f}" for k, v in self.modulators.items())
            lines.append(f"  Modulators: {mods}")

        if self.step_records:
            steps_to_show = []
            n = len(self.step_records)
            if n >= 1:
                steps_to_show.append(self.step_records[0])
            if n >= 3:
                steps_to_show.append(self.step_records[n // 2])
            if n >= 2:
                steps_to_show.append(self.step_records[-1])

            lines.append("  Dynamics (selected steps):")
            for sr in steps_to_show:
                top = ', '.join(f"{nid}={act:.2f}" for nid, act in sr.top_active[:4])
                lines.append(f"    Step {sr.step_num}: [{top}]")

        if self.top_edges:
            lines.append("  Top Routes (edges):")
            for src, tgt, etype, contrib in self.top_edges[:5]:
                lines.append(f"    {src} →({etype})→ {tgt}  contrib={contrib:.3f}")

        if self.memory_effects:
            lines.append("  Memory Boost:")
            for cid, boost in self.memory_effects.items():
                lines.append(f"    {cid}: +{boost:.3f}")

        if self.selected_goal:
            lines.append(f"  Goal: {self.selected_goal}")
            if self.goal_candidates:
                cands = ', '.join(f"{g}={a:.2f}" for g, a in self.goal_candidates[:4])
                lines.append(f"    Candidates: [{cands}]")

        if self.language_selected:
            lines.append("  Word Selection:")
            for wc in self.language_selected[:8]:
                lines.append(f"    '{wc.word}' score={wc.score:.3f} ({wc.reason})")

        if self.final_words:
            lines.append(f"  Output: {' '.join(self.final_words)}")

        lines.append("─────────────────────")
        return '\n'.join(lines)

    def format_full(self) -> str:
        """Format full trace with all details."""
        lines = []
        lines.append("═══ FULL THOUGHT TRACE ═══")

        lines.append("\n[INPUT MAPPING]")
        for token, concepts in self.input_mapping:
            lines.append(f"  '{token}' → {concepts}")

        lines.append("\n[INITIAL ACTIVATIONS]")
        for nid, val in sorted(self.initial_activations.items(), key=lambda x: x[1], reverse=True):
            if val > 0:
                lines.append(f"  {nid}: {val:.4f}")

        lines.append("\n[MODULATORS]")
        for k, v in self.modulators.items():
            lines.append(f"  {k}: {v:.3f}")

        lines.append("\n[DYNAMICS STEPS]")
        for sr in self.step_records:
            top = ', '.join(f"{nid}={act:.3f}" for nid, act in sr.top_active[:6])
            lines.append(f"  Step {sr.step_num:2d}: [{top}]")

        lines.append("\n[TOP CONTRIBUTING EDGES]")
        for src, tgt, etype, contrib in self.top_edges:
            lines.append(f"  {src} →({etype})→ {tgt}  contribution={contrib:.4f}")

        lines.append("\n[MEMORY EFFECTS]")
        if self.memory_effects:
            for cid, boost in self.memory_effects.items():
                lines.append(f"  {cid}: +{boost:.4f}")
        else:
            lines.append("  (none)")

        lines.append("\n[GOAL SELECTION]")
        lines.append(f"  Selected: {self.selected_goal}")
        for g, a in self.goal_candidates:
            marker = " ◄" if g == self.selected_goal else ""
            lines.append(f"    {g}: {a:.4f}{marker}")

        lines.append("\n[LANGUAGE CANDIDATES]")
        for wc in self.language_candidates[:15]:
            lines.append(f"  '{wc.word}' pos={wc.pos} score={wc.score:.3f} | {wc.reason}")

        lines.append("\n[SELECTED WORDS]")
        for i, wc in enumerate(self.language_selected):
            lines.append(f"  {i+1}. '{wc.word}' pos={wc.pos} score={wc.score:.3f}")

        lines.append(f"\n[OUTPUT] {' '.join(self.final_words)}")
        lines.append("══════════════════════════")
        return '\n'.join(lines)
