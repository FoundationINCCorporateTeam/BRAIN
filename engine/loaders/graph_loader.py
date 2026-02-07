"""
Graph loader: parses graph.brain file.
"""
from ..graph.node import Node
from ..graph.edge import Edge
from ..graph.brain import BrainGraph


def load_graph(filepath: str) -> BrainGraph:
    """Parse a graph.brain file and return a BrainGraph."""
    graph = BrainGraph()
    errors = []
    pending_edges = []

    with open(filepath, 'r') as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('|')
            record_type = parts[0].strip()

            try:
                if record_type == 'N':
                    if len(parts) < 7:
                        errors.append(f"Line {line_num}: NODE record needs 7 fields, got {len(parts)}")
                        continue
                    node_id = parts[1].strip()
                    node_type = parts[2].strip()
                    label = parts[3].strip()
                    baseline = float(parts[4].strip())
                    decay = float(parts[5].strip())
                    threshold = float(parts[6].strip())
                    node = Node(node_id, node_type, label, baseline, decay, threshold)
                    graph.add_node(node)

                elif record_type == 'E':
                    if len(parts) < 5:
                        errors.append(f"Line {line_num}: EDGE record needs 5 fields, got {len(parts)}")
                        continue
                    source_id = parts[1].strip()
                    target_id = parts[2].strip()
                    edge_type = parts[3].strip()
                    weight = float(parts[4].strip())
                    pending_edges.append((line_num, source_id, target_id, edge_type, weight))

                else:
                    errors.append(f"Line {line_num}: Unknown record type '{record_type}'")

            except Exception as e:
                errors.append(f"Line {line_num}: Parse error: {e}")

    # Add edges after all nodes are loaded
    for line_num, src, tgt, etype, w in pending_edges:
        try:
            edge = Edge(src, tgt, etype, w)
            graph.add_edge(edge)
        except ValueError as e:
            errors.append(f"Line {line_num}: Edge error: {e}")

    # Validate
    validation_errors = graph.validate()
    errors.extend(validation_errors)

    if errors:
        error_msg = "Graph validation errors:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    return graph
