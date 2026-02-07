#!/usr/bin/env python3
"""
Neuron Conversation Engine - Main Entry Point
CPU-only | Deterministic | No pretrained models

Usage: python3 main.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import ConversationEngine


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lexicon_path = os.path.join(base_dir, 'data', 'lexicon.brain')
    graph_path = os.path.join(base_dir, 'data', 'graph.brain')

    for path, name in [(lexicon_path, 'lexicon'), (graph_path, 'graph')]:
        if not os.path.isfile(path):
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)

    try:
        engine = ConversationEngine(lexicon_path, graph_path)
    except ValueError as e:
        print(f"ERROR loading data files:\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(engine.startup_summary())
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == 'exit':
            print("Goodbye!")
            break

        if cmd == 'debug':
            engine.debug_mode = not engine.debug_mode
            print(f"Debug mode: {'ON' if engine.debug_mode else 'OFF'}")
            continue

        if cmd == 'showbrain':
            print(engine.show_brain())
            continue

        if cmd == 'profile':
            print(f"Turns: {engine.turn_count}")
            print(f"Memory episodes: {len(engine.memory.episodes)}")
            print(f"Seed: {engine.seed}")
            print(f"Modulators: {engine.modulators}")
            continue

        if cmd.startswith('seed '):
            try:
                new_seed = int(cmd.split()[1])
                engine.set_seed(new_seed)
                print(f"Seed set to {new_seed}")
            except (ValueError, IndexError):
                print("Usage: seed <number>")
            continue

        response, trace, elapsed = engine.process_input(user_input)

        print(f"\nBot: {response}")
        print(trace.format_compact())

        if engine.debug_mode:
            print(trace.format_full())

        print(f"  [{elapsed*1000:.1f}ms]\n")


if __name__ == '__main__':
    main()
