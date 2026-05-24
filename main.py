"""
Main entry point — runs all 6 questions in sequence.

Usage:
    python main.py
    python main.py --part 1          # run only Part 1 (Q1 + Q2)
    python main.py --part 2          # run only Part 2 (Q3 + Q4)
    python main.py --part 3          # run only Part 3 (Q5 + Q6)
    python main.py --question 3      # run only Q3
"""

import argparse
import os

def ensure_dirs():
    for d in ["data", "output/plots", "output/tables", "output/models"]:
        os.makedirs(d, exist_ok=True)


def run_q1():
    print("\n" + "="*60)
    print("PART 1 | Q1: Corpus Construction & Preprocessing")
    print("="*60)
    from src.part1_q1_corpus import run
    return run()

def run_q2():
    print("\n" + "="*60)
    print("PART 1 | Q2: Word2Vec Model Training")
    print("="*60)
    from src.part1_q2_word2vec import run
    return run()

def run_q3():
    print("\n" + "="*60)
    print("PART 2 | Q3: Semantic Proximity Analysis")
    print("="*60)
    from src.part2_q3_similarity import run
    return run()

def run_q4():
    print("\n" + "="*60)
    print("PART 2 | Q4: Thematic Clustering of Word Vectors")
    print("="*60)
    from src.part2_q4_clustering import run
    return run()

def run_q5():
    print("\n" + "="*60)
    print("PART 3 | Q5: Co-occurrence Network Construction")
    print("="*60)
    from src.part3_q5_network import run
    return run()

def run_q6():
    print("\n" + "="*60)
    print("PART 3 | Q6: Centrality Analysis & Community Detection")
    print("="*60)
    from src.part3_q6_centrality import run
    return run()


QUESTION_MAP = {
    1: run_q1, 2: run_q2,
    3: run_q3, 4: run_q4,
    5: run_q5, 6: run_q6,
}

PART_MAP = {
    1: [1, 2],
    2: [3, 4],
    3: [5, 6],
}


if __name__ == "__main__":
    ensure_dirs()

    parser = argparse.ArgumentParser(description="Run Word2Vec & Network Analysis assignment")
    parser.add_argument("--part",     type=int, choices=[1, 2, 3], help="Run a specific part")
    parser.add_argument("--question", type=int, choices=[1,2,3,4,5,6], help="Run a single question")
    args = parser.parse_args()

    if args.question:
        QUESTION_MAP[args.question]()
    elif args.part:
        for q in PART_MAP[args.part]:
            QUESTION_MAP[q]()
    else:
        for q in [1, 2, 3, 4, 5, 6]:
            QUESTION_MAP[q]()

    print("\n" + "="*60)
    print("All outputs saved to output/plots/ and output/tables/")
    print("="*60)
