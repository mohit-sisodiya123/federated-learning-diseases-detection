# main.py
"""
Single entry point for the entire project.

Runs the full pipeline:
  Phase 5 experiments  ->  Phase 6 analysis  ->  saved results and plots.
"""

import os


def main():
    print("=" * 70)
    print("Privacy-Preserving Federated Deep Learning for Healthcare")
    print("with Malicious Client Detection")
    print("=" * 70)
    print()
    print("This will run:")
    print("  1. Three experiments (Normal FL, FL + Attack, FL + Detection)")
    print("  2. Analysis, plots, and the results report")
    print()
    print("Output will be saved in the 'results/' folder.")
    print()

    # Phase 5: run the three experiments and produce initial plots/CSV
    from experiments.run_phase5 import main as run_phase5
    run_phase5()

    # Phase 6: re-run for per-round data + generate report
    from experiments.run_phase6 import main as run_phase6
    run_phase6()

    print()
    print("=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print("Results folder : results/")
    print("Plots folder   : results/plots/")
    print("Report         : results/results_analysis.md")
    print()
    print("For the demo, open 'docs/demo_guide.md'.")
    print("For viva prep, open 'docs/viva_questions.md'.")


if __name__ == "__main__":
    main()