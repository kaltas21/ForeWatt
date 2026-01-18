#!/usr/bin/env python3
"""
ForeWatt Technical Report Generator
====================================

Master script that combines all section scripts to generate the complete
ForeWatt Technical Report as a Word document (.docx).

Usage:
    python generate_report.py

Output:
    ForeWatt_Technical_Report.docx

Author: ForeWatt Team
Date: January 2026
"""

import os
import sys
from datetime import datetime

# Add Report directory to path for imports
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPORT_DIR)

from utils_docx import (
    create_document,
    add_title_page,
    add_page_break,
    save_section,
    FigureCounter,
    TableCounter
)

# Import section generators
from section_intro import generate_section as generate_intro
from section_data import generate_section as generate_data
from section_features import generate_section as generate_features
from section_model_selection import generate_section as generate_model_selection
from section_experiments import generate_section as generate_experiments
from section_system import generate_section as generate_system
from section_closing import generate_section as generate_closing


def generate_full_report():
    """
    Generate the complete ForeWatt Technical Report.

    Combines all sections into a single Word document with proper
    formatting, figure numbering, and table numbering.

    Returns:
        str: Path to the generated report file
    """
    print("=" * 60)
    print("ForeWatt Technical Report Generator")
    print("=" * 60)
    print()

    # Create the main document
    print("Creating document...")
    doc = create_document()

    # Initialize counters for figures and tables
    figure_counter = FigureCounter(start=1)
    table_counter = TableCounter(start=1)

    # =========================================================================
    # Title Page
    # =========================================================================
    print("Adding title page...")
    add_title_page(
        doc,
        title="ForeWatt Technical Report",
        subtitle="Electricity Market Forecasting Platform for the Turkish Market",
        authors=["ForeWatt Development Team"],
        date=datetime.now().strftime("%B %Y")
    )
    add_page_break(doc)

    # =========================================================================
    # Abstract and Introduction (Section 1)
    # =========================================================================
    print("Generating Section 1: Abstract and Introduction...")
    generate_intro(doc)
    add_page_break(doc)

    # =========================================================================
    # Section 2: System Design
    # =========================================================================

    # 2.1 Data Collection
    print("Generating Section 2.1: Data Collection...")
    doc = generate_data(doc, figure_counter)
    add_page_break(doc)

    # 2.2 Feature Engineering
    print("Generating Section 2.2: Feature Engineering...")
    doc, figure_counter, table_counter = generate_features(doc, figure_counter, table_counter)
    add_page_break(doc)

    # 2.3 Model Selection
    print("Generating Section 2.3: Model Selection...")
    doc, table_counter = generate_model_selection(doc, table_counter)
    add_page_break(doc)

    # 2.4 Experiments and Results
    print("Generating Section 2.4: Experiments and Results...")
    doc, figure_counter, table_counter = generate_experiments(doc, figure_counter, table_counter)
    add_page_break(doc)

    # 2.5 System Architecture and 2.6 Frontend
    print("Generating Sections 2.5-2.6: System Architecture and Frontend...")
    doc = generate_system(doc, figure_counter)
    add_page_break(doc)

    # =========================================================================
    # Section 3: Conclusions, References, and Appendix
    # =========================================================================
    print("Generating Section 3: Conclusions, References, and Appendix...")
    generate_closing(doc)

    # =========================================================================
    # Save the document
    # =========================================================================
    output_filename = "ForeWatt_Technical_Report.docx"
    output_path = save_section(doc, output_filename)

    print()
    print("=" * 60)
    print("Report generation complete!")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"Total figures: {figure_counter.current() - 1}")
    print(f"Total tables: {table_counter.current() - 1}")
    print()

    return output_path


def generate_individual_sections():
    """
    Generate each section as a separate document for review.

    Useful for testing and reviewing individual sections before
    combining into the full report.
    """
    print("Generating individual section documents...")
    print()

    sections = [
        ("section_intro.py", "Section 1: Abstract and Introduction"),
        ("section_data.py", "Section 2.1: Data Collection"),
        ("section_features.py", "Section 2.2: Feature Engineering"),
        ("section_model_selection.py", "Section 2.3: Model Selection"),
        ("section_experiments.py", "Section 2.4: Experiments and Results"),
        ("section_system.py", "Sections 2.5-2.6: System Architecture and Frontend"),
        ("section_closing.py", "Section 3: Conclusions, References, Appendix"),
    ]

    for script, description in sections:
        script_path = os.path.join(REPORT_DIR, script)
        if os.path.exists(script_path):
            print(f"  - {description}")
            os.system(f"python {script_path}")
        else:
            print(f"  - [MISSING] {description}: {script}")

    print()
    print("Individual sections generated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ForeWatt Technical Report"
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Generate individual section documents instead of full report"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Generate both individual sections and full report"
    )

    args = parser.parse_args()

    if args.both:
        generate_individual_sections()
        print()
        generate_full_report()
    elif args.individual:
        generate_individual_sections()
    else:
        generate_full_report()
