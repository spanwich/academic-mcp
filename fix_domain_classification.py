#!/usr/bin/env python3
"""
Domain Classification Fix Script
================================

This script identifies and corrects misclassified papers in the Academic MCP database.
Based on analysis, approximately 30-40% of papers in the top domains are misclassified.

Phase 1: Generate misclassification report
Phase 2: Execute automatic domain reassignments
Phase 3: Verify and summarize changes
"""

import sqlite3
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Database path
DB_PATH = Path("data/papers.db")

# Domain classification rules based on semantic analysis
# Each rule: (patterns_in_title_or_abstract, correct_domain)
CLASSIFICATION_RULES = [
    # ICS/SCADA and Industrial Control Systems
    (
        ["iec 61850", "iec61850", "electric power system", "substation", "smart grid protocol"],
        "ICS/SCADA protocols and power grid communications"
    ),
    (
        ["modbus", "dnp3", "ics protocol", "scada protocol", "industrial protocol"],
        "ICS/SCADA protocols and power grid communications"
    ),
    (
        ["ics anomaly", "industrial control.*anomaly", "scada anomaly", "industrial.*intrusion"],
        "ICS anomaly detection and intrusion prevention"
    ),
    (
        ["ot security", "operational technology security", "ics security", "scada security"],
        "operational technology (OT) security"
    ),
    (
        ["ics testbed", "scada testbed", "industrial testbed"],
        "ICS/SCADA testbed simulation"
    ),

    # Machine Learning (disambiguate from kernel isolation)
    (
        ["support vector machine", "svm", "isolation forest", "anomaly detection.*kernel",
         "kernel.*anomaly", "distributional kernel", "isolation kernel.*svm",
         "kernel-based.*machine learning", "kernel method"],
        "machine learning kernel methods"
    ),
    (
        ["deep learning", "neural network", "cnn", "lstm", "transformer model"],
        "deep learning and neural networks"
    ),
    (
        ["attribution", "explainability", "interpretable ml", "xai"],
        "machine learning explainability"
    ),

    # Cross-Domain Solutions
    (
        ["cross.?domain solution", "cds", "cross domain security", "data diode",
         "guard", "high assurance.*domain"],
        "cross-domain security solutions"
    ),

    # Protocol Analysis
    (
        ["protocol reverse", "protocol analysis", "protocol extraction", "message format"],
        "protocol reverse engineering and analysis"
    ),

    # Vehicle and Automotive Security
    (
        ["in-vehicle", "can bus", "automotive.*intrusion", "vehicle network",
         "controller area network", "obd", "automotive security"],
        "automotive and in-vehicle network security"
    ),

    # Sensor and CPS Attacks
    (
        ["sensor attack", "cusum", "chi-squared.*attack", "false data injection",
         "sensor compromise"],
        "cyber-physical systems sensor security"
    ),

    # Hypervisor/Virtualization (not microkernel verification)
    (
        ["xen", "hypervisor.*comparison", "kvm", "vmware", "virtual machine monitor",
         "hardware.*virtualization", "guest.*virtualization"],
        "virtualization and hypervisor systems"
    ),

    # Firmware
    (
        ["firmware.*vulnerab", "firmware.*corpus", "firmware analysis", "firmware testing",
         "embedded firmware", "iot firmware"],
        "firmware security and vulnerability research"
    ),
    (
        ["in-vivo.*firmware", "stateless instrumentation", "firmware testing"],
        "firmware testing and instrumentation"
    ),

    # TEE and Secure Enclaves
    (
        ["trusted execution environment", "tee", "trustzone", "tz-", "sgx.*enclave",
         "arm trustzone", "secure enclave"],
        "trusted execution environments"
    ),

    # Digital Twins and Cloud
    (
        ["digital twin", "kubernetes", "okd", "container orchestration"],
        "digital twins and cloud orchestration"
    ),

    # Coding Standards
    (
        ["power of 10", "coding rule", "coding standard", "misra", "safety-critical code",
         "coding guideline"],
        "safety-critical coding standards"
    ),

    # Data Infrastructure
    (
        ["national data infrastructure", "data policy", "data governance",
         "data sharing framework"],
        "data infrastructure and governance"
    ),

    # IOMMU and Memory
    (
        ["iommu", "dma.*protection", "device passthrough"],
        "IOMMU and device isolation"
    ),

    # GPU Security
    (
        ["gpu.*isolation", "gpu.*security", "graphics.*security", "cuda.*security"],
        "GPU security and isolation"
    ),

    # Program Slicing (narrow scope)
    (
        ["program slicing", "backward slice", "forward slice", "slicing criterion"],
        "program slicing and context-aware program repair"
    ),

    # Real-time specifically
    (
        ["wcet", "worst-case execution", "real-time scheduling", "task scheduling",
         "deadline.*scheduling"],
        "real-time systems scheduling"
    ),

    # Mixed-criticality specifically
    (
        ["mixed.?criticality", "criticality level", "dual.?criticality",
         "safety.?critical.*real-time"],
        "mixed-criticality scheduling with temporal isolation"
    ),

    # seL4 and microkernels verification
    (
        ["sel4.*verification", "sel4.*proof", "isabelle.*microkernel",
         "formal verification.*kernel", "kernel.*isabelle", "coq.*kernel"],
        "microkernel formal verification using Isabelle/HOL"
    ),

    # CHERI and capability systems
    (
        ["cheri", "capability.*hardware", "capability.*memory safety"],
        "capability hardware for memory safety"
    ),

    # Verified parsers
    (
        ["verified.*parser", "zero-copy parser", "everparse", "message format.*verif"],
        "verified secure parsers"
    ),

    # Timed automata
    (
        ["timed automata", "uppaal", "duration calculus", "plc-automata",
         "real-time.*automata"],
        "timed automata and real-time verification"
    ),

    # Static analysis
    (
        ["static verification", "static analysis", "module verification"],
        "static verification of operating systems"
    ),

    # Separation kernels
    (
        ["separation kernel", "mils", "multiple independent levels"],
        "separation kernel and MILS architecture"
    ),

    # L4 family
    (
        ["l4 microkernel", "l4 family", "okl4", "fiasco"],
        "L4 microkernel family"
    ),

    # Rust verification
    (
        ["rust.*verif", "verus", "linear ghost", "prusti"],
        "Rust program verification"
    ),

    # CAmkES framework
    (
        ["camkes", "component.*model.*embedded", "seL4.*component"],
        "CAmkES component architecture"
    ),

    # Linux drivers verification
    (
        ["linux driver", "device driver.*verif", "pancake", "driver.*verification"],
        "verified device drivers"
    ),

    # Kernel compartmentalization
    (
        ["kernel.*compartment", "kernel.*isolation", "subsystem.*isolation",
         "lxd", "intra-kernel"],
        "kernel compartmentalization"
    ),

    # MPU isolation
    (
        ["mpu.*isolation", "memory protection unit", "mpu-based"],
        "MPU-based embedded isolation"
    ),

    # Robotics
    (
        ["underactuated", "robotics", "needle steering", "autonomous.*robot"],
        "underactuated robotics"
    ),

    # Network segmentation
    (
        ["network segment", "network isolation", "vlan", "micro-segment"],
        "network segmentation"
    ),

    # Side-channel attacks
    (
        ["side.?channel", "acoustic.*attack", "spectr", "meltdown", "timing attack"],
        "side-channel attacks"
    ),

    # Speculation attacks
    (
        ["speculation.*attack", "speculative execution", "transient execution"],
        "speculative execution security"
    ),

    # RTOS
    (
        ["rtos", "freertos", "real-time operating system", "embedded.*os"],
        "real-time operating systems"
    ),

    # Partitioning hypervisors
    (
        ["partitioning hypervisor", "static partitioning", "jailhouse", "xtratum",
         "bao hypervisor"],
        "partitioning hypervisors for embedded systems"
    ),

    # Software-defined vehicles
    (
        ["software-defined vehicle", "sdv", "vehicle.*heterogeneous"],
        "software-defined vehicles"
    ),

    # Message-passing protocols
    (
        ["message.?passing.*protocol", "session type", "timed.*protocol"],
        "timed message-passing protocols"
    ),

    # Heterogeneous computing
    (
        ["heterogeneous.*computing", "heterogeneous.*platform", "mpsoc",
         "system-on-chip", "fpga.*cpu"],
        "heterogeneous multiprocessor systems"
    ),

    # Model checking and formal methods general
    (
        ["model checking", "reachability.*checking", "safety verification",
         "temporal logic"],
        "model checking and formal verification"
    ),

    # Loop bounds and WCET analysis
    (
        ["loop bound", "infeasible path", "wcet analysis"],
        "WCET analysis and loop bound determination"
    ),

    # Big data and distributed systems
    (
        ["big data", "distributed.*asymmetric", "mapreduce"],
        "big data processing systems"
    ),

    # RISC-V TEE
    (
        ["risc-v.*tee", "risc-v.*trust"],
        "RISC-V trusted execution"
    ),

    # Model-driven development
    (
        ["model-driven", "hamr", "aadl", "architecture analysis"],
        "model-driven development for embedded systems"
    ),

    # Concurrent verification
    (
        ["concurrent.*verif", "parallel.*verif", "thread.*safety"],
        "concurrent program verification"
    ),
]

# Papers that need manual classification based on specific knowledge
MANUAL_CLASSIFICATIONS = {
    # Machine Learning papers misclassified as kernel isolation
    "ting_2018_isolationkernelits": "machine learning kernel methods",
    "ting_2020_isolationdistributionalkernel": "machine learning kernel methods",
    "fung2024attributions": "ICS anomaly detection and intrusion prevention",

    # Cross-domain solutions
    "__fundamentalscrossdomain": "cross-domain security solutions",
    "__introductioncrossdomain": "cross-domain security solutions",

    # ICS/SCADA
    "peterbishop_2023_iec61850principles": "ICS/SCADA protocols and power grid communications",
    "stouffer_2023_guideoperationaltechnology": "operational technology (OT) security",

    # Vehicle security
    "lee_2017_otidsnovelintrusion": "automotive and in-vehicle network security",

    # Sensor attacks
    "murguia_2016_cusumchisquaredattack": "cyber-physical systems sensor security",

    # Protocol analysis
    "luo2024dynpre": "protocol reverse engineering and analysis",

    # Data infrastructure
    "sadlier__nationaldatainfrastructure": "data infrastructure and governance",

    # Virtualization
    "barham_2003_xenartvirtualization": "virtualization and hypervisor systems",

    # Firmware
    "helmke2025firmware": "firmware security and vulnerability research",
    "shi2024ipea": "firmware testing and instrumentation",

    # Coding standards
    "holzmann_2006_power10rules": "safety-critical coding standards",

    # Digital twins
    "ribeiro_2023_digitaltwinmigration": "digital twins and cloud orchestration",

    # TEE
    "han2023mytee": "trusted execution environments",
    "kong2025tzdatashield": "trusted execution environments",

    # IOMMU
    "ben-yehuda_2007_pricesafetyevaluating": "IOMMU and device isolation",

    # GPU
    "yadlapalli_2022_gguardenablingleakageresilient": "GPU security and isolation",

    # Hypervisor comparisons
    "hwang_2013_componentbasedperformancecomparison": "virtualization and hypervisor systems",

    # CHERI
    "amar_2023_cheriotcompletememory": "capability hardware for memory safety",
    "amar_2025_cheriotrtosos": "capability hardware for memory safety",
    "fuchs_2024_safespeculationcheri": "capability hardware for memory safety",

    # Cloud/virtualization security
    "asvija_2019_securityhardwareassisted": "virtualization and hypervisor systems",

    # Real-time verification
    "kim_2025_verirtendtoendverification": "real-time distributed systems verification",

    # Embedded virtualization
    "bock_2020_realtimevirtualizationxvisor": "partitioning hypervisors for embedded systems",

    # Timed automata
    "dewulf_2008_robustsafetytimed": "timed automata and real-time verification",
    "chaochen_2004_durationcalculus": "timed automata and real-time verification",

    # Big data
    "el-rouby_2021_nileosdistributedasymmetric": "big data processing systems",

    # Verified parsers
    "ramananandro_2019_everparseverifiedsecure": "verified secure parsers",

    # seL4 specific
    "heiser_2016_l4microkernels": "L4 microkernel family",
    "heiser_2020_sel4australia": "seL4 formal verification and deployment",
    "klein_2018_formallyverifiedsoftware": "seL4 formal verification and deployment",
    "t.murray_2013_sel4generalpurpose": "seL4 information flow verification",
    "nunes_2023_parselverifiedrootoftrust": "seL4 trusted computing",
    "parker_2023_highperformancenetworkingsel4": "seL4 networking",
    "belt_2023_modeldrivendevelopmentsel4": "model-driven development for embedded systems",
    "dematos_2024_sel4basedtrustedexecution": "RISC-V trusted execution",

    # Rust
    "lattuada_2023_verusverifyingrust": "Rust program verification",

    # CAmkES
    "kuz_2007_camkescomponentmodel": "CAmkES component architecture",
    "sudvarg_2022_concurrencyframeworkpriorityaware": "CAmkES component architecture",

    # Kernel isolation
    "narayanan_2019_lxdsisolationkernel": "kernel compartmentalization",
    "narayanan_2020_lightweightkernelisolation": "kernel compartmentalization",
    "mckee_2022_preventingkernelhacks": "kernel compartmentalization",
    "khan_2023_ecembeddedsystems": "kernel compartmentalization",

    # Linux drivers
    "chen_2024_veldverifiedlinux": "verified device drivers",
    "zhao__verifyingdevicedrivers": "verified device drivers",

    # Static verification
    "zakharov_2015_configurabletoolsetstatic": "static verification of operating systems",
    "starostin_2008_correctmicrokernelprimitives": "microkernel primitives verification",

    # Loop bounds
    "sewell_2016_completehighassurancedetermination": "WCET analysis and loop bound determination",

    # Partitioning hypervisors
    "martins_2023_sheddinglightstatic": "partitioning hypervisors for embedded systems",
    "shen_2022_shyperembeddedhypervisor": "partitioning hypervisors for embedded systems",
    "lavernelle_2024_assessmentspatialisolation": "partitioning hypervisors for embedded systems",
    "crespo_2010_partitionedembeddedarchitecture": "partitioning hypervisors for embedded systems",
    "west_2016_virtualizedseparationkernel": "separation kernel and MILS architecture",
    "wulf_2022_virtualizationreconfigurablemixedcriticality": "partitioning hypervisors for embedded systems",

    # MPU isolation
    "sensaoui_2019_indepthstudympubased": "MPU-based embedded isolation",

    # Heterogeneous platforms
    "harris_2025_performancemodelingnonuniform": "heterogeneous multiprocessor systems",
    "gracioli_2019_designingmixedcriticality": "heterogeneous multiprocessor systems",
    "fong_1993_heterogeneousmultiprocessingcomputer": "heterogeneous multiprocessor systems",
    "xu_2025_bettertogetherinterferenceawareframework": "heterogeneous multiprocessor systems",
    "kloda_2023_lazyloadscheduling": "heterogeneous multiprocessor systems",
    "valente_2025_newhwsw": "heterogeneous multiprocessor systems",
    "minanda_2025_modifiedfirmwarebasedboot": "software-defined vehicles",

    # Timed automata
    "alur_1994_theorytimedautomata": "timed automata and real-time verification",
    "bengtsson_1996_uppaalatoolsuite": "timed automata and real-time verification",
    "dierks_2001_plcautomatanewclass": "timed automata and real-time verification",
    "lehmann_2025_modeltemplatereachabilitybased": "timed automata and real-time verification",
    "lehmann_2025_provablysafecontroller": "timed automata and real-time verification",
    "yao_2025_semanticlogicalrelations": "timed message-passing protocols",

    # Virtualization in embedded/real-time
    "alonso_2022_influencevirtualizationrealtime": "partitioning hypervisors for embedded systems",
    "cinque_2022_virtualizingmixedcriticalitysystems": "partitioning hypervisors for embedded systems",
    "queiroz_2023_testinglimitsgeneralpurpose": "partitioning hypervisors for embedded systems",
    "hughes_2019_quantifyingperformancedeterminism": "partitioning hypervisors for embedded systems",
    "asmussen_2025_distrustingcoresseparating": "hardware-based isolation",

    # Real-time scheduling
    "lee_2023_estimatingprobabilisticsafe": "real-time systems scheduling",
    "l.xu_2016_faulttolerantrealtimescheduling": "real-time systems scheduling",
    "willcock_2025_fullypolynomialtime": "real-time systems scheduling",
    "edward_2018_efficienttaskscheduling": "heterogeneous multiprocessor scheduling",
    "wang_2025_securitydriventaskscheduling": "heterogeneous multiprocessor scheduling",
    "d.ma_2024_componentbasedmixedcriticalityrealtime": "mixed-criticality scheduling with temporal isolation",
    "zhang_2025_mixedcriticalitydagsscheduling": "mixed-criticality scheduling with temporal isolation",
}


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def classify_paper(paper_id: str, title: str, abstract: str) -> Optional[str]:
    """
    Determine the correct domain for a paper based on title and abstract.
    Returns None if no confident classification can be made.
    """
    # Check manual classifications first
    if paper_id in MANUAL_CLASSIFICATIONS:
        return MANUAL_CLASSIFICATIONS[paper_id]

    # Combine title and abstract for matching
    text = f"{title or ''} {abstract or ''}".lower()

    # Check each rule
    for patterns, domain in CLASSIFICATION_RULES:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return domain

    return None


def analyze_papers() -> dict:
    """
    Analyze all papers and identify misclassifications.
    Returns a dict with analysis results.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get all papers with their current domains
    cursor.execute("""
        SELECT paper_id, title, abstract, domain
        FROM papers
        WHERE domain IS NOT NULL
        ORDER BY domain, paper_id
    """)

    papers = cursor.fetchall()

    results = {
        "total_papers": len(papers),
        "misclassified": [],
        "correct": [],
        "uncertain": [],
        "by_current_domain": defaultdict(list),
        "by_new_domain": defaultdict(list),
    }

    for paper_id, title, abstract, current_domain in papers:
        suggested_domain = classify_paper(paper_id, title or "", abstract or "")

        if suggested_domain is None:
            results["uncertain"].append({
                "paper_id": paper_id,
                "title": title,
                "current_domain": current_domain,
            })
        elif suggested_domain.lower() != current_domain.lower():
            results["misclassified"].append({
                "paper_id": paper_id,
                "title": title,
                "current_domain": current_domain,
                "suggested_domain": suggested_domain,
                "abstract_preview": (abstract or "")[:200] + "..." if abstract else None
            })
            results["by_current_domain"][current_domain].append(paper_id)
            results["by_new_domain"][suggested_domain].append(paper_id)
        else:
            results["correct"].append({
                "paper_id": paper_id,
                "title": title,
                "domain": current_domain,
            })

    conn.close()
    return results


def generate_report(results: dict) -> str:
    """Generate a detailed misclassification report."""
    lines = [
        "=" * 80,
        "DOMAIN MISCLASSIFICATION ANALYSIS REPORT",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 80,
        "",
        "SUMMARY",
        "-" * 40,
        f"Total papers analyzed: {results['total_papers']}",
        f"Misclassified papers: {len(results['misclassified'])}",
        f"Correctly classified: {len(results['correct'])}",
        f"Uncertain (no rule match): {len(results['uncertain'])}",
        f"Misclassification rate: {len(results['misclassified']) / results['total_papers'] * 100:.1f}%",
        "",
        "=" * 80,
        "MISCLASSIFIED PAPERS BY CURRENT DOMAIN",
        "=" * 80,
        "",
    ]

    # Group by current domain
    by_domain = defaultdict(list)
    for paper in results["misclassified"]:
        by_domain[paper["current_domain"]].append(paper)

    for domain in sorted(by_domain.keys(), key=lambda d: len(by_domain[d]), reverse=True):
        papers = by_domain[domain]
        lines.append(f"\n### {domain}")
        lines.append(f"    {len(papers)} papers need reassignment\n")

        for paper in papers:
            lines.append(f"    Paper ID: {paper['paper_id']}")
            lines.append(f"    Title: {paper['title']}")
            lines.append(f"    -> Move to: {paper['suggested_domain']}")
            lines.append("")

    lines.extend([
        "",
        "=" * 80,
        "NEW DOMAIN DESTINATIONS",
        "=" * 80,
        "",
    ])

    for domain in sorted(results["by_new_domain"].keys()):
        papers = results["by_new_domain"][domain]
        lines.append(f"  {domain}: {len(papers)} papers incoming")

    lines.extend([
        "",
        "=" * 80,
        "DETAILED REASSIGNMENT LIST",
        "=" * 80,
        "",
    ])

    for paper in sorted(results["misclassified"], key=lambda p: p["paper_id"]):
        lines.append(f"- {paper['paper_id']}")
        lines.append(f"  Title: {paper['title']}")
        lines.append(f"  FROM: {paper['current_domain']}")
        lines.append(f"  TO:   {paper['suggested_domain']}")
        lines.append("")

    return "\n".join(lines)


def execute_reassignments(results: dict, dry_run: bool = False) -> dict:
    """
    Execute the domain reassignments in the database.
    Returns a summary of changes made.
    """
    conn = get_connection()
    cursor = conn.cursor()

    changes = {
        "papers_updated": 0,
        "domains_created": [],
        "domains_updated": [],
        "errors": [],
    }

    # Get existing domains
    cursor.execute("SELECT name FROM domains")
    existing_domains = {row[0].lower(): row[0] for row in cursor.fetchall()}

    for paper in results["misclassified"]:
        paper_id = paper["paper_id"]
        old_domain = paper["current_domain"]
        new_domain = paper["suggested_domain"]

        try:
            # Check if new domain exists (case-insensitive)
            if new_domain.lower() not in existing_domains:
                if not dry_run:
                    cursor.execute(
                        "INSERT INTO domains (name, paper_count, created_at) VALUES (?, 0, ?)",
                        (new_domain, datetime.now().isoformat())
                    )
                existing_domains[new_domain.lower()] = new_domain
                changes["domains_created"].append(new_domain)
            else:
                # Use existing domain name (preserve case)
                new_domain = existing_domains[new_domain.lower()]

            if not dry_run:
                # Update paper
                cursor.execute(
                    "UPDATE papers SET domain = ? WHERE paper_id = ?",
                    (new_domain, paper_id)
                )

                # Update old domain count
                cursor.execute(
                    "UPDATE domains SET paper_count = paper_count - 1 WHERE name = ?",
                    (old_domain,)
                )

                # Update new domain count
                cursor.execute(
                    "UPDATE domains SET paper_count = paper_count + 1 WHERE name = ?",
                    (new_domain,)
                )

            changes["papers_updated"] += 1
            changes["domains_updated"].append((old_domain, new_domain))

        except Exception as e:
            changes["errors"].append(f"{paper_id}: {str(e)}")

    if not dry_run:
        # Clean up domains with 0 papers
        cursor.execute("DELETE FROM domains WHERE paper_count <= 0")
        deleted = cursor.rowcount
        if deleted > 0:
            changes["domains_deleted"] = deleted

        conn.commit()

    conn.close()
    return changes


def verify_changes() -> str:
    """Verify the changes and generate a summary."""
    conn = get_connection()
    cursor = conn.cursor()

    lines = [
        "",
        "=" * 80,
        "POST-REASSIGNMENT VERIFICATION",
        "=" * 80,
        "",
        "Current Domain Distribution:",
        "-" * 40,
    ]

    cursor.execute("""
        SELECT name, paper_count
        FROM domains
        ORDER BY paper_count DESC
    """)

    for name, count in cursor.fetchall():
        lines.append(f"  {count:3d} papers: {name}")

    cursor.execute("SELECT COUNT(*) FROM papers")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM domains")
    domain_count = cursor.fetchone()[0]

    lines.extend([
        "",
        f"Total papers: {total}",
        f"Total domains: {domain_count}",
    ])

    conn.close()
    return "\n".join(lines)


def main():
    """Main execution function."""
    print("=" * 80)
    print("DOMAIN CLASSIFICATION FIX SCRIPT")
    print("=" * 80)

    # Phase 1: Analysis
    print("\n[Phase 1] Analyzing papers for misclassifications...")
    results = analyze_papers()

    # Generate and save report
    report = generate_report(results)
    report_path = Path("domain_misclassification_report.txt")
    report_path.write_text(report)
    print(f"Report saved to: {report_path}")
    print(f"\nFound {len(results['misclassified'])} misclassified papers out of {results['total_papers']}")

    # Phase 2: Execute reassignments
    print("\n[Phase 2] Executing domain reassignments...")
    changes = execute_reassignments(results, dry_run=False)

    print(f"  Papers updated: {changes['papers_updated']}")
    print(f"  New domains created: {len(changes['domains_created'])}")
    for domain in changes["domains_created"]:
        print(f"    - {domain}")
    if changes.get("domains_deleted"):
        print(f"  Empty domains cleaned up: {changes['domains_deleted']}")
    if changes["errors"]:
        print(f"  Errors: {len(changes['errors'])}")
        for error in changes["errors"]:
            print(f"    - {error}")

    # Phase 3: Verification
    print("\n[Phase 3] Verifying changes...")
    verification = verify_changes()
    print(verification)

    # Append verification to report
    with open(report_path, "a") as f:
        f.write(verification)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
