#!/usr/bin/env python3
"""
Collect Linux kernel security CVEs for the benchmark dataset.

Queries the NVD API v2.0 for Linux kernel CVEs, filters by CVSS >= 7.0,
and focuses on specific subsystems (net/ipv4, crypto, net/ipv6, netfilter).

Usage:
    export NVD_API_KEY="your-api-key"
    python3 scripts/collect_linux_kernel_security.py

Output:
    data/linux_kernel_security.jsonl  (one JSON per line)

Note: This script collects basic CVE data. Manual enrichment with
git patch data and ground truth labels is recommended afterward.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Optional

NVD_API_BASE = "https://api.nist.gov/ntia/rest/event/v2"
CVE_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Target subsystems and their keywords for matching
SUBSYSTEM_KEYWORDS = {
    "net/ipv4": ["tcp", "ipv4", "ip_fragment", "ip_route", "inet_", "sock_", "net/ipv4", "tcp_sock", "udp"],
    "net/ipv6": ["ipv6", "ip6_", "net/ipv6", "inet6"],
    "crypto": ["crypto", "ahash", "akcipher", "skcipher", "shash", "rfc4106", "rfc4309"],
    "netfilter": ["netfilter", "nft_", "xt_", "nf_", "conntrack", "arp_tables", "ip_tables"],
    "fs": ["vfs", "proc_", "sysfs", "tmpfs", "fat_", "nfs", "cifs", "ext4"],
}

# Vulnerability type keywords
VULN_TYPE_KEYWORDS = {
    "use-after-free": ["use-after-free", "uaf", "use after free", "kfree", "slab"],
    "buffer-overflow": ["buffer overflow", "out-of-bounds", "OOB", "overflow", "bounds"],
    "null-ptr-deref": ["null pointer", "null deref", "NULL dereference", "null ptr"],
    "race-condition": ["race condition", "race", "time-of-check", "TOCTOU", "concurrent"],
    "info-leak": ["info leak", "information leak", "kernel info", "stack info", "leak"],
    "refcount": ["refcount", "reference count", "put_", "get_", "ref_count"],
    "integer-overflow": ["integer overflow", "integer underflow", "overflow"],
}


def fetch_cves_from_nvd(api_key: str, start_year: int = 2022, end_year: int = 2025) -> list:
    """Fetch Linux kernel CVEs from NVD API v2.0."""
    all_cves = []

    # Query for Linux kernel CVEs using CPE
    cpe_query = 'cpe:2.3:o:linux:linux_kernel:*:*:*:*:*:*:*:*'

    for year in range(start_year, end_year + 1):
        last_mod_start = f"{year}-01-01T00:00:00.000"
        last_mod_end = f"{year}-12-31T23:59:59.999"

        params = {
            "key": api_key,
            "cpeName": cpe_query,
            "lastModStartDate": last_mod_start,
            "lastModEndDate": last_mod_end,
            "resultsPerPage": 2000,
        }

        # Build query string
        query_str = "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
        url = f"{CVE_API_BASE}?{query_str}"

        print(f"  Fetching CVEs for {year}...")

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "llm-benchmark-runner/1.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            vulns = data.get("vulnerabilities", [])
            print(f"    Found {len(vulns)} vulnerabilities for {year}")
            all_cves.extend(vulns)

            # Rate limiting
            time.sleep(1)

        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"    ERROR: API key invalid or rate limited. Waiting 10s...")
                time.sleep(10)
            elif e.code == 429:
                print(f"    ERROR: Rate limited. Waiting 30s...")
                time.sleep(30)
            else:
                print(f"    HTTP error {e.code}: {e.reason}")
        except Exception as e:
            print(f"    Error fetching {year}: {e}")

    return all_cves


def get_cvss_score(cve_data: dict) -> Optional[float]:
    """Extract CVSS v3.1 score from CVE data."""
    metrics = cve_data.get("metrics", {})

    # Try CVSS v3.1 first
    if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
        return metrics["cvssMetricV31"][0].get("cvssV31", {}).get("baseScore")

    # Fall back to CVSS v3.0
    if "cvssMetricV30" in metrics and metrics["cvssMetricV30"]:
        return metrics["cvssMetricV30"][0].get("cvssV30", {}).get("baseScore")

    # Fall back to CVSS v2.0
    if "cvssMetricV2" in metrics and metrics["cvssMetricV2"]:
        return metrics["cvssMetricV2"][0].get("cvssV2", {}).get("baseScore")

    return None


def match_subsystem(description: str, references: list) -> Optional[str]:
    """Match CVE to a subsystem based on description and references."""
    desc_lower = description.lower()

    for subsystem, keywords in SUBSYSTEM_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in desc_lower:
                return subsystem

    # Check references
    for ref in references:
        ref_url = ref.get("url", "").lower()
        for subsystem, keywords in SUBSYSTEM_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in ref_url:
                    return subsystem

    return None


def infer_vuln_type(description: str, weaknesses: list) -> list:
    """Infer vulnerability types from description and CWE data."""
    desc_lower = description.lower()
    types = []

    # Check CWE IDs
    cwe_ids = set()
    for weakness in weaknesses:
        for desc in weakness.get("description", []):
            if "CWE-" in desc:
                cwe_ids.add(desc.split("CWE-")[1].split(")")[0])

    # CWE to vuln type mapping
    cwe_to_type = {
        "416": "use-after-free",
        "125": "buffer-overflow",
        "787": "buffer-overflow",
        "119": "buffer-overflow",
        "400": "resource-exhaustion",
        "415": "double-free",
        "190": "integer-overflow",
        "362": "race-condition",
        "200": "info-leak",
    }

    for cwe_id, vtype in cwe_to_type.items():
        if cwe_id in cwe_ids:
            types.append(vtype)

    # Also check description keywords
    for vtype, keywords in VULN_TYPE_KEYWORDS.items():
        if vtype not in types:
            for keyword in keywords:
                if keyword in desc_lower:
                    types.append(vtype)
                    break

    return types if types else ["unknown"]


def assign_difficulty(cvss_score: float, vuln_types: list) -> str:
    """Assign difficulty based on CVSS score and vulnerability type."""
    if cvss_score >= 9.0:
        return "hard"
    if any(t in vuln_types for t in ["use-after-free", "race-condition"]):
        return "hard"
    if cvss_score >= 8.0:
        return "medium"
    return "medium"


def main():
    api_key = os.environ.get("NVD_API_KEY")
    if not api_key:
        print("ERROR: NVD_API_KEY environment variable not set.")
        print("Get a free API key at: https://nvd.nist.gov/developers/request-an-api-key")
        sys.exit(1)

    print("Collecting Linux kernel security CVEs...")
    print(f"  API key: {api_key[:8]}...{api_key[-4:]}")

    # Fetch CVEs
    all_cves = fetch_cves_from_nvd(api_key)
    print(f"\nTotal CVEs fetched: {len(all_cves)}")

    # Filter and process
    filtered = []
    subsystem_counts = {}

    for cve_data in all_cves:
        cve_id = cve_data.get("id", "")
        cvss = get_cvss_score(cve_data)

        # Filter by CVSS >= 7.0
        if not cvss or cvss < 7.0:
            continue

        # Extract description
        descriptions = cve_data.get("descriptions", [])
        description = ""
        for desc in descriptions:
            if desc.get("lang") == "en":
                description = desc.get("value", "")
                break

        if not description:
            continue

        # Match subsystem
        references = cve_data.get("references", [])
        subsystem = match_subsystem(description, references)

        # Only include target subsystems
        if subsystem not in SUBSYSTEM_KEYWORDS:
            continue

        # Infer vulnerability type
        weaknesses = []
        if "weaknesses" in cve_data:
            weaknesses = cve_data["weaknesses"]
        vuln_types = infer_vuln_type(description, weaknesses)

        # Assign difficulty
        difficulty = assign_difficulty(cvss, vuln_types)

        # Extract CWE ID
        cwe_id = "CWE-Unknown"
        for weakness in weaknesses:
            for desc in weakness.get("description", []):
                if "CWE-" in desc:
                    cwe_id = desc.split(")")[0].split("(")[-1] if "(" in desc else desc
                    break

        # Get published date
        published = cve_data.get("published", "")
        discovery_year = published[:4] if published else "unknown"

        # Track subsystem counts
        subsystem_counts[subsystem] = subsystem_counts.get(subsystem, 0) + 1

        # Build record
        record = {
            "cve_id": cve_id,
            "cvss_score": cvss,
            "subsystem": subsystem,
            "title": description[:200] if len(description) > 200 else description,
            "description": description,
            "cwe_id": cwe_id,
            "vuln_type": vuln_types[0] if vuln_types else "unknown",
            "vuln_types": vuln_types,
            "affected_function": "",  # To be filled manually
            "affected_file": "",  # To be filled manually
            "base_commit": "",  # To be filled manually
            "fix_commit": "",  # To be filled manually
            "vuln_code_context": "",  # To be filled manually
            "fix_patch": "",  # To be filled manually
            "discovery_year": discovery_year,
            "task_type": "vulnerability_identification",
            "difficulty": difficulty,
            "ground_truth": {
                "has_vulnerability": True,
                "vuln_types": vuln_types,
                "affected_variables": [],  # To be filled manually
                "root_cause_keywords": vuln_types,
            },
        }

        filtered.append(record)

    # Limit to 20-50 CVEs for v1
    max_cves = 50
    if len(filtered) > max_cves:
        # Prioritize by CVSS score
        filtered.sort(key=lambda x: x["cvss_score"], reverse=True)
        filtered = filtered[:max_cves]

    print(f"\nFiltered CVEs (CVSS >= 7.0, target subsystems): {len(filtered)}")
    print(f"Subsystem distribution:")
    for subsystem, count in sorted(subsystem_counts.items()):
        print(f"  {subsystem}: {count}")

    # Write to JSONL
    os.makedirs("data", exist_ok=True)
    output_path = "data/linux_kernel_security.jsonl"
    with open(output_path, "w") as f:
        for record in filtered:
            f.write(json.dumps(record) + "\n")

    print(f"\nDataset written to: {output_path}")
    print(f"Total entries: {len(filtered)}")
    print(f"\nNext steps:")
    print(f"  1. Manually enrich with git patch data (fix_commit, vuln_code_context, fix_patch)")
    print(f"  2. Fill in affected_function, affected_file, and ground_truth fields")
    print(f"  3. Validate: 20-50 CVEs, CVSS >= 7.0, mix of net/ipv4 + crypto")


if __name__ == "__main__":
    main()
