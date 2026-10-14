"""Generate the distractor corpus: realistic look-alike documents that make retrieval hard.

The 23 gold documents alone are too easy to search (recall@5 ~0.95), so changes
to retrieval would show no effect. These ~100 seeded documents add hard
negatives: other teams, projects, APIs and incidents written in the same
sentence patterns, look-alike codes (B-4012 vs E-4012, LAP-14-STD vs
LAP-14-PRO), meeting notes that mention gold entities in passing, and plain
noise (job posts, case studies).

Rules, enforced below: no distractor may answer an unanswerable eval question
or contradict a gold fact, so every gold label stays correct.

    python eval/make_distractors.py      # rewrites eval/corpus/distractors/ deterministically
"""
from __future__ import annotations

import random
import re
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent / "corpus" / "distractors"
rng = random.Random(20261014)

# Anything that would make an unanswerable question answerable, or clash with gold facts.
FORBIDDEN = [
    r"\bMadrid\b", r"\bParis\b", r"E-9999", r"\bv3\b", r"APAC", r"\bEcho\b", r"INC-2024-099", r"VND-0500",
    r"Premium health", r"bereavement", r"Wi-?Fi", r"\b3\.0\b", r"\b2026 revenue", r"mobile phone",
    r"dental", r"parking", r"\bpets?\b", r"bonus", r"cafeteria", r"hackathon", r"ticker", r"Japan",
    r"120 requests", r"INC-2024-031", r"INC-2024-058", r"VND-0042", r"VND-0117", r"VND-0230",
    r"BUG-1893", r"CVE-2025-1182", r"\bE-4012\b", r"\bE-4021\b", r"\bE-4102\b", r"\bE-5003\b",
    r"\bE-5030\b", r"\bE-6007\b", r"\bE-7101\b", r"\bE-7110\b", r"LAP-14-PRO", r"LAP-13-AIR",
    r"MON-27-4K", r"MON-24-FHD", r"DCK-USB-C", r"HDS-BT-200", r"ERR_ORDER", r"ERR_TOKEN_EXPIRED",
    r"Orders API", r"X-Api-Token", r"programming language", r"total headcount", r"employees in total",
    r"is led by (Priya Shah|Tom Becker|Amara Okafor|Daniel Kim|Nadia Ivanova)",
    r"(Data Platform|Mobile|Payments|Security|People Systems) team (is led|is based|owns)",
    r"owned by the (Data Platform|Mobile|Payments|Security|People Systems) team",
]

FIRST = ["Lukas", "Mira", "Omar", "Hannah", "Ravi", "Chloe", "Felix", "Ines", "Jonas", "Leila", "Mateo",
         "Nora", "Pavel", "Rosa", "Samir", "Tara", "Viktor", "Yara", "Elias", "Greta", "Hugo", "Ida",
         "Karim", "Lina", "Marek", "Nadine", "Oskar", "Priscila", "Quentin", "Rahel"]
LAST = ["Weber", "Costa", "Haddad", "Fischer", "Iyer", "Martin", "Braun", "Pereira", "Novak", "Ahmadi",
        "Ruiz", "Lindqvist", "Horvat", "Santos", "Khalil", "Moreau", "Petrov", "Nakamura", "Berg",
        "Keller", "Dubois", "Lund", "Farah", "Wolff", "Kowalski", "Neumann", "Okoro", "Varga", "Ziegler", "Ortiz"]
OFFICES = ["Berlin", "Lisbon", "London", "Porto", "Munich", "Dublin"]
TEAMS = ["Search", "Infrastructure", "Identity", "Billing Operations", "Growth", "Data Science",
         "Developer Experience", "Customer Success", "Reliability", "Localization", "Integrations",
         "Notifications", "Fraud Detection", "Design Systems", "Quality Engineering", "Partner Portal",
         "Warehouse Systems", "Mobile Platform", "Payments Risk", "Security Operations Center",
         "Internal Tools", "Machine Learning", "Accessibility", "Documentation", "Field Engineering"]
PROJECTS = ["Falcon", "Granite", "Harbor", "Iris", "Juniper", "Kestrel", "Lumen", "Maple", "Nimbus",
            "Onyx", "Pioneer", "Quarry", "Ridge", "Summit", "Tundra", "Umbra", "Vertex", "Willow",
            "Zephyr", "Aurora", "Basalt", "Comet", "Drift", "Ember"]
SYSTEMS = [("Billing Service", "B"), ("Warehouse Scanner", "W"), ("Badge Access", "A"),
           ("Video Conferencing", "V"), ("HR Portal", "H"), ("Expense App", "X")]
PROBLEMS = ["the session cookie is missing", "the request payload is too large", "the device clock is out of sync",
            "the account has no license assigned", "the upstream service did not answer in time",
            "the file format is not supported", "the user lacks the required role",
            "the local cache is corrupted", "the configuration file is missing a value",
            "the network proxy rejected the connection", "the record was modified by someone else",
            "the barcode could not be read", "the badge has been deactivated", "the meeting room is double-booked"]
FIXES = ["Sign out and sign in again.", "Reduce the size and retry.", "Enable automatic time sync.",
         "Ask your manager to assign a license.", "Retry after a few minutes.", "Convert the file and upload it again.",
         "Request the role in the access portal.", "Clear the application cache.", "Restore the default configuration.",
         "Contact the network team.", "Reload the record and repeat the change.", "Clean the scanner window.",
         "Visit reception to reactivate the badge.", "Pick another room."]
APIS = ["Customers", "Invoices", "Shipments", "Catalog", "Inventory", "Reports"]


def person() -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def team_docs() -> dict[str, str]:
    docs, teams = {}, TEAMS[:]
    for i in range(5):
        parts = []
        for t in teams[i * 5:(i + 1) * 5]:
            parts.append(f"# {t} Team\n\nThe {t} team is led by {person()}. The {t} team is based in the "
                         f"{rng.choice(OFFICES)} office. The team has {rng.randint(4, 22)} members and meets "
                         f"every {rng.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday'])} for planning.")
        docs[f"team_directory_{i + 1}.md"] = "\n\n".join(parts)
    return docs


def project_docs() -> dict[str, str]:
    docs = {}
    for i in range(6):
        parts = []
        for p in PROJECTS[i * 4:(i + 1) * 4]:
            owner = rng.choice(TEAMS)
            budget = rng.choice(["150,000", "220,000", "380,000", "640,000", "910,000", "1.5 million"])
            quarter = f"Q{rng.randint(1, 4)} {rng.choice([2025, 2026])}"
            what = rng.choice(["a new internal search index", "the migration of build pipelines",
                               "a redesign of the partner onboarding flow", "automated invoice reconciliation",
                               "a fraud scoring service", "a new design token library",
                               "the consolidation of monitoring dashboards", "a translation memory service"])
            parts.append(f"# Project {p}\n\nProject {p} is {what}. Project {p} is owned by the {owner} team. "
                         f"Project {p} has a budget of {budget} USD and is scheduled to launch in {quarter}.")
        docs[f"project_portfolio_{i + 1}.md"] = "\n\n".join(parts)
    return docs


def error_docs() -> dict[str, str]:
    docs = {}
    for system, prefix in SYSTEMS:
        parts = []
        for block in ("Connection", "Account", "Data"):
            lines = []
            for _ in range(3):
                code = f"{prefix}-{rng.choice([4012, 4021, 4102, 5003, 5030, 6007, 7101, 7110, 4120, 5300])}"
                k = rng.randrange(len(PROBLEMS))
                lines.append(f"Error {code} in the {system} means {PROBLEMS[k]}. {FIXES[k]}")
            parts.append(f"# {system} {block} Errors\n\n" + "\n\n".join(dict.fromkeys(lines)))
        docs[f"errors_{prefix.lower()}_{system.lower().replace(' ', '_')}.md"] = "\n\n".join(parts)
    return docs


def api_docs() -> dict[str, str]:
    docs = {}
    for name in APIS:
        limit = rng.choice([60, 90, 240, 300, 450, 900])
        docs[f"api_{name.lower()}.md"] = (
            f"# {name} API v2\n\nThe {name} API v2 base URL is https://api.example.com/{name.lower()}. "
            f"Requests are authenticated with an OAuth bearer token.\n\n"
            f"# {name} API Rate Limits\n\nThe {name} API v2 allows {limit} requests per minute per client. "
            f"When the limit is exceeded, the API returns HTTP 429.\n\n"
            f"# {name} API Webhooks\n\nWebhook deliveries for the {name} API are retried up to "
            f"{rng.choice([3, 4, 5, 8])} times over {rng.choice([6, 12, 48])} hours.")
    return docs


def incident_docs() -> dict[str, str]:
    docs = {}
    used = {31, 58, 99}
    for _ in range(12):
        n = rng.choice([x for x in range(10, 98) if x not in used])
        used.add(n)
        year = rng.choice([2023, 2024])
        iid = f"INC-{year}-{n:03d}"
        sev = rng.choice(["SEV2", "SEV3"])
        cause = rng.choice(["a misconfigured load balancer", "a full disk on a database replica",
                            "an expired API token in a nightly job", "a faulty network switch",
                            "a memory leak in the notification service", "a DNS record that was deleted"])
        docs[f"postmortem_{iid.lower()}.md"] = (
            f"# Incident {iid}\n\n{iid} was a {sev} incident on {rng.randint(1, 28)} "
            f"{rng.choice(['January', 'May', 'July', 'October', 'November'])} {year} that lasted "
            f"{rng.randint(12, 240)} minutes. The root cause was {cause}. The {rng.choice(TEAMS)} team led the response.\n\n"
            f"# Follow-up Actions for {iid}\n\nAdd alerting for the failing component and review the runbook "
            f"within {rng.choice([5, 10, 14])} business days.")
    return docs


def vendor_docs() -> dict[str, str]:
    names = ["DataVault", "OfficeHub", "CleanCo", "TranslateNow", "CloudBackup", "LegalEase", "TravelDesk",
             "SnackBox", "SignalWire", "PayrollPlus", "DeskPro", "Learnly"]
    used = {42, 117, 230, 500}
    parts = []
    for name in names:
        n = rng.choice([x for x in range(1, 999) if x not in used])
        used.add(n)
        parts.append(f"VND-{n:04d} is {name}, which provides "
                     f"{rng.choice(['document storage', 'office supplies', 'cleaning services', 'translation', 'backup storage', 'legal advice', 'travel booking', 'payroll processing', 'helpdesk software', 'online courses'])}.")
    return {"vendor_register_2024.md": "# Vendor Register 2024\n\n" + "\n\n".join(parts[:6]),
            "vendor_register_2025.md": "# Vendor Register 2025\n\n" + "\n\n".join(parts[6:])}


def hardware_docs() -> dict[str, str]:
    return {"retired_hardware.md": (
        "# Retired Laptops\n\nSKU LAP-14-STD was the 14-inch standard laptop and cost 1,250 USD. "
        "SKU LAP-15-WS was the 15-inch workstation and cost 2,300 USD. Both were retired in 2023.\n\n"
        "# Retired Monitors\n\nSKU MON-27-QHD was the 27-inch QHD monitor and cost 350 USD. "
        "SKU MON-22-HD was the 22-inch HD monitor and cost 120 USD.\n\n"
        "# Retired Accessories\n\nSKU DCK-USB-A was the USB-A docking station and cost 150 USD. "
        "SKU HDS-WR-100 was the wired headset and cost 45 USD.")}


def meeting_docs() -> dict[str, str]:
    topics = ["Atlas: no update this week.", "Beacon design review moved to next sprint.",
              "Cedar vendor contract is with legal.", "Delta user research sessions are booked.",
              "Reminder to finish the annual security awareness course.",
              "The expense tool will show a new receipt upload screen.",
              "Desk booking in the London office now uses the new app.",
              "The VPN client will get a new version next month.",
              "Health plan enrollment questions go to the People team."]
    docs = {}
    for i in range(24):
        team = rng.choice(TEAMS)
        lines = [f"Attendees: {', '.join(person() for _ in range(rng.randint(3, 5)))}."]
        lines += rng.sample(topics, 3)
        lines.append(f"Action item: {person()} will share the {rng.choice(PROJECTS)} roadmap by Friday.")
        docs[f"meeting_notes_{i + 1:02d}.md"] = (
            f"# {team} Weekly Sync {i + 1}\n\n" + " ".join(lines) +
            f"\n\n# Decisions in Sync {i + 1}\n\nThe team agreed to {rng.choice(['move standups to 09:30', 'pause the feature freeze', 'adopt the new on-call rota', 'archive old dashboards', 'run a retrospective next week'])}.")
    return docs


def noise_docs() -> dict[str, str]:
    docs = {}
    roles = ["Backend Engineer", "Product Designer", "Data Analyst", "Site Reliability Engineer",
             "Technical Writer", "Account Executive", "Recruiter", "QA Engineer", "Frontend Engineer", "Product Manager"]
    for i, role in enumerate(roles, 1):
        docs[f"job_post_{i:02d}.md"] = (
            f"# {role}\n\nWe are hiring a {role} to join the {rng.choice(TEAMS)} team in our "
            f"{rng.choice(OFFICES)} office. You will work with {rng.randint(3, 12)} colleagues.\n\n"
            f"# What We Offer\n\nFlexible hours, a learning culture and a modern office close to public transport.")
    industries = ["retail", "logistics", "healthcare", "banking", "education", "travel", "insurance", "media",
                  "manufacturing", "energy"]
    for i, ind in enumerate(industries, 1):
        docs[f"case_study_{i:02d}.md"] = (
            f"# Case Study: A {ind.title()} Customer\n\nA {ind} company moved its reporting to our platform "
            f"and cut month-end close time by {rng.randint(20, 60)}%. The rollout took {rng.randint(4, 16)} weeks.\n\n"
            f"# Results for the {ind.title()} Customer\n\nThe customer now serves {rng.randint(2, 40)} thousand "
            f"users and plans to expand to {rng.randint(2, 9)} more countries.")
    for i in range(1, 11):
        docs[f"blog_post_{i:02d}.md"] = (
            f"# Engineering Blog {i}: {rng.choice(['Scaling queues', 'Faster builds', 'Testing at scale', 'Better on-call', 'Design tokens', 'Observability basics', 'Caching strategies', 'Schema migrations', 'Feature flags', 'Incident reviews'])}\n\n"
            f"This post describes how the {rng.choice(TEAMS)} team approached the problem, what failed first, "
            f"and the {rng.randint(3, 7)} lessons the team learned along the way.")
    return docs


def main() -> None:
    docs: dict[str, str] = {}
    for make in (team_docs, project_docs, error_docs, api_docs, incident_docs, vendor_docs,
                 hardware_docs, meeting_docs, noise_docs):
        docs.update(make())
    bad = [(name, pat) for name, text in docs.items() for pat in FORBIDDEN if re.search(pat, text)]
    if bad:
        raise SystemExit("distractors violate the rules:\n  " + "\n  ".join(f"{n}: {p}" for n, p in bad))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name, text in sorted(docs.items()):
        (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")
    print(f"wrote {len(docs)} distractor documents to {OUT}")


if __name__ == "__main__":
    main()
