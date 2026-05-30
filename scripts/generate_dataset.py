from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "emails.csv"

RISK_EVENTS = [
    "checkout outage",
    "payment failure",
    "production database backup failure",
    "security alert",
    "failed login spike",
    "SLA breach warning",
    "legal notice",
    "compliance audit request",
    "payroll approval deadline",
    "executive client escalation",
    "cloud access removal",
    "data pipeline failure",
    "vendor service suspension",
    "contract renewal deadline",
    "incident bridge request",
    "refund exception approval",
    "tax filing document request",
    "release rollback decision",
    "customer account lockout",
    "confidential board update",
]

NORMAL_TOPICS = [
    "weekly newsletter",
    "optional wellness webinar",
    "team lunch menu",
    "office plant schedule",
    "parking notice",
    "book club reminder",
    "training catalog update",
    "new coffee machine",
    "photo contest reminder",
    "monthly all-hands recording",
    "conference room rename",
    "desk clean-up day",
    "recipe exchange",
    "gym membership discount",
    "software tip of the week",
    "new hire introductions",
    "community volunteering signup",
    "holiday decoration guidelines",
    "lost and found note",
    "internal blog post",
]

TEAMS = [
    "engineering",
    "finance",
    "support",
    "sales",
    "security",
    "legal",
    "operations",
    "hr",
    "data",
    "facilities",
]

CLIENTS = [
    "Acme Retail",
    "Northwind Health",
    "BluePeak Bank",
    "Vertex Logistics",
    "Summit Energy",
    "UrbanCart",
    "Nimbus Travel",
    "BrightPath Schools",
]

DEADLINES = [
    "before 10 AM",
    "within 30 minutes",
    "by noon",
    "before end of day",
    "within 24 hours",
    "before the client call",
    "before payroll closes",
    "before the release window",
]

NORMAL_TIMES = [
    "next week",
    "next month",
    "on Friday",
    "this quarter",
    "when convenient",
    "during office hours",
    "after the all-hands",
    "by the end of the month",
]

IMPORTANT_SUBJECTS = [
    "Urgent: {event} for {client}",
    "Action required: {event}",
    "{team} escalation: {event}",
    "Immediate review needed for {event}",
    "Deadline {deadline}: {event}",
    "Production risk: {event}",
    "Customer impact alert: {event}",
    "Approval needed: {event}",
]

IMPORTANT_BODIES = [
    "{client} is blocked because of a {event}. Please respond {deadline} with owner, status, and next action.",
    "The {team} team found a {event}. This may affect customers, revenue, access, or compliance unless handled {deadline}.",
    "We need a decision on the {event} {deadline}. Attach the latest evidence and confirm the accountable owner.",
    "A high-priority request came in from {client} regarding {event}. Leadership expects an update {deadline}.",
    "The current {event} can delay operations and trigger escalation. Please join the response thread {deadline}.",
    "A control check flagged {event}. Send acknowledgement, risk notes, and resolution plan {deadline}.",
]

NORMAL_SUBJECTS = [
    "{topic} from {team}",
    "FYI: {topic}",
    "{team} update: {topic}",
    "Reminder: {topic}",
    "{topic} available",
    "Optional: {topic}",
    "New note about {topic}",
    "Shared update: {topic}",
]

NORMAL_BODIES = [
    "Sharing a general update about {topic}. Review it {timeframe} if it is useful for your work.",
    "The {team} team published details for {topic}. No action is required unless you want to participate.",
    "This is a routine note about {topic}. You can read it {timeframe}.",
    "Please see the attached information about {topic}. It is informational and can be reviewed {timeframe}.",
    "A friendly reminder about {topic}. Participation is optional and there is no operational impact.",
    "The latest information for {topic} is now available. Save it for reference whenever convenient.",
]


def make_important(rng: random.Random, index: int) -> tuple[str, str, str]:
    event = rng.choice(RISK_EVENTS)
    team = rng.choice(TEAMS)
    client = rng.choice(CLIENTS)
    deadline = rng.choice(DEADLINES)
    subject = rng.choice(IMPORTANT_SUBJECTS).format(
        event=event,
        team=team.title(),
        client=client,
        deadline=deadline,
    )
    body = rng.choice(IMPORTANT_BODIES).format(
        event=event,
        team=team,
        client=client,
        deadline=deadline,
    )
    body += f" Reference ID EMAIL-{index:05d}."
    return subject, body, "important"


def make_normal(rng: random.Random, index: int) -> tuple[str, str, str]:
    topic = rng.choice(NORMAL_TOPICS)
    team = rng.choice(TEAMS)
    timeframe = rng.choice(NORMAL_TIMES)
    subject = rng.choice(NORMAL_SUBJECTS).format(topic=topic, team=team.title())
    body = rng.choice(NORMAL_BODIES).format(topic=topic, team=team, timeframe=timeframe)
    body += f" Reference ID EMAIL-{index:05d}."
    return subject, body, "normal"


def generate_rows(total_rows: int, seed: int) -> list[tuple[str, str, str]]:
    if total_rows < 2:
        raise ValueError("total_rows must be at least 2")

    rng = random.Random(seed)
    important_count = total_rows // 2
    normal_count = total_rows - important_count
    rows = [make_important(rng, idx) for idx in range(important_count)]
    rows.extend(make_normal(rng, idx) for idx in range(normal_count))
    rng.shuffle(rows)
    return rows


def write_dataset(rows: list[tuple[str, str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "body", "label"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic email priority dataset.")
    parser.add_argument("--rows", type=int, default=10000, help="Number of dataset rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible output.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    args = parser.parse_args()

    rows = generate_rows(args.rows, args.seed)
    write_dataset(rows, args.output)
    labels = {label: sum(1 for row in rows if row[2] == label) for label in ("important", "normal")}
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"important={labels['important']} normal={labels['normal']}")


if __name__ == "__main__":
    main()
