# AI Job Search

![Pip, the courier bird](assets/mascot/pip_flight_loop.gif)

A local job-discovery and review tool. It collects postings from job boards,
decides up front which boards it is allowed to read, and hands you a queue to
work through by hand.

Everything runs on your machine. There is no scheduler, no application
submission, and no automatic sending.

---

## What this actually does

Four things, in order:

1. **Decides** whether each source may be read, and records why.
2. **Collects** postings from permitted sources into a local store.
3. **Shows** you a queue of jobs worth reading.
4. **Records** your decision on each one.

It does **not** apply to jobs. There is no `applied`, `submitted` or `rejected`
status anywhere in this project, and that is deliberate.

---

## Quick start

```powershell
# 1. What am I allowed to read?
py tools/check_access.py report

# 2. What would a run do? (no network requests)
py tools/discover.py --dry-run

# 3. Collect (spends real requests against per-source daily budgets)
py tools/discover.py --report data/ops/daily.html

# 4. Review the queue (read-only, offline)
py tools/review.py --queue

# 5. Record your decision on a job
py tools/set_status.py --job-id "<job-id>" --status interested
```

On Windows set `PYTHONPATH` first, in every shell:

```powershell
$env:PYTHONPATH="."
```

Job IDs are the posting URLs. `py tools/review.py --queue` prints them.

---

## Step by step

### 1. Access is decided before anything is fetched

Each source gets an explicit recorded decision, based on its `robots.txt`, its
terms, and where you want to work. Nothing is fetched from a source without a
decision, and an unreadable terms page resolves to `unknown`, not "allowed".

```powershell
py tools/check_access.py report          # what is decided and why
py tools/check_access.py verify --source weworkremotely
```

Current state:

| Source | Decision | Why |
|---|---|---|
| `weworkremotely` | **permitted** | public feed, robots allow |
| `myjobmag.co.ke` | **permitted** | RSS feed path only, robots allow |
| `remotive.com` | **permitted** | public API only; robots disallow the site, so only the documented API is used |
| `hiring.cafe` | **restricted** | serves an anti-bot challenge; collecting would mean bypassing an access control |

That last one is the important one: the server says no in a way that would
require circumvention, so this project does not collect from it.

### 2. Check the run before making it

```powershell
py tools/discover.py --dry-run
```

Prints what would run, what would be skipped, and what would be refused — with
zero network requests. Run this first, every time.

### 3. Collect

```powershell
py tools/discover.py --report data/ops/daily.html
```

One manual pass. There is no scheduler; you start it. Requests are rate-limited
and capped per source per day, and the budget is persisted, so restarting does
not reset it. Total live requests made so far: **6**.

### 4. Review the queue

```powershell
py tools/review.py --queue
```

Read-only: no network, no writes, no changes of any kind. Output is
deterministic, so you can diff two runs and see only real changes.

```
# | JOB ID | TITLE | COMPANY | SOURCE | ELIGIBLE | FRESHNESS | STATUS | MATCH
```

Current store: **207 jobs, 101 in the actionable queue**, all Kenya-eligible,
all `new`, none assessed.

A job stays in the queue unless you dismiss it. Missing analysis is never a
reason to hide a job.

### 5. Record your decision

```powershell
py tools/set_status.py --list-statuses
py tools/set_status.py --job-id "<job-id>" --status reviewing
py tools/job_status.py show --job-id "<job-id>"   # full decision trail
```

Statuses are `new`, `reviewing`, `interested`, `shortlisted`, `dismissed`.
Every change is appended, never overwritten.

The queue orders least-settled first: `new` → `reviewing` → `interested` →
`shortlisted`, with `dismissed` excluded.

---

## Honest status of match assessment

`tools/match.py` exists and is deliberately hard to misuse: it assesses only
jobs you name, verifies every quote against the real record, discards anything
it cannot find, and never invents a tier or score.

**It has never produced a usable result.** Three controlled trials against a
local `llama3.2` model (3B, CPU-only) produced nine requests and zero
assessments:

| Attempt | Result |
|---|---|
| 120s timeout | timed out on all 3 jobs |
| 600s timeout, preloaded | returned unparseable output |
| 600s + JSON mode + bounded repair | valid JSON containing no evidence |

The machine (6 cores, 7.3 GB RAM, no GPU) cannot host a model capable of this
task, and no other installed model is larger except `llama2`, whose 4,096-token
context is far too small for the prompt. **Further local-model trials are not
approved and are not recommended on this hardware.**

Until a provider result exists, every job reads **not assessed** — and that is
shown as unassessed rather than dressed up.

### Review hints (not assessments)

`app/jobs/hints.py` computes transparent signals — shared terms between a
posting and your profile, title overlap, seniority wording, required vs
preferred phrasing.

These are **arithmetic, not judgement**. They carry no tier, no score, and no
claim about fit. Shared words are not shared skill, and an absent signal means
the posting did not say, not that the answer is no. They cannot change
eligibility, status, or queue membership.

---

## Layout

| Path | What lives there |
|---|---|
| `app/jobs/` | discovery, store, eligibility, freshness, status, match, assessment, hints |
| `app/sources/` | source adapters, transport, access control, per-source budgets |
| `app/reporting/` | the review report and queue rendering |
| `app/llm/` | provider-neutral LLM contract and the Ollama client |
| `app/profile/` | candidate profile loading |
| `app/policies/` | profile, evaluation and document rules |
| `tools/` | the command-line entry points |
| `data/` | your store. Git-ignored. |
| `documents/` | your CV and source material |

### The store

Everything lives under `data/`, git-ignored:

| File | What it holds |
|---|---|
| `jobs.jsonl` | every posting collected |
| `seen.json` | which postings have been seen |
| `freshness.jsonl` | when each source last listed each job |
| `access.json` | recorded access decisions |
| `access_attempts.jsonl` | every request ever made |
| `matches.jsonl` | match assessments, append-only |

---

## Prerequisites

- Python 3.10+
- No LLM required — the tool works fully without one

Only needed for match assessment, which is not currently recommended:

- Ollama at `http://localhost:11434`

---

## Development

```powershell
py -m unittest discover -s tests -t .    # 1290 tests
py tools/security_guards.py
```

Both must pass before a change is merged. Tests open no sockets and load no
models.

---

## Rules this project holds to

- **Job postings are untrusted data.** They are never executed as instructions
  or shell commands.
- **Nothing is hidden.** A job is removed from view only if you dismiss it. A
  low match score never hides anything.
- **Missing analysis is not a bad result.** Unassessed is shown as unassessed.
- **Access fails closed.** No readable terms page means `unknown`, and `unknown`
  means no collection.
- **Append-only history.** Assessments, statuses and requests are recorded, not
  overwritten.
- **No applications.** Nothing is sent anywhere on your behalf.

See [SECURITY.md](SECURITY.md) and `app/policies/` for detail.