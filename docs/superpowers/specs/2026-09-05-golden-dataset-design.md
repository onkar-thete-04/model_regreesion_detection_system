# Phase 2 — Golden Dataset Design

Date: 2026-09-05
Feature: customer support email classifier (category + one-sentence summary)

## Goal

Create a human-verified ground-truth dataset that the CI regression pipeline will
use to grade the email classifier. The dataset is the fixed "eval bar" — the
classifier is judged against it; if the bar changes, the version bumps.

## Scope

- 75 hand-written test cases (no LLM generation — ground truth is human-verified).
- Coverage across all four categories: billing, technical, account, general.
- Deliberate edge cases: ambiguous, extremely short, typos, mixed-language, sarcastic.
- Versioned single JSON file with stable IDs.

## Data Schema

Top-level:

| field     | type   | description |
|-----------|--------|-------------|
| version   | int    | dataset version, starts at 1 |
| feature   | string | "email_classifier" |
| version_date | string (ISO-8601) | when this version was authored |
| cases     | array  | list of GoldenCase objects |

Each case:

| field               | type           | description |
|---------------------|----------------|-------------|
| id                  | string         | stable content-hash ID (see below) |
| input               | string         | the customer email text |
| expected.category   | string         | single correct category (billing/technical/account/general) |
| expected.category_any | array (optional) | acceptable categories for ambiguous cases; grader accepts any of these when present |
| expected.summary    | string (optional) | ideal one-sentence summary; omitted for cases where a unique ideal summary is ill-defined |
| expected_difficulty | int (1-5)      | 1 = trivial, 5 = adversarial |
| notes               | string         | why this case matters |

## ID Scheme

Deterministic content-based hash, immutable under reordering/insertion:

- `id = "c-" + sha256(category + "\x00" + input).hexdigest()[:10]`

No renumbering on insert or delete; an ID survives even if the JSON is reordered.
Cases are never removed silently — deprecated cases move out of `cases` only on a
version bump (out of scope for phase 2).

## Expected Output Semantics (agreed)

- Ordinary case: `category` set, `category_any` absent, `summary` set.
- Ambiguous case: `category` still set to the best single answer (for reporting),
  and `category_any` lists all acceptable answers the grader will accept.
- Hard cases where no unique ideal summary exists: `summary` omitted (grader skips
  summary check for those, per difficulty).

## Difficulty Scale (numeric 1-5)

1 — obvious, single clear intent
2 — straightforward with minor noise
3 — moderate: requires light inference or mild ambiguity
4 — hard: ambiguity, sarcasm, mixed language, heavy typos
5 — adversarial: traps for the model (e.g. sarcasm that inverts meaning)

## Edge Case Distribution (target)

- Ambiguous (could be two categories): ~6
- Extremely short (< 3 words or single word): ~6
- Typos / misspellings: ~6
- Mixed language: ~4
- Sarcastic / tone-inversion: ~5
- Clean, ordinary per category: remainder

## Files

- `datasets/golden.json` — rewritten to the schema above (v1, 75 cases).
- `llm_regress/types.py` — add `GoldenCase`, `GoldenDataset`, and Pydantic models
  for `ExpectedOutput` (category, category_any, summary).
- `llm_regress/dataset.py` — update `Dataset.load` (and `Case`) to parse the new
  schema; expose `GoldenDataset.load` returning Pydantic models.

## Constraints

- No LLM generation of cases. Hand-authored.
- Cases read real-looking; no fabricated hyperlinks or PII (use placeholder emails).
- Energy/SaaS-neutral tone for clean cases so difficulty rating stays meaningful.

## Testing

- A loader test asserting: 75 cases, all ids unique + stable-format, all categories
  valid, difficulty in 1-5, every case has notes, category_any supersets category.
