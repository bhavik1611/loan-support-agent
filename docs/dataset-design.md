# Cred capstone: dataset design

Date: 2026-09-06.
Source of requirements: `resources/capstone/problem_statement_banking.md` only.

This document decides the loan-application dataset for the Cred support agent.
It covers the records, not the knowledge base, which is still open.

Twenty decisions were settled across four grilling rounds.
Every quantitative claim below carries the probe that produced it.
Probes are throwaway and were not committed; the figures they produced are what matters, and each is reproducible from the generator described here.

## 1. The framing decision

The brief fixes six fields on each record: `record_id`, `category`, `status`, `loan_amount_inr`, `days_since_created`, `flagged_for_fraud_review`.

Those six fields are a **projection**, not the source of truth.
Behind them sits a record that models the application's life; `LOAN_APPLICATIONS` is a view over it.

This exists to fix a defect that flat generation cannot avoid.
When `status` and `days_since_created` are drawn independently, nothing stops a loan being `Disbursed` and 25 days old and sitting on an urgent-attention queue, or reaching `Disbursed` within two days of submission.
Both occurred in the earlier design.
Deriving status from the clock makes the contradiction impossible rather than merely unlikely.

Two alternatives were rejected.
A real sourced dataset such as Lending Club breaks one-command reproducibility, adds licensing and personal-data questions, and the brief asks for your own seeded generator.
Realistic values on a flat six-field generator leaves the contradiction in place.

## 2. What one application looks like

Each application carries, internally:

- an opaque member reference, never a name
- the product
- the day it was created, as an offset in the range 0 to 30
- a full event history: every stage change, each with its date and who moved it
- the amount
- the fraud flag

`status` and `days_since_created` are both read off the event history.
They cannot disagree, because they come from one clock.

The history entries name an actor: the system, a credit officer, an underwriter.
That is what makes it an audit trail rather than a log.

## 3. The generator

**Products and their speeds.** Each product has its own time from submission to decision.

| Product | Days to decision | Amount floor | Amount ceiling |
|---|---|---|---|
| Personal Loan | 2 to 10 | 50,000 | 25 L |
| Auto Loan | 3 to 14 | 1 L | 30 L |
| Education Loan | 7 to 25 | 1 L | 50 L |
| Business Loan | 10 to 35 | 2 L | 75 L |
| Home Loan | 20 to 50 | 10 L | 1.5 cr |

These bands are an assumption about a plausible Indian lender, not anything the brief supplies.
They were put to review and drew no objection, which is not the same as approval; every amount in the project inherits them, so they remain the most useful thing to challenge.

**Amounts** are drawn skewed toward the band floor rather than uniformly.
A uniform draw puts 35% of home loans above 1 crore with a median of 79 lakh, which no Indian home-loan book resembles.
Skewed, the median lands at 41 lakh with 18.5% above a crore.

**Status** is whatever stage the application had reached by the observation day.
One application in ten enters a stalled review that drags 22 to 34 days, which is what a real lending desk lives with and what makes waiting time a signal at all.

**The fraud flag** is not a coin flip.
It rises for an amount near the top of its product band and for a member applying repeatedly, at roughly double the base odds for either.

**Members** are drawn from a pool large enough that about one application in ten belongs to a member who has another.
Roughly seven members hold two applications that are both still open.

**Corpus size** is around 500 records, not the brief's floor of 40.

## 4. Escalation

Only `Submitted` and `Under Review` applications are scored.
A finished loan returns its status and no urgency number, because there is nothing for a human to take over.

The score combines the two signals the brief names:

```
score = 0.45 * fraud_flag + 0.55 * (days_since_created / oldest_open_application)
```

Recency is normalised against the other open applications rather than against a fixed 30 days.
The threshold is the score at the 90th percentile of open-application age, so the queue is the slowest tenth.

**A consequence worth stating plainly.**
With these weights, a fraud-flagged application clears the threshold at any age.
The flag is therefore a sufficient condition for escalation, and the score's remaining work is ordering the queue and admitting genuinely stalled files that carry no flag.
That is defensible, since a flagged application warrants human eyes regardless of age, but it was decided by default rather than chosen.
Lowering the fraud weight below the threshold is the lever if that reading is wrong.

## 5. Reproducibility

`dataset.py` regenerates the corpus from a stated seed and prints the report the brief asks for.

A snapshot and its SHA-256 are committed, and a verify target re-derives the hash and diffs.
If a grader's machine produces anything different, the check fails loudly rather than quietly grading different data.

This matters because determinism is only verified same-version.
Two fresh Python 3.14.0 interpreters produce identical draws.
Cross-version stability of `random.randint` and `random.choice` is **not** verified, and CPython has changed those internals before.
The hash is what turns that unverified assumption into something that announces itself.

The seed is not asserted, it is searched.
The scan script and its results table ship with the repo, so the choice is visible rather than taken on trust.

## 6. Measurements behind the design

| Claim | Figure | How it was produced |
|---|---|---|
| Independent draws escalate finished loans | 6 of 10 escalating records at seed 1 were already Disbursed or Rejected; 0 after gating | 48-record generator, seed scan |
| Deriving status from age kills the recency signal | 22 of 22 escalating records across four seeds were fraud-flagged | gated escalation probe |
| That collapse is structural, not a bad draw | an open application tops out at 24 days; recency alone needs day 27.3 | arithmetic on the dwell table |
| A small corpus cannot hold rare things | at 48 records, 74 of 100 seeds contain zero disbursed home loans; at 500, 7 do | 100 seeds per size |
| Corpus size stabilises the fraud band | fraud rate spans 6.2% to 33.3% at 48 records, 12.0% to 23.8% at 500 | 100 seeds per size |
| One flat amount range is unusable | 284 of 500 records (57%) hold an amount impossible for their product | 500 records, seed 3 |
| A uniform draw inside a band is wrong too | 35% of home loans above 1 crore, median 79 lakh | 20,000 draws |
| An uncorrelated fraud flag says nothing | 17.1% to 18.9% across every product and both ends of every band | 500 records, 40 seeds |
| Gentle correlation keeps the flag informative | 31% of flags predictable from the two signals, against 19% by chance | 500 records, 40 seeds |
| Strong correlation does not | 57% predictable | 500 records, 40 seeds |
| Realistic product speeds skew slow products open | only 36% of home loans can appear Disbursed inside a 30-day window | 20,000 draws per product |

## 7. Known tensions, accepted deliberately

**The 30-day ceiling versus slow products.**
A home loan takes 20 to 50 days end to end, and the brief caps `days_since_created` at 30.
Only 36% of home loans can therefore ever appear as `Disbursed`.
Accepted as honest realism: a month's snapshot of a real loan book does skew that way, and at 500 records disbursed home loans still appear reliably.

**Cred is modelled as a bank, not as a distributor.**
The real Cred does not hold the loans behind it; partner lenders do.
Carrying a partner name would be truer, and it was rejected.
The brief's own required policy topics include minimum-balance rules, account closure, joint accounts and NRI accounts, which are deposit-account topics only a bank has.
Modelling Cred as the bank keeps the records and the policy documents describing one institution.

A side effect of that choice is that a risk which was never measured no longer applies.
Partner-specific policy documents might have flattened Precision@3 through near-duplicate text.
That was raised during the rounds as unverified, on the mistaken grounds that `sentence-transformers` was unavailable.
It is in fact installed in the repo's `venv/`, at version 5.7.0 alongside `chromadb` 1.5.9; the earlier check ran against the system interpreter instead.
The risk could therefore have been measured and was not.
With no partners there are no partner-variant documents, so the question is now moot rather than answered, and Q14 was decided on the brief's own topic list rather than on this.

## 8. Not decided here

The knowledge base is untouched: how many documents, how long each is, how they are chunked, and how retrieval ground truth is labelled.
Those questions decide Precision@3 and Recall@3, and they are the next conversation.

## 9. Relationship to `capstone-v1-spec.md`

That document is not edited by this one.
Its sections 6.1 and 7.1 describe the independent-draw generator and the escalation design that the measurements above show to be broken.
Reconciling the two files is a separate call, and deleting or rewriting the older one has not been done.

## 10. The twenty decisions

| # | Question | Decision |
|---|---|---|
| Q1 | How much of an application's life do we keep? | Full event history |
| Q2 | Where do the records live at runtime? | A database file the agent queries |
| Q3 | How does waiting time become a signal? | Stalled applications exist, and waiting is judged against the other open ones |
| Q4 | Can a finished loan escalate? | No, never scored |
| Q5 | What does reproducible mean? | One command plus a committed fingerprint |
| Q6 | One institution across policy and records? | Yes, one shared product table |
| Q7 | How is the seed defended? | Publish the search |
| Q8 | Do products move at different speeds? | Yes, per-product timelines |
| Q9 | How many records? | Around 500 |
| Q10 | What does one history entry record? | Stage, date, and who moved it |
| Q11 | Does the agent use that history? | Yes, the lookup answers with the story |
| Q12 | Can one member hold several applications? | Yes, opaque reference, no names |
| Q13 | How busy is the escalation queue? | The slowest tenth of open applications |
| Q14 | Does a record carry a partner bank? | No, Cred is the bank |
| Q15 | Where inside its band does an amount land? | Skewed toward the floor |
| Q16 | Does the fraud flag correlate with anything? | Yes, large-for-its-product and repeat member |
| Q17 | How stuck does a stuck application get? | One in ten, dragging 22 to 34 days |
| Q18 | How common is a repeat member? | About one in ten of the book |
| Q19 | How strong is the fraud correlation? | Gentle, roughly double the odds per signal |
| Q20 | What uses the member link? | The existing lookup mentions it |
