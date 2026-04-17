# Parallelism — K=1 vs K>1

The `parallelism` field on `spec.yaml` is the single knob that toggles between
**Karpathy's sequential loop** (K=1) and a **batched parallel hill-climb** (K>1).
Same `train.py`, same `eval.py`, same ledger — only the search strategy differs.

## K=1 — the original Karpathy loop

```
round 1:  propose → train → evaluate → decide  (keep if > best)
round 2:  propose → train → evaluate → decide
round 3:  propose → train → evaluate → decide
...
```

One proposal per round. Greedy hill climb. This is exactly what Karpathy's
reference implementation does, and it's what MVP-1 does by default.

Pros: cheap (one LLM call and one training run per trial), faithful to the
original, easy to reason about.

Cons: sequential. A 100-trial study at ~15 s per trial is ~25 minutes wall-clock.
And a greedy hill-climb can get stuck in local optima — if round 3's proposal
is worse than round 2's, you discard it and try again from the same best.

## K>1 — batched parallel hill-climb

```
round 1:  propose K diffs  →  [train, train, ..., train]   ← K trials in parallel
                              [eval,  eval,  ..., eval]
                              decide_round:  keep the best ONE (tournament)
round 2:  propose K diffs from the new best  →  ...
```

K = `spec.parallelism` trials per round. All K share the same parent (the
round's incoming best). After all K evaluate, a cohort tournament picks at most
one winner per round — the largest improvement over the incoming best. The
others are rejected; their diffs are still written to the ledger so they
aren't re-proposed next round.

**Cost.** K× training runs per round. The proposer call is still ONE call
though (it returns K diverse diffs in a single response, see
`OpenAIProposer.propose_batch`), so proposer cost is roughly flat.

**Wall-clock.** MVP-1 runs the K trials *sequentially* (1 CPU, no speedup), so
K>1 in local mode buys you **exploration diversity** (less greedy) but not
speed. MVP-2 runs the K trials *in parallel* across K Lambda containers, and
wall-clock drops by ~K×.

**Quality.** K diverse proposals per round means more thorough exploration of
nearby variants per step. Helps when the LLM's first-guess proposals are noisy
or when you're near a local optimum.

## Picking K

| spec.parallelism | Use when |
| --- | --- |
| `1` | You want the Karpathy original. Cheapest, slowest, still good for deep explorations. |
| `2–4` | "Safe default" for K>1 — low cost overhead, modest diversity win. |
| `5–10` | Typical MVP-2 choice. ~5–10× wall-clock speedup on Step Functions, room for diverse LLM proposals per round. |
| `10–20` | Aggressive exploration. Only useful if the problem surface has many local optima; costs K× training. |
| `> 40` | Rejected by `submit()` — Standard Map state caps at 40 concurrent branches. Distributed Map (10k) is MVP-3. |

## Budget semantics change slightly at K>1

With K=1, `daily_budget_usd` is checked after every proposal — the loop stops
at the first trial that would push total spend past the budget.

With K>1, one LLM call charges **K proposals' worth of tokens up front**, and
the Step Functions Map then runs all K trials in parallel (local mode: the K
trials run sequentially). Budget is checked *at round boundaries* in the
`decide` step. That means a round can overshoot the budget mid-flight — once
the K Lambdas are running you can't preempt them cleanly. In practice the
overshoot is bounded by the round's cost; pick `daily_budget_usd` with a buffer
if you're running K=20.

## Faithfulness to Karpathy

K=1 is a byte-identical sequential loop (the golden test
`tests/test_loop_golden.py` guards this). K>1 is a natural generalization —
same propose/train/evaluate/decide building blocks, same ledger shape, just
more trials per round. The LLM has no idea whether it's being called in K=1 or
K=10 mode; the only difference is whether the system prompt asks for one new
source or K diverse new sources.

If in doubt: start with K=1 and an MVP-1 run_local() on your laptop. Once the
recipe is good, bump `parallelism` to 10 and `submit()` to the cloud for
overnight sweeps.
