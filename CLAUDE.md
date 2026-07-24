# neurom — NL PGD branch (`nl_pgd`)

**Goal:** extend `docs/examples/1d_2-parametric_beam_PGD/` from 2 to 5 parameters
(x, E1, E2, α, n), then replace CP-PGD with non-linear PGD decompositions.

**Vocabulary:** `u = Σⱼ ∏ₖ wⱼᵏ(xₖ)`. The `wⱼᵏ` are **monoms**, the `uⱼ` are **modes**.

**Method:** PGD — *a priori* MOR: the basis is built on the fly, never by
compressing precomputed solutions. Modes come from global minimisation (FENNI
Part I), not alternating-direction power iterations. Implemented in
`src/neurom/decompositions/pgd.py`.

## Rules

1. **Reuse before writing.** Search the library for an existing primitive before
   adding one. Match the surrounding code's idiom.
2. **Inject dependencies** (energies, criteria, factories) rather than
   subclassing or hard-coding.
3. **New decompositions go in `src/neurom/decompositions/pgd.py`**, beside
   `CPPGD` — not in new modules.
4. **The user runs work files**, unless asked to run a campaign.
5. **Never add Claude as co-author**, and no "Generated with Claude Code" or 🤖
   in commits or PR descriptions.
6. **Stage explicit paths.** Never `git add -A` or `git add .`.

## Layout

| Path | Contents |
|---|---|
| `src/neurom/` | library |
| `tests/` | tests for `src/` only |
| `docs/examples/<case>/` | example script + `.md` |
| `docs/examples/<case>/tests/` | tests that load an example by path |
| `docs/notes/<date>-<topic>.md` | long-form derivations, ablations |

Example test dirs must be listed in `pyproject.toml`'s `testpaths`.

## CHANGELOG.md

Read it at session start. Append newest-first before session end.

Record: architecture changes in `src/`; experimental status — what converges,
what fails and how (divergence, NaN, poor accuracy), the test that shows it,
known-good settings. **Negative results matter** — write them down so they are
not retried blindly. Skip: formatting, comment tweaks, inconclusive debugging.

**Format — hard limits:**

- Open with a **State** line: which files exist/changed, whether they run
  standalone, what is still missing. Before any results.
- ≤ 12 bullets, ≤ 40 lines. Overflow goes to `docs/notes/` and gets linked.
- Measured numbers in a table, not prose.

**Measurements vs. thresholds.** `assert error < 0.15` is a bound, not a result.
Before claiming better/worse/no-regression, measure both sides and quote both,
or say you did not measure. Flag single runs vs. ablations, and plausible
mechanisms vs. established ones.

## Pointers

- NL PGD spec: `~/Library/CloudStorage/GoogleDrive-solal21a@gmail.com/Mon Drive/Solal_PhD/PhD_Solal_vault/5_Projects/NL PGD/premiers_tests/NL_PGD_rundown_4_claude.md`
- References (Ammar 2006, Nouy 2010, FENNI I & II): `docs/notes/references.md`
