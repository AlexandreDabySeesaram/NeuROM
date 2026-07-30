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
3. **New decompositions go in `src/neurom/decompositions/`** — `pgd.py` beside
   `CPPGD`, or a sibling module when the format warrants it
   (`polynomial_pgd.py` holds `PolynomialNLPGD`). Export from
   `decompositions/__init__.py`.
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

## Running experiments

Always `.venv/bin/python` — bare `python` has no torch.

`sweep.py` in the example directory is the entry point, not ad-hoc scripts. Its
JSONL ledger is the source of truth for results; a number that only exists in a
chat transcript did not happen. Rows are keyed by the config's content hash, so
editing the queue never loses past results. Never hand-edit the ledger — use
`delete_row`.

Figures live in `plots/` and redraw from a checkpoint (`show_run`,
`plot_losses`, `plot_extremes`). **Never retrain to change a label.**

**Units, stated once because two of them differ:**

| knob | unit |
|---|---|
| `min_iter`, `max_iter` | per **stage** — a 2-stage schedule spends 2× per mode |
| `MaxStages`, `n_modes_max` | per **mode** (rank) |
| cost | quadratic in **terms**, `1 + \|I\|` per mode — not in modes |

**Every number carries its provenance.** Mesh, rank, and iteration budget beside
any result, and tiny-mesh or single-run results labelled as such. Most numbers
are misread without it.

## Known negative results — do not retry blindly

- **`staged` (CP mode, then `C` alone) has no working `coefficient_lr`.** Above
  `1e-5` the `p=3` row diverges; below, the `p=2` row never leaves zero. The
  cause is structural, not a tuning failure: `C_p ~ A^(1−p)`, so the rows of one
  `C` sit orders apart once `A` has converged, and Adam steps by `~lr` whatever
  the gradient. `joint` escapes it by releasing `C` at the mode's seed. Full
  derivation and sweep: `docs/notes/2026-07-29-nl-pgd-coefficient-scale.md`.
- **`SimultaneousTrainer` below ~300 iterations per stage never launches the new
  mode** (amplitude `3e1` vs mode 0's `3e5`), and `RelativeGain` then calls the
  run converged after two stages. `STAGE_MIN_ITER` records this, but the sweep
  does not consult it — a `RunConfig` is the single source of every knob, so
  `cfg.min_iter` wins and must be set explicitly.

**Open decision.** Reparameterising the coefficients as
`C_λ = C̃_λ · A^(1 − Σⱼλⱼ/d)` inside `PolynomialNLPGD` would put every row on an
`O(1)` scale and make `staged` viable. Deliberately not done: it changes
`state_dict` semantics, so it needs a format version and a decision on whether
`A` is live or frozen per stage. Not a bug — do not "fix" it silently.

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

- References (Ammar 2006, Nouy 2010, FENNI I & II): `docs/notes/references.md`
- the plan for the non-linear PGD implementation `/Users/solal/Library/CloudStorage/GoogleDrive-solal21a@gmail.com/Mon Drive/Solal_PhD/PhD_Solal_vault/5_Projects/NL PGD/nl_pgd/non-linear_pgd_list.md`
