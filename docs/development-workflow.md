# Development workflow

## Default flow

For material changes use:

`decision/spec → GitHub issue → isolated branch → implementation/experiment → validation evidence → PR → review → human/collaborator gate where required → merge`.

Do not make material implementation or governance changes directly on `main`.

## Issue/spec expectations

A worker-ready issue should identify:

- problem/objective;
- scope and explicit non-goals;
- affected pipeline/product surface;
- acceptance criteria;
- validation/evidence expected;
- data/time-window assumptions where relevant;
- whether the change alters modeling/evaluation methodology;
- whether a collaborator reserved matter is implicated;
- any human gate before merge, Production, cost, permissions, or external publication.

## Branch and PR discipline

Prefer one narrow approved task per isolated branch/PR. Keep changes auditable and reversible. Avoid mixing model experiments, UI redesign, infrastructure changes, governance changes, and unrelated cleanup in one PR unless the approved issue explicitly requires them together.

Routine branch work does not require Rabin's intermediate approval. Eduardo + ChatGPT may fully iterate and validate on the isolated branch. Once the branch is solid and ready for shared `main`, a single consolidated PR review/approval or merge request to Rabin is the preferred collaboration point when practical. Rabin approval is mandatory only for reserved shared matters or another explicit repository rule.

PRs should explain:

- what changed and why;
- what did not change;
- exact validation run;
- relevant metrics/results before and after;
- leakage/chronology implications for model work;
- deployment/runtime implications;
- collaborator-reserved-matter status;
- remaining risks or follow-up.

## Validation by change type

### Data / feature engineering

Validate schema/output expectations, chronology, missingness behavior, and especially leakage safety. Show that features for a game only use information available before that game.

### Model training / evaluation

Record data window, train/test split, model/configuration, seed where relevant, and comparable metrics. Prefer chronological evaluation appropriate to the time-series/game sequence. Do not treat in-sample improvement as deployment evidence.

### Market / recommendation logic

Validate probability/odds conversion, vig removal/fair-probability logic, edge threshold semantics, bankroll/Kelly calculations, caps, and deterministic examples. Keep sportsbook/provider failures explicit.

### Streamlit / UI

Validate app startup and affected interactions. For subjective visual changes, provide an accessible rendered preview before asking for approval.

### Documentation / governance

Check that files do not contradict current code/project state or collaborator rights. Project Instructions must remain within the observed ChatGPT Project limit and must be synchronized exactly with the live UI only after merge.

## Production and external side effects

Repository validation does not authorize Production changes. Require Eduardo's explicit approval before changing deployment behavior, enabling automatic Production behavior, creating paid/recurring services, expanding credentials/permissions, or activating worker routes/schedules/webhooks.

No sportsbook account access or automated wager placement is part of the normal development workflow.

## Collaboration review

Routine reversible work can proceed under Eduardo/ChatGPT day-to-day coordination without involving Rabin during intermediate branch work. If the issue crosses a reserved shared matter in `docs/collaboration-governance.md`, obtain meaningful Rabin participation on the decision before treating that matter as approved.

Once a reserved matter is decided, Rabin does not automatically need to inspect every routine implementation detail unless the collaborators choose that review level.

## Worker use

No default execution worker is active at onboarding. Do not reuse another project's worker route. If a future worker is proposed, Project Systems must review logical identity, project-scoped physical route, access/credential boundaries, wake behavior, cost, and smoke evidence before activation.

Never claim a worker/CI/deployment is running without a trustworthy current signal.

## Continuation

Every material workstream should leave:

- current state;
- exact next owner;
- one immediate next action;
- whether Eduardo must act now.

Blocked work must state the awaited dependency and concrete unblock signal.
