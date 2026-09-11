# Baseball Moneyline agent instructions

This repository is the canonical collaborative repository for the Baseball Moneyline project and follows Eduardo's Project Systems operating model.

## Project identity

- Canonical repository: `rabin0208/baseball_moneyline`.
- Governance mode: collaborative.
- Proposed ChatGPT control tower: `00 — Baseball Moneyline HQ`.
- Repository owner/collaborator: Rabin (`rabin0208`).
- Day-to-day Project Systems coordination: Eduardo + ChatGPT, subject to the collaboration contract in `docs/collaboration-governance.md`.
- No default execution worker is active at onboarding. Project HQ lifecycle automation and any dedicated Grok route are deferred until separately reviewed and approved.

## Required reading before material work

Read only what the task needs, starting with:

1. `PROJECT_STATUS.md`;
2. the authoritative GitHub issue/PR for the task;
3. `docs/collaboration-governance.md` when ownership, publication, commercialization, reuse, provenance, or shared-work rights are relevant;
4. `docs/architecture.md` for pipeline/model/system boundaries;
5. `docs/development-workflow.md` for branch, validation, and review expectations;
6. the specific code/data/results needed for the task.

Do not rely on old chat history when current repository truth is sufficient.

## Authority order

1. explicit current joint collaborator decision or approved reserved-matter decision where required;
2. explicit current Eduardo decision for delegated day-to-day matters that do not cross a reserved matter;
3. approved GitHub issue/PR and this repository's committed operating docs;
4. current code, tests, data contracts, and validated results;
5. Project Systems shared standards;
6. chat history as working context only.

If authorities conflict, stop and surface the conflict rather than guessing.

## Collaboration boundaries

This is not an Eduardo-only repository. Rabin's substantive collaborator/ownership role remains meaningful.

Eduardo + ChatGPT may coordinate routine reversible development, experiments, documentation, issue/PR administration, and Project Systems mechanics without requiring Rabin to review every change.

Rabin participation is required for reserved shared matters defined in `docs/collaboration-governance.md`, including collaborator ownership/governance, fundamental product direction, material reuse/provenance/IP decisions, publication or commercialization of shared work, and destructive disposition of substantial shared work.

Do not copy Baseball code, data, trained models, documentation, or other substantive shared assets into the planned Hockey project unless collaborator reuse/provenance rights are explicitly resolved. Learning concepts and independently reimplementing them is allowed when rights remain unclear.

## Execution model

Default governed change:

`decision/spec → GitHub issue → isolated branch → narrow auditable change → validation/evidence → PR → review → human/collaborator gate where required`.

Never write material changes directly to `main`.

Do not redesign the model, data strategy, odds source, deployment architecture, or product direction merely as part of Project Systems onboarding or routine maintenance.

Refresh GitHub before reporting live issue, PR, CI, deployment, or worker status. Never claim background/worker activity without a trustworthy current signal.

## Technical baseline

The current project uses the MLB Stats API and Python/scikit-learn pipeline described in `README.md` and `docs/architecture.md`, including lagged pre-game features, model training/evaluation, market/ROI analysis, daily recommendations, and a Streamlit dashboard.

Preserve leakage-safe pre-game feature semantics. Changes that alter train/test chronology, target construction, odds interpretation, bankroll logic, recommendation thresholds, or evaluation methodology require explicit issue-level acceptance criteria and evidence.

## Protected gates

Require explicit Eduardo approval before:

- Production/deployment changes or automatic Production behavior;
- creating or activating any execution-worker route, webhook, schedule, or background wake;
- credential/permission expansion or new paid/recurring services;
- auto-merge;
- destructive/irreversible operations;
- significant architecture/product changes not already approved.

Also require the collaborator/reserved-matter decision defined in `docs/collaboration-governance.md` when the change crosses a shared reserved matter.

No sportsbook account access, automated wager placement, or betting-account automation is authorized by Project Systems onboarding.

## Continuation and review

Every material response should leave a clear continuation: current state, exact next owner, one immediate next action, and whether Eduardo must act now.

Use GitHub as the cross-chat handoff layer. When blocked, identify what is awaited, who owns it, the concrete unblock signal, and affected scope.

Before asking for approval, provide the evidence needed to decide. For visual/UI changes, provide an accessible rendered preview/review surface and say what to inspect.
