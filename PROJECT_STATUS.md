# Baseball Moneyline project status

## Current checkpoint

Canonical repository: `rabin0208/baseball_moneyline`.

Governance mode: collaborative.

Proposed ChatGPT control tower: `00 — Baseball Moneyline HQ`.

Project Systems onboarding is in progress under `EduardoRasanmar/project-systems#35`.

The existing technical/product baseline is preserved. Onboarding does not redesign the model, data pipeline, dashboard, market logic, deployment, or collaborator ownership.

## Collaboration model

Rabin (`rabin0208`) remains a substantive collaborator and repository owner. Eduardo + ChatGPT may coordinate routine reversible development, experiments, documentation, issue/PR administration, and Project Systems mechanics without requiring Rabin to inspect every routine change.

Reserved shared matters require collaborator participation as defined in `docs/collaboration-governance.md`, including ownership/governance, fundamental product direction, material provenance/reuse/IP, publication/commercialization of shared work, and destructive disposition of substantial shared work.

Baseball-to-Hockey direct copying of shared code, data, trained models, docs, or other substantive assets is not authorized unless reuse/provenance rights are explicitly resolved. Conceptual learning and independent reimplementation remain possible.

## Technical baseline

The repository currently provides:

- MLB Stats API schedule/history ingestion;
- EDA and cleaned historical game data;
- lagged pre-game team, head-to-head, rest-day, calendar, and pitcher-outcome features;
- logistic regression, random forest, and gradient boosting model training;
- chronological season holdout and walk-forward/fixed-train evaluation paths;
- current-season and next-day predictions;
- sportsbook moneyline ingestion and market/ROI evaluation;
- daily bet-recommendation logic with capped fractional-Kelly staking;
- a Streamlit probability/recommendation dashboard.

Current `main` at onboarding start: `9bb24797535372301dff912398e707502b325b9a` (`Merge pull request #15 ... Add Kelly criterion`).

See `README.md` and `docs/architecture.md` for the current implementation map.

## Runtime / workers

- Project HQ lifecycle automation: deferred.
- Default execution worker: none.
- Dedicated Grok/other provider route: not active and not authorized by onboarding.
- Background schedules/webhooks: none authorized by onboarding.

Any worker route, wake behavior, schedule, credential expansion, paid service, or Production automation requires a separate Project Systems review and Eduardo approval.

## Protected boundaries

- No direct material writes to `main`; use isolated branches and PRs.
- Preserve collaborator reserved-matter gates.
- No model/product redesign merely for onboarding.
- No sportsbook account access or automated wager placement.
- No credential/permission expansion without explicit approval.
- No Production/deployment change without explicit approval.
- No cross-project physical worker reuse without Project Systems review.
- No direct Baseball asset transfer into Hockey while provenance/reuse rights remain unresolved.

## Onboarding acceptance still pending

The project should not be called fully onboarded until:

1. `AGENTS.md`, `PROJECT_STATUS.md`, `CHATGPT_PROJECT_INSTRUCTIONS.md`, and focused governance/architecture/workflow docs are reviewed and merged;
2. the Project Systems registry entry is reviewed and merged;
3. the ChatGPT Project is created and its live Project Instructions are verified as an exact copy of committed `CHATGPT_PROJECT_INSTRUCTIONS.md`;
4. worker/runtime automation remains explicitly deferred or is separately onboarded with evidence.

## Immediate continuation

**STATE:** REVIEW — collaborative onboarding contract prepared for human/collaborator review.

**NEXT OWNER:** Eduardo for onboarding PR review; Rabin only where the review changes a reserved shared matter or where Eduardo chooses to request collaborator confirmation.

**NEXT ACTION:** review the onboarding PR, especially the collaboration reserved-matter wording and deployable ChatGPT Project Instructions. Do not merge until Eduardo explicitly approves the human gate.

**EDUARDO ACTION NOW:** review when Project Systems returns the PR evidence.
