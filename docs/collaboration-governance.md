# Collaboration governance

## Purpose

`rabin0208/baseball_moneyline` is a genuinely collaborative project. Project Systems onboarding must preserve meaningful collaborator rights without forcing Rabin to review every routine reversible change.

This document defines the project-local collaboration boundary. It does not transfer ownership, assign IP beyond what the collaborators have actually agreed, or replace any separate legal agreement.

## Day-to-day delegated operating authority

Eduardo + ChatGPT may coordinate routine reversible work when it stays within already-approved product direction and does not cross a reserved shared matter. Examples include:

- issue/PR administration and backlog organization;
- documentation and Project Systems mechanics;
- bounded experiments and technical investigations;
- bug fixes and maintenance;
- reversible implementation work on isolated branches;
- validation, benchmark, and evaluation improvements that do not materially redefine the product or shared rights;
- ordinary review/merge coordination when no reserved matter is implicated and existing repository permissions/policies permit it.

This delegation is intended to reduce coordination overhead. It does not make Rabin nominal, advisory-only, or subordinate in the shared project.

## Branch-first collaboration model

The default working pattern is deliberately asynchronous:

1. Eduardo + ChatGPT may specify, implement, iterate, and validate routine reversible work on an isolated branch without asking Rabin for intermediate approval.
2. Rabin does not need to participate in routine planning, branch-level iteration, experiments, documentation edits, or technical refinement merely because the repository is collaborative.
3. Once a branch is solid, validated, and ready to enter shared `main`, Eduardo may ask Rabin for one consolidated PR review/approval or merge when practical. This is the preferred low-overhead collaboration point rather than involving him throughout the work.
4. Rabin approval is a governance requirement before merge only when the change crosses a reserved shared matter below or another explicit repository rule requires it. Routine reversible work should not be blocked on repeated collaborator approvals.
5. No material work should be written directly to `main`; branch/PR history remains the auditable collaboration surface.

This model keeps Rabin meaningfully involved where shared ownership actually matters while allowing Eduardo + ChatGPT to make real progress independently between integration points.

## Reserved shared matters

Rabin must meaningfully participate before a decision is treated as approved when it concerns any of the following:

1. **Collaborator ownership or governance** — changing substantive collaborator rights, decision authority, repository ownership/control, or the collaborative operating model itself.
2. **Fundamental product direction** — a material change to what the shared project is, its primary purpose, or its core modeling/product thesis rather than an ordinary implementation iteration.
3. **Material provenance, reuse, or IP disposition** — licensing, transferring, relicensing, directly reusing, or authorizing substantial shared code, data, trained models, documentation, research outputs, or other shared assets outside this project when rights are not already clear.
4. **Publication or commercialization of shared work** — public release, sale, licensing, paid productization, commercial partnership, or other external exploitation that materially relies on shared project work and goes beyond the project's already-established public repository/dashboard operation.
5. **Destructive disposition of substantial shared work** — deleting, abandoning, privatizing, transferring, or irreversibly replacing a material portion of shared work or project history.
6. **Other decisions that materially redefine or dispose of substantive shared work** — when a reasonable collaborator would expect joint participation because the change goes beyond routine delegated development.

When uncertain whether a change is reserved, stop and ask rather than silently treating it as routine.

## Review mechanics

Reserved-matter approval does not require Rabin to inspect every implementation detail or every routine PR. The relevant decision should be summarized clearly, with the exact consequence, tradeoff, and requested approval. Once the reserved matter is resolved, routine implementation may proceed under the normal branch/PR workflow unless the collaborators explicitly require additional review.

Routine reversible work should remain auditable in GitHub. Significant ambiguity returns to Eduardo and, when applicable, Rabin before execution.

## Baseball → Hockey reuse boundary

The planned Hockey project may learn architectural, modeling, evaluation, and product concepts from Baseball.

Until collaborator reuse/provenance rights are explicitly resolved, do **not** directly copy or transfer substantive Baseball assets into Hockey, including code, data, trained models, notebooks, documentation, or other shared artifacts. Prefer independent reimplementation of concepts when reuse rights are unclear.

A future explicit collaborator decision may narrow or relax this boundary.

## Project Systems / runtime boundary

Project Systems may onboard governance files and registry metadata without creating a worker route or changing runtime behavior. Any dedicated execution worker, cross-project worker sharing, webhook, schedule, credential/permission expansion, paid provider commitment, or Production automation requires separate Project Systems review and Eduardo approval, plus collaborator participation if the change also crosses a reserved shared matter.
