# Documentation maintenance

Documentation is part of the change, not a follow-up by default. Any pull request
that changes user-visible behavior, configuration, deployment, supported inputs,
or architecture must update the affected documentation in the same pull request.

## Ownership

- The pull request author owns the documentation-impact assessment and either
  updates the documentation or explains why no update is needed.
- A maintainer of the affected subsystem reviews technical accuracy. Documentation-
  only changes should still be reviewed by a maintainer of the code or deployment
  configuration they describe.
- The release or deployment reviewer validates operational instructions when a
  change affects runtime configuration, dependencies, images, or infrastructure.
- If a same-PR update is genuinely impractical, the pull request must link a
  tracked follow-up, identify the responsible role or team, and state the target
  milestone. A vague promise to update documentation later is not sufficient.

These responsibilities are role-based because the repository does not currently
document stable GitHub user or team handles for CODEOWNERS assignments.

## Sources of truth

Validate documentation claims against implementation rather than copying older
documentation:

- Main application API routes and request/response behavior: `app/api/`
- Main application service behavior: `app/services/`
- Session-manager API routes and lifecycle behavior:
  `leanworks/agent/session_manager/app.py`
- Agent tools and document workflows: `leanworks/agent/tools/`
- RAG backends, indexing, and retrieval: `leanworks/rag/`
- Defaults and feature configuration: `leanworks/setting.py`
- Environment resolution: `leanworks/utils/env.py`
- Packaging and dependencies: `requirements*.txt`, `setup.py`, and `pyproject.toml`
- Deployment behavior: `deploy/` and `deploy.sh`
- Supported behavior and regressions: `tests/`
- Documentation automation: `.github/workflows/readme-drift.yml` and
  `scripts/check_readme_drift.py`

When sources disagree, treat executable code and deployed configuration as the
current behavior and either correct the documentation or call out the discrepancy
for maintainers.

## Automated-check scope

The README drift workflow is a focused guardrail, not comprehensive documentation
validation. It checks the root `README.md` for retired Pinecone claims, documented
main-application route paths and explicit HTTP verbs, `Chat`/`AsyncChat` imports
and basic constructor/method call compatibility, and a selected allowlist of
configuration defaults.

It does not validate every prose claim, required call argument, async usage,
external service route, session-manager route, link, code example, deployment
instruction, or other documentation file such as `deploy/README.md`. Manual
review against the sources of truth above remains required even when the
automated check passes.

## Required review checks

For documentation-affecting changes, verify all applicable items:

- Names and paths for endpoints, tools, modules, files, and commands are current.
- Configuration keys, environment variables, defaults, limits, and supported file
  types match the implementation.
- Architecture diagrams and prose identify the active backend and do not present
  legacy or fallback paths as the default.
- Setup, test, build, and deployment commands were exercised or the reason they
  were not run is recorded in the pull request.
- Examples are internally consistent and do not reference removed files or routes.
- Links and anchors resolve, and rendered Markdown is readable.
- Async behavior, failure modes, compatibility paths, and deprecations are stated
  where they affect users or operators.
