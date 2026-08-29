## Summary

<!-- What changed, and why? -->

## Validation

<!-- List commands/checks run and their results. If none, explain why. -->

- [ ] Relevant automated tests pass, or the reason tests were not run is stated.
- [ ] Manual verification is described for behavior not covered by tests.

## Documentation impact

See the [documentation maintenance policy](https://github.com/LeanWorks-ai/leanworks/blob/main/.github/DOCUMENTATION_MAINTENANCE.md)
for ownership, automated-check scope, and review expectations.

Run the documentation guardrails from the repository root:

```bash
python scripts/check_readme_drift.py
python -m pytest tests/test_readme_drift.py
```

- [ ] I assessed whether this change affects user, operator, API, architecture,
      configuration, setup, or deployment documentation.
- [ ] I updated affected documentation in this pull request, or documented why no
      update is required below.
- [ ] Endpoint/tool names, paths, configuration keys, defaults, limits, supported
      formats, dependencies, and examples were checked against source code.
- [ ] Architecture and backend claims describe the active implementation; legacy,
      deprecated, and fallback paths are clearly labeled.
- [ ] Documentation links and Markdown rendering were checked where applicable.
- [ ] The README drift checker and its focused test suite pass, or failures and the
      reason they do not apply are documented below.

Documentation impact and evidence:

<!--
State "none" with a short reason, or list the updated files and source-of-truth
code/configuration used to validate them.
-->

## Ownership and follow-up

<!-- Name the responsible role/team; do not leave an unassigned documentation TODO. -->

- Technical accuracy reviewer:
- Operational reviewer, if deployment/runtime behavior changed:
- Follow-up issue and target milestone, if a same-PR docs update is impractical:

## Risks

<!-- Note compatibility, rollout, migration, security, or documentation-drift risks. -->
