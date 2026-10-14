# Changelog

All notable changes to this repository should be documented in this file.

This file follows a simple changelog format with dated entries.

Version numbers in this repository follow the `MAJOR.MINOR.PATCH` pattern:

- `PATCH`
  Increase the last digit for small, backward-compatible changes such as bug fixes, minor configuration corrections, small refactors, documentation updates, or limited operational fixes.
- `MINOR`
  Increase the middle digit for new, backward-compatible functionality such as a new `SUBJECT` solution, a relevant new analysis flow, new jobs, or new framework capabilities that do not break existing solutions.
- `MAJOR`
  Increase the first digit for breaking changes such as architecture changes, naming convention changes, structural changes that require solution updates, or removal of previously supported behavior.

Recommended examples:

- `0.1.0` -> `0.1.1`
  Small fix in configuration, helper behavior, notebook logic, or documentation.
- `0.1.1` -> `0.2.0`
  New solution, new important capability, or new compatible repository feature.
- `0.2.0` -> `1.0.0`
  Breaking architectural or operational change that requires adaptation in existing solutions.

Each release should separate changes into these categories:

- `Architecture`
  Changes to the shared MLOps framework, bundle structure, common helpers, targets, deploy flow, naming conventions, environment rules, or evaluator behavior.
- `New Solutions`
  Addition of a brand-new `SUBJECT` solution.
- `Solution Changes`
  Changes inside an existing solution, such as new `analysis_name` variants, feature updates, training changes, prediction updates, reports, or subject-specific fixes.
- `Documentation`
  README, changelog, onboarding, and repository guidance updates.
- `Operational`
  Changes to evaluator jobs, operational routines, or deployment checks that are not general architecture changes.

## [Unreleased]

### Architecture

- Updated `models/00_config.py` so development also reads from production lakehouse layers.
- Clarified the environment behavior in the documentation and configuration:
  - `dev` reads from `prod_silver` and `prod_gold`
  - `train` reads from `prod_silver` and `prod_gold`
  - `prod` reads from `prod_silver` and `prod_gold`
  - `dev` and `train` write to `ml_train`
  - `prod` writes to `ml_prod`
- Refined the repository guidance so new implementations distinguish between:
  - creating a new `SUBJECT` solution
  - adding a new `analysis_name` inside an existing subject

### Documentation

- Initial project documentation in `README.md` covering repository structure, MLOps architecture, development workflow, naming conventions, solution creation flow, and mandatory helper functions.
- Documented that validation and alias promotion belong to the evaluator pipeline, not directly to solution training notebooks.
- Initial changelog file for tracking repository versions and notable changes.

## [0.1.0] - 2026-04-24

### Architecture

- Base Databricks Asset Bundle structure for `feature-engineering`.
- Environment targets for `dev`, `train`, and `prod`.
- Shared MLOps helpers in `core/utils.py` and `core/uc_feature_store.py`.
- Evaluator operational flow in `ops/evaluator`.

### New Solutions

- Added the `reference_ml_template` solution as the repository reference implementation.

### Operational

- Added solution jobs in `dabs/resources/jobs_solutions.yml`.

