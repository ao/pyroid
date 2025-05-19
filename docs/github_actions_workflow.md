# GitHub Actions Workflow Documentation

## Overview

This document describes the GitHub Actions workflow for the Pyroid project. The workflow is designed to automate testing and publishing of the package.

## Workflow Components

### 1. Tests Workflow (tests.yml)

The tests workflow runs automated tests to ensure code quality and functionality.

**Triggers:**
- Push to any branch
- Pull requests to main branch

**Jobs:**
- **rust-tests**: Runs Rust tests on Ubuntu, Windows, and macOS
- **python-tests**: Runs Python implementation tests on Ubuntu

### 2. Publish Workflow (publish.yml)

The publish workflow builds and publishes the package to PyPI when a new version is tagged.

**Triggers:**
- Push of tags matching the pattern "v*" (e.g., v0.1.0, v1.2.3)
- Manual workflow dispatch with version input

**Jobs:**
- **linux**: Builds wheels for Linux
- **windows**: Builds wheels for Windows
- **macos**: Builds wheels for macOS (x86_64 and aarch64)
- **sdist**: Builds source distribution
- **release**: Collects all artifacts and publishes to PyPI

**Version Management:**
- The version is extracted from the tag name (removing the 'v' prefix)
- The extracted version is used to update version strings in:
  - pyproject.toml
  - Cargo.toml
  - python/pyroid/__init__.py
  - setup.py

## Release Process

To release a new version:

1. Ensure all changes are committed and pushed to the main branch
2. Create and push a new tag with the desired version:
   ```
   git tag v1.2.3
   git push origin v1.2.3
   ```
3. The publish workflow will automatically:
   - Extract the version from the tag (1.2.3)
   - Update version strings in all relevant files
   - Build wheels for all supported platforms
   - Publish the package to PyPI

## Benefits of This Approach

1. **Simplicity**: The release process is straightforward and easy to understand
2. **Manual Control**: Releases are explicitly triggered by creating tags
3. **Consistency**: The version is always derived from the tag name
4. **Quality Assurance**: Tests run on every push to ensure code quality