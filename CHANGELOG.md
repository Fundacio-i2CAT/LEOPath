# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-06-17
### Added
- DRA-style hop-only logical-coordinate routing baseline (`dra_routing`).
- Pivot-weighted topological routing capturing inter-plane ISL asymmetry.
- Explicit-path routing family with strict and dynamic final-egress modes,
  refresh-interval control, and SRv6-style header-byte accounting.
- Seam (cylinder) +Grid topology for geometrically honest evaluation.
- CesiumJS constellation viewer with selectable route overlay and GSL controls.
- MkDocs documentation site and evaluation run matrix tooling.
### Notes
- Release archived for citation in the Computer Networks manuscript on
  topological forwarding for LEO mega-constellations.

## [0.1.1] - 2025-11-25
### Documentation
- Added PyPI installation instructions and badges.
- Updated project URLs in metadata.

## [0.1.0] - 2025-11-25
### Added
- Initial release of LEOPath simulator.
- Basic routing algorithms (Shortest Path, Topological).
- Visualization tools.
- Docker support.
