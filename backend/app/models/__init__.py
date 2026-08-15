"""Persistence-facing document builders.

Kept separate from :mod:`app.schemas` on purpose: the API contract and the
storage layout have different requirements. In particular the stored analysis
document is *privacy-shaped* - see :func:`app.models.analysis.build_analysis_document`.
"""
