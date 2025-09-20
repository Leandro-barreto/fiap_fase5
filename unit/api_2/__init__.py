"""Package for API route tests.

This directory mirrors the ``api`` tests but has been renamed to
``api_2`` to work around issues with import resolution when running
pytest.  By providing an ``__init__.py`` the directory becomes a
package and test discovery functions as expected.
"""
