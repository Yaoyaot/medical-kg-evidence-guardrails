# Anonymous review release checklist

Before updating the reviewer-facing mirror:

- run `python scripts/rebuild_repository_manifest.py`;
- run `python reproduce_quick.py`;
- run `python reproduce_frozen_results.py`;
- search text files for author names, affiliations, emails, ORCID identifiers,
  public repository URLs, local usernames, and absolute paths;
- confirm that no API credential, UMLS file, raw response archive, manuscript,
  title page, debug log, cache, or temporary output is tracked;
- download the Anonymous GitHub ZIP and rerun both reproduction commands in a
  clean environment without `.git`;
- verify that every file is smaller than the Anonymous GitHub per-file limit;
- pin the reviewer mirror to the verified final commit.

Only text files are transformed by Anonymous GitHub. Binary files must be
treated as non-anonymized and should not be included unless independently
audited.
