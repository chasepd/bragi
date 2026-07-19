# Privacy Review

Before publishing a release or sharing a source snapshot, scan checked-in
documentation and test fixtures for maintainer-specific terms. Keep those terms
only in a local temporary file; do not commit the file, its contents, or scan
output that includes fixture text.

Put one non-empty fixed string per line in a local file. Include the full name,
given-name and alias variants, plus distinctive fixture phrases that could
re-identify the person. Then run:

```bash
git grep -I -l -F -f /path/to/private-terms.txt -- README.md docs tests
```

The command prints only matching filenames. A successful privacy review has no
output and `git grep` exits with status 1. If it exits with status 0, replace or
remove the matching personal material before publishing. Treat any other exit
status as a scan error and resolve it before proceeding.

Use clearly fictional, neutral scenarios for checked-in roleplay fixtures. Do
not add real save exports, transcripts, biographies, or relationship details to
tests or documentation.
