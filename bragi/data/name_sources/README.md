# Ordinary Name Source Lists

These compact first-name lists support non-fantasy scenario generation in Bragi.
They are derived from ordinary U.S. name popularity data published by the Social
Security Administration through data.gov.

Source dataset:

- Baby Names from Social Security Card Applications - National Data
- Publisher: Social Security Administration
- data.gov identifier: US-GOV-SSA-338
- Dataset page: https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-data
- SSA background: https://www.ssa.gov/oact/babynames/background.html
- License metadata: https://creativecommons.org/publicdomain/zero/1.0/

Notes:

- The upstream SSA ZIP is not downloaded at runtime.
- The lists are intentionally compact and bundled for deterministic offline use.
- The issue originally suggested lists from `organisciak/names`, but that
  repository has no published GitHub license metadata, so Bragi does not bundle
  those files.
