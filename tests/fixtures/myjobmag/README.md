# MyJobMag RSS fixture

`sample.xml` is a **real captured response**, truncated to the first 8
`<item>` elements and with description bodies replaced by a redaction
marker, because MyJobMag descriptions routinely contain recruiter email
addresses and phone numbers.

- Captured: 2026-10-09
- Source: https://www.myjobmag.co.ke/jobsxml_by_categories.xml
- Original sample: 371687 bytes / 100 items; fixture: 5261 bytes / 8 items
- `link` elements are preserved verbatim: the original posting URL is
  required evidence for the run and must not be rewritten.
- Items deliberately retained even when mojibake or out-of-scope, so the
  encoding and location-evidence paths are exercised by real data.
