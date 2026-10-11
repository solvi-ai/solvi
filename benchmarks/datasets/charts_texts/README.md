# charts_texts — 50 paragraphs from US federal news releases, with a hand reading of every number

The set behind `benchmarks/charts_real_texts.py`: does `solvi.experimental.charts` draw verified charts from real
texts it was not written for?

| file | what |
|---|---|
| `texts.jsonl` | 50 records: `id` (t01–t50), `agency`, `source_url`, `fetched` (the date the page was fetched), `paragraph_index` (the paragraph's place in the page's `<p>`/`<li>` sequence), `words`, `text` |
| `labels.json` | the hand reading: for each text every number (as written, where, its value with its sign, its unit, what it is a value of, or that it is a year / a date / a table number / a rank / a threshold) and the one chart a reader would draw (type, unit, scale, series of label → value) |
| `reads.log` | created by the benchmark's loader: one line per read of `texts.jsonl` (time, process, mode, command) |

## Licence

Every text is a work of the United States government (the Bureau of Economic Analysis, the Energy Information
Administration, the USDA Economic Research Service and National Agricultural Statistics Service, NOAA's National
Centers for Environmental Information, the Department of Energy), prepared by federal employees as part of their
duties, and therefore in the public domain in the United States (17 U.S.C. § 105). The source URL of each paragraph is
in its record.

## How the paragraphs were chosen

By hand, fetched with `curl` on 2026-10-10, one paragraph per release, no selection on how well a chart comes out (no
proposer or checker was run on any candidate):

1. **Releases** were taken in the order of each agency's own listing page (newest first): BEA news releases
   (`bea.gov/news/current-releases`, then the BEA RSS feed), EIA *Today in Energy* (`eia.gov/todayinenergy`, then
   the next older article ids), EIA press releases (`eia.gov/pressroom/releases.php`), USDA ERS *Amber Waves*
   (`ers.usda.gov/amber-waves`), USDA NASS newsroom (`nass.usda.gov/Newsroom`), NOAA NCEI monthly climate reports
   (`ncei.noaa.gov/news`), DOE newsroom (`energy.gov/newsroom`). Pages that could not be fetched with curl (bls.gov,
   usda.gov, cdc.gov, ssa.gov, transportation.gov: HTTP 403; census.gov: the listing is built by JavaScript) were
   left out. Scanned and yielding nothing: noaa.gov news (7), fda.gov press announcements (10), nasa.gov news
   releases (11), usgs.gov news releases (5), fisheries.noaa.gov stories (9), and some releases of the agencies above.
2. **The paragraph**: in each release, the first paragraph (in page order) of 60–250 words that holds at least three
   numbers of a comparable structure — values that would sit on one axis: shares, counts or amounts by category, a
   series over periods. Paragraphs that are lists of table titles, boilerplate or notes were not taken. A release
   without such a paragraph was skipped.
3. **Caps decided before labelling**: EIA press releases were capped at the first three that qualified (the STEO
   releases repeat one structure, the Brent price path), and a paragraph repeating one already in the set nearly word
   for word (an EIA article restating another's WTI/breakeven paragraph) was skipped.

Every release scanned and its outcome is in `selection.tsv` (162 releases scanned, 50 taken).

Counts: BEA 13, EIA 18 (15 *Today in Energy*, 3 press releases), USDA ERS 10, NOAA NCEI 7, USDA NASS 1, DOE 1.
Lengths 61–191 words. The texts are 2026 releases (the fetch date is 2026-10-10).

## Labels

Written by the builder of the benchmark (an AI agent, Claude, labelling by hand reading — not a human annotator;
stated as such), before any proposer was run on these texts. For each text:

- **every number**: the run of digits or the number word as written (`as_written`, its `occurrence` among identical
  whole-number matches and its `start`/`end`), `role` = `value` for a quantity a chart could show, otherwise what it is
  (`year`, `date`, `reference` for "table 2", `rank`, `threshold`, `age`, `ratio`, `count` for "all 50 states", `name`
  for "COVID-19"); for a value: `value` with the sign the text means ("decreased 29.1 percent" → −29.1; "less than
  0.1 percent" is labelled 0.1 with the bound in `of`), `unit` (a base unit from the `units` table) and `of` (what it
  is a value of). A generator checks that every run of digits in a text lies inside a labelled number.
- **how a drawn point is judged** (bar-2): `label_keywords` — groups separated by `;`, alternatives by `|`; a drawn
  point's label agrees when every group has an alternative in the point's label, its series name or the chart's title
  (case-insensitive), and none of `label_excludes` is there; e.g. `"urban ; 2022|from"` for the urban obesity rate of
  March 2022–March 2023 (a label must say which of the four rates it is). The unit agrees when the chart's unit (as
  `solvi.experimental.charts.canon_unit` canonicalises it) is one the `units` table accepts for the number's base unit
  and the drawn magnitude × scale is the labelled magnitude in that unit (a "$ per barrel" price drawn in USD agrees; a
  "billion cubic feet per day" volume drawn as "cubic" does not). The sign is compared and reported apart.
- **the one chart a reader would draw** (`chart`): kind, unit, scale, series of `[label, value]`, a note when there is
  a choice. Every text in the set has one (the selection rule asks for three comparable numbers).

`text_sha256` ties each label to its text.
