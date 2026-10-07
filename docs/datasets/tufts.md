# Dataset card: Tufts Dental Database (tufts_test)

A second, independent test set for tooth numbering (task 1.14). It comes from a different
source (Tufts University School of Dental Medicine, USA) than DENTEX, so scores on it show
whether the model works beyond the data it was trained on. **Test only: never train or tune on it.**

Rebuild: put the download in `data/raw/tufts/`, then
`python scripts/tufts_to_yolo.py --raw data/raw/tufts --out data/yolo/tufts_test` (about a minute).
It creates `data/yolo/tufts_test/` (125 MB) with a single `test` split and the same 32 classes
in the same order as DENTEX (`opg.data.dentex.TEETH_CLASSES`).

## Source and licence

- Website: http://tdd.ece.tufts.edu/ (Panetta Visualization, Sensing & Simulation Research
  Laboratory, Tufts University). Access needs a request form; the download link is sent on approval.
- 1,000 panoramic radiographs with expert tooth outlines and abnormality labels, plus
  eye-tracking and think-aloud data (not used here).
- Cite (the terms require **both** in any publication):
  - Website: http://tdd.ece.tufts.edu/
  - Paper: K. Panetta, R. Rajendran, A. Ramesh, S. P. Rao and S. Agaian, "Tufts Dental Database:
    A Multimodal Panoramic X-Ray Dataset for Benchmarking Diagnostic Systems," *IEEE Journal of
    Biomedical and Health Informatics*, vol. 26, no. 4, pp. 1650–1659, 2022,
    doi: 10.1109/JBHI.2021.3117575. (The terms list it as 2021, the early-access year.)
- Terms of use (from the access request form; summary, the full text is on the form):
  - Non-commercial research and education only.
  - No access for "research associates and colleagues", and no distribution to anyone who
    hasn't agreed to the terms themselves.
  - No reproducing the database or posting it "into any webpage or any other storage means".
  - Images may be shown only in research papers or at research conferences.
  - No scraping or recording with automated programs.
  - Contact: panettavisonsensinglab@gmail.com

## How we got it (date requested, date approved)

- Requested: 5 Oct 2026, by Krisha, college email, for an academic student project supervised
  by Dr. Nirali Nanavati.
- Approved: 6 Oct 2026.

## What's in it (images, teeth, image size)

```
Radiographs/1.JPG ... 1000.JPG          1,000 images, all 1615 x 840 px
Segmentation/teeth_polygon.json         tooth outlines + boxes  <- what we read
Segmentation/teeth_bbox.json            same boxes, no outlines
Segmentation/teeth_mask/, maxillomandibular/   mask images (1,000 each)
Expert/ (expert.json, mask/, gaze_map/), Student/Student/ (same)   abnormalities, eye-tracking
```

- Annotations are a Labelbox export, one entry per image:
  `{"External ID": "797.jpg", "Label": {"objects": [{"title": "1", "bounding box": [...], "polygons": [[[x, y], ...], ...]}]}}`.
- The tooth number is in `title` (Universal). The shape is in `polygons`: **several outlines per
  tooth**, traced from a mask, as `[x, y]` pixel pairs.
- 26,005 labelled teeth: **25,259 permanent** and 746 primary (A–T).
- Permanent teeth per image (968 images with teeth): 1 to 32, mean 26.1. 157 images have all 32;
  126 have fewer than 20.
- Wisdom teeth are the rarest: 18 = 354, 28 = 367, 38 = 379, 48 = 380 (roughly 750–950 for every other tooth).
- Median tooth box: 63 × 159 px.

## Conversion (Universal to FDI, outlines to boxes)

- Tooth number: `title` → FDI with `opg.fdi.from_universal` (Universal 1 = 18, 16 = 28, 17 = 38,
  32 = 48). All 52 titles found (1–32, A–T) are valid; `--list-titles` shows none unmapped.
- Box: smallest/largest x and y of the tooth's outlines, in original-image pixels. Each tooth's
  outline list also contains many specks of a few pixels, a few of them far from the tooth (on
  853.JPG one speck would stretch tooth 32's box across the whole image). So outlines smaller
  than 10% of the tooth's largest outline are dropped first (`SPECK_FRACTION`; 5%, 10% and 20%
  give identical boxes). Only 3 teeth keep two large pieces: 38.JPG tooth 7, 744.JPG tooth 17
  (tooth traced in two parts) and 85.JPG primary tooth H (left out anyway).
- The supplied `bounding box` is **not used**: its order is `[y1, x1, y2, x2]`, it sits a few
  pixels inside the outline, and some are broken (797.JPG teeth 4, 10, 15, 21, 22, 23 have 1–4 px boxes).
- No flipping. The images carry an "L" marker at the bottom right (patient's left on image's
  right), the same orientation as DENTEX.
- Checked by eye on 6 Oct 2026: 20 random images (`--random`, seed 0) in FDI and 3 in Universal.
  Every box sits on a tooth; 18 is top far left and 48 bottom far left in all of them.

## Missing teeth and anything odd

- **32 images have no teeth at all** (edentulous patients; all 32 checked by eye, 302 and 417 show small metal attachments but no teeth: 2, 6, 19, 107, 118,
  140, 223, 261, 283, 292, 302, 327, 330, 339, 417, 459, 501, 503, 515, 526, 528, 612, 666, 706,
  748, 771, 845, 856, 939, 945, 974, 1032). They are kept with empty label files: a good model
  should find no teeth there.
- **82 images are children with primary teeth** (mixed dentition). The 746 primary teeth have no
  class (DENTEX classes are the 32 adult teeth), so they are left out of the labels, but they
  are still visible in the image. A model trained on adults may number them as permanent teeth,
  which counts against it. Scoring with and without these 82 images is worth reporting.
- No image labels the same tooth number twice (same check as DENTEX's `repeated_numbers`).
- Image names: the export uses lower case (`1.jpg`), the files are `1.JPG`; the reader matches
  ignoring case and keeps the real file name.
- Images are numbered 1 to 1,051 with gaps (1,000 files).

## Can it be shared inside the team?

**No.** The terms forbid giving colleagues access and copying the data to "any other storage
means", so there is **no `tufts_yolo_v1.zip` on Drive**, and the converted set can't be passed
around either (it contains the images).

Anyone who needs the set (e.g. to score a model on it) requests access at http://tdd.ece.tufts.edu/
with their own college email, puts the download in `data/raw/tufts/`, and runs
`python scripts/tufts_to_yolo.py`. The conversion is deterministic, so everyone gets the same set.

Also keep Tufts images (including `view_labels.py` check images) off WhatsApp, Drive and GitHub.
Showing them in the class presentation is not clearly covered by "research papers or research
conferences": ask the supervisor before putting a Tufts X-ray on a slide. Scores and tables are fine.
