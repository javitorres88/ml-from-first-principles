# Instructor notes

Notes for teaching with this site — in a course, a reading group, or on your own. The site is a companion to the
CSC 74020 lecture notes, not a replacement: the lectures carry the argument, the site carries the practice.

## How the pieces fit a course

| Component | Contact time | Use it for |
| --- | --- | --- |
| Module pages | ~20 h of reading + lab time in total (see the per-module estimate in the header) | preparation before class, or self-study |
| Interactive labs | 5–10 min each, 3 per module | in-class demo, or "predict then run" exercises |
| Quizzes + flashcards | 10 min per module | entry ticket / exit ticket, spaced repetition |
| Practice problems | 20–30 min per module | homework with worked solutions released afterwards |
| Portfolio projects | 6–10 h each | the assessable artefact; do four to build a portfolio |

Suggested pace for a 14-week semester, one module per week with two weeks of slack for modules 07, 10–12 (the
deep-learning block is the densest):

```
W1  01                  W8   09
W2  02                  W9   10
W3  03                  W10  11
W4  04                  W11  12
W5  05                  W12  13
W6  06                  W13  14 + 15
W7  07 + 08             W14  16 + project presentations
```

## Using the labs in class

Each lab is a question with a manipulable answer, so run them as predictions rather than demonstrations:

1. Show the initial state and ask the room what will happen when the slider moves.
2. Take two or three answers with reasons.
3. Move the slider. Ask who was right and why.
4. Only then read the lab note underneath, which states the lesson explicitly.

Labs that reliably produce discussion:

- **02** — k = 1 versus k = 15 on the same data; then the train/test split moving.
- **03** — hand-fitting the regression line, then failing to beat the dashed optimum.
- **05** — the bias–variance decomposition curve; the coefficient path as λ grows.
- **07** — four optimisers racing on an elongated bowl; learning rate past the stability threshold.
- **09** — k-means step-through on two moons, then DBSCAN on the same points.
- **12** — the causal mask with and without the √d_k scaling (watch the softmax saturate).
- **15** — value iteration versus Q-learning on the same grid; make the reward sparse.
- **16** — exact Shapley values versus raw contributions.

## Assessment design

The site is written so that assessment is not recall-based:

- **Quizzes** are diagnostic — every wrong option is a documented misconception with an explanation of why it is
  wrong. Use them as entry tickets and discuss the most-chosen distractor.
- **Practice problems** are graded on reasoning: the solutions show the derivation, not just the answer. Release
  solutions after the deadline.
- **Projects** carry a rubric with four criteria (correctness and testing 30%, experimental design 30%, analysis 25%,
  communication 15%). The "definition of done" in each brief is deliberately mechanical — every number regenerable by
  one command, every from-scratch implementation tested against a reference — so grading is about the protocol and the
  interpretation, not about whether the code ran.

A defensible course grade: 30% projects, 30% final project, 25% practice problems, 15% quizzes.

## Academic-integrity framing

The library in `python/` and the code panes in the module pages are reference implementations. Students may read
them, and should be encouraged to; they may not submit them. The project briefs ask for something the library does not
contain — a protocol, a comparison, an audit, a report — which is where the assessment lives.

## Offline and low-resource use

The site has no build step to view and no network calls: KaTeX is vendored, the only data file is
`assets/data/iris.json`, and every other dataset is generated in the browser. To hand out a copy:

```bash
git clone <repo> && cd <repo>
python -m http.server 8000     # or open index.html directly
```

For a fully static handout, zip `index.html`, `modules/`, `assets/` — that is the whole site. Progress marks are kept
in `localStorage`, per browser, so students can work through modules offline and keep their tick-marks.

## Known limitations (worth saying out loud in class)

- Everything runs on small synthetic data or iris. That is deliberate — the point is the mechanism — but it means the
  site never shows the engineering that real-scale training needs.
- PyTorch appears only as the third code pane; the deep-learning labs are hand-written forward passes and toy
  optimisers, not framework tutorials.
- Convolution, attention and the CNN labs visualise a handful of channels and one or two layers. Scale is discussed,
  not demonstrated.
- The mathematics follows the lecture notes' notation; where a second treatment differs (e.g. the bias–variance
  decomposition), the note says which convention is in use.

## Extending the site

1. Edit the partial in `content/modules/`, never the generated page in `modules/`.
2. Register any structural change (title, objectives, lab names, timing) in `tools/modules.mjs`.
3. Run `node tools/build.mjs`; the page must rebuild with no warnings.
4. Mirror any new algorithm in `python/src/mlfs/` and add a parity test — `pytest` must stay green.
5. Keep the six-part structure: intuition → mathematics → labs → code → assessment → project.

Screenshot QA (optional, needs a local Chrome): `node tools/shot.mjs http://localhost:8000/modules/foundations.html _shots/check 3`
prints any console errors the page produced while it scrolls.

## Attribution

The lecture material, notation and topic sequence belong to Prof. Jonathan Gryak (CUNY Graduate Center, CSC 74020,
Fall 2025). If you reuse these notes in another course, keep the attribution in `README.md` and on the landing page.
