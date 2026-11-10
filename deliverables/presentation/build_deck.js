// Rebuild the deck: npm install pptxgenjs@3.12.0, then: node build_deck.js
// (uses the pptx skill's apply_theme.js; set NODE_PATH to the folder holding node_modules)
// Presentation deck: Classifying imagined shapes from EEG (NIT Warangal, M.Tech CSE)
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require("C:/Users/mjsid/.claude/skills/synced/59a347e8-e788-4e13-828e-91f1c7ad917f_01037e08-f5bc-472a-b495-7f54cf1e50ff/pptx/scripts/apply_theme.js");

const ROOT = "C:/Users/mjsid/Documents/AA Project/deliverables";
const FIG = (f) => path.join(ROOT, "report/figures", f);
const OUT = path.join(ROOT, "presentation/Imagined_Shape_EEG_Presentation.pptx");

const P = {
  navy: "1B2A4A", navy2: "2C3E66", teal: "128C7E", tealLight: "D9F0EC", coral: "E4572E",
  ink: "1F2937", muted: "5B6472", tint: "EEF3F8", white: "FFFFFF", line: "D5DCE6",
};
const THEME = {
  name: "NITW EEG", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1F2937", lt1: "FFFFFF", dk2: "1B2A4A", lt2: "EEF3F8", accent1: "128C7E",
    accent2: "E4572E", accent3: "2A78D6", accent4: "EDA100", accent5: "6250D6", accent6: "1BAF7A",
    hlink: "2A78D6", folHlink: "6250D6" },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Classifying imagined shapes from EEG signals";
pres.author = "A. N. Rao, P. A. Deepakkumar, M. J. Siddhartha";
pres.subject = "M.Tech CSE project, NIT Warangal";

const W = 13.333, H = 7.5, MX = 0.6;
const FOOT = "NIT Warangal  |  Imagined-Shape EEG Classification";

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: P.navy },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: P.white },
  margin: [0.4, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.32, w: W - 2 * MX, h: 0.85,
      fontFace: "Cambria", fontSize: 30, bold: true, color: P.navy, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: MX, y: H - 0.45, w: 7, h: 0.3, fontSize: 10, color: P.muted, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: H - 0.45, w: 0.6, h: 0.3, fontSize: 10, color: P.muted, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING_DARK", background: { color: P.navy },
  objects: [],
  slideNumber: { x: W - MX - 0.6, y: H - 0.45, w: 0.6, h: 0.3, fontSize: 10, color: "AAB6CC", align: "right" },
});

// ---------------------------------------------------------------- helpers
const T = (slide, text, o) => slide.addText(text, { isTextBox: true, fontFace: "Calibri", color: P.ink, margin: 0, ...o });
const card = (slide, x, y, w, h, fill, name) => slide.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: name });
const badge = (slide, x, y, n, fill) => {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.46, h: 0.46, fill: { color: fill }, line: { color: fill } });
  T(slide, String(n), { x, y, w: 0.46, h: 0.46, fontSize: 16, bold: true, color: P.white, align: "center", valign: "middle" });
};
const stat = (slide, x, y, w, big, label, color) => {
  T(slide, big, { x, y, w, h: 0.75, fontFace: "Cambria", fontSize: 36, bold: true, color: color || P.navy });
  T(slide, label, { x, y: y + 0.75, w, h: 0.55, fontSize: 13, color: P.muted, valign: "top" });
};
const bullets = (items, size = 16) => items.map((t, i) => ({
  text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6, fontSize: size } }));
const img = (slide, file, x, y, w, ratio, alt) => slide.addImage({ path: FIG(file), x, y, w, h: w / ratio, altText: alt });

// ================================================================ 1. title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
card(s, MX, 0.55, 1.35, 1.35, P.white, "logo card");
s.addImage({ path: FIG("nitw_logo.png"), x: MX + 0.1, y: 0.65, w: 1.15, h: 1.15, altText: "NIT Warangal logo" });
T(s, "NATIONAL INSTITUTE OF TECHNOLOGY, WARANGAL", { x: 2.25, y: 0.75, w: 9, h: 0.45, fontSize: 18, bold: true, color: P.white });
T(s, "M.Tech, Department of Computer Science and Engineering  ·  Project Report", { x: 2.25, y: 1.2, w: 10, h: 0.4, fontSize: 14, color: "C9D3E6" });
T(s, "Classifying imagined shapes from EEG signals", { x: MX, y: 2.45, w: W - 2 * MX, h: 1.0, fontFace: "Cambria", fontSize: 40, bold: true, color: P.white });
T(s, "Stability-aware selection of channels, frequency bands and time windows (circle / square / pentagram)", { x: MX, y: 3.45, w: W - 2 * MX, h: 0.6, fontSize: 20, italic: true, color: "9FE3D6" });
const team = [["Abhay Nagesh Rao", "26CSM1R01"], ["Pandya Aditya Deepakkumar", "26CSM1R02"], ["Mulakalapalli Jayachandra Siddhartha", "26CSM1R07"]];
team.forEach(([n, r], i) => {
  const x = MX + i * 4.1;
  T(s, n, { x, y: 5.2, w: 3.9, h: 0.4, fontSize: 16, bold: true, color: P.white });
  T(s, "Roll No.: " + r, { x, y: 5.6, w: 3.9, h: 0.35, fontSize: 13, color: "C9D3E6" });
});
T(s, "National Institute of Technology Warangal", { x: MX, y: H - 0.75, w: 8, h: 0.35, fontSize: 12, color: "AAB6CC" });
s.addNotes("Good morning. Our project asks whether we can tell which shape a person is imagining from their EEG, and whether automatically selecting a small set of EEG channels, frequency bands and time windows helps. We will cover the problem, our method, a strict evaluation, the results, and what they teach us.");

// ================================================================ 2. problem
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Can we read an imagined shape from EEG?", { placeholder: "title" });
T(s, bullets([
  "Visual-imagery BCI: the participant imagines a shape; a classifier infers it from the EEG epoch",
  "Imagery signals are weak and vary across people, sessions, time windows and frequency bands",
  "Key challenge: find reliable information without selecting features that only fit one split",
  "Our question: does a stability-aware genetic algorithm that jointly selects channels, bands and windows beat using everything?",
], 17), { x: MX, y: 1.45, w: 6.9, h: 4.2, valign: "top" });
// shape icons
const sx = 8.2;
card(s, sx, 1.5, 4.5, 3.6, P.tint, "classes card");
s.addShape(pres.shapes.OVAL, { x: sx + 0.45, y: 2.0, w: 1.0, h: 1.0, fill: { color: P.white }, line: { color: P.teal, width: 3 } });
s.addShape(pres.shapes.RECTANGLE, { x: sx + 1.75, y: 2.0, w: 1.0, h: 1.0, fill: { color: P.white }, line: { color: P.teal, width: 3 } });
s.addShape(pres.shapes.STAR_5_POINT, { x: sx + 3.05, y: 1.95, w: 1.1, h: 1.1, fill: { color: P.white }, line: { color: P.teal, width: 3 } });
["circle", "square", "pentagram"].forEach((t, i) => T(s, t, { x: sx + 0.3 + i * 1.3, y: 3.1, w: 1.3, h: 0.35, fontSize: 14, color: P.ink, align: "center" }));
T(s, "3 classes  ·  chance = 33.3 %", { x: sx + 0.3, y: 3.75, w: 3.9, h: 0.45, fontSize: 18, bold: true, color: P.navy, align: "center" });
T(s, "Proposal: circle / square / triangle / hexagon. No public dataset has triangle and hexagon, so we used the 3-class benchmark (as planned).",
  { x: sx + 0.3, y: 4.25, w: 3.9, h: 0.75, fontSize: 12, color: P.muted, align: "center" });
s.addNotes("A brain-computer interface infers intent from brain activity. Here the participant imagines a shape. The signal is weak and variable, so the risk is selecting features that only work on one split. Our proposal targeted four shapes, but the only suitable public dataset has circle, square and pentagram, so as anticipated in the proposal we worked on that three-class benchmark. Chance is 33.3 percent.");

// ================================================================ 3. literature
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Related work and the research gap", { placeholder: "title" });
const hdr = (t) => ({ text: t, options: { bold: true, color: P.white, fill: { color: P.navy }, fontSize: 13 } });
const cell = (t, o = {}) => ({ text: t, options: { fontSize: 12, color: P.ink, ...o } });
s.addTable([
  [hdr("Study"), hdr("Method"), hdr("Result")],
  [cell("Esfahani & Sundararajan 2012"), cell("ICA + Hilbert-Huang spectra + LDA; 5 imagined 3D primitives"), cell("44.6 % (chance 20 %)")],
  [cell("Costa et al. 2019"), cell("Alpha-band CSP + SVM; 7 geometric figures"), cell("52 %; includes our target shapes")],
  [cell("Llorella et al. 2020"), cell("CNN with GA parameter search; 4 objects + rest"), cell("60 %; GAs already used")],
  [cell("Llorella et al. 2021"), cell("CNN + Black-Hole search; 7 shapes"), cell("35 %; many classes are hard")],
  [cell("Korik et al. 2024"), cell("FBCSP + sliding windows + MI selection + LDA"), cell("Closest pipeline; feedback helps")],
  [cell("Gao et al. 2026"), cell("Public 22-subject, 32-ch dataset; EEGNet"), cell("75.1 % on 3 figures (within-subject)")],
], { x: MX, y: 1.4, w: 7.9, colW: [2.3, 3.6, 2.0], border: { type: "solid", pt: 0.5, color: P.line }, rowH: 0.42, valign: "middle" });
card(s, 8.85, 1.4, 3.9, 3.85, P.tealLight, "gap card");
T(s, "Gap we address", { x: 9.1, y: 1.6, w: 3.4, h: 0.4, fontSize: 18, bold: true, color: P.teal });
T(s, "GAs, CSP and time windows are not new on their own. Our contribution: treat channel + band + window selection as ONE multi-objective search that weighs accuracy, repeatability and cost, and test it against matched baselines.",
  { x: 9.1, y: 2.05, w: 3.4, h: 3.0, fontSize: 15, color: P.ink, valign: "top" });
T(s, "Accuracies come from different datasets and protocols; they do not rank the methods.", { x: MX, y: 5.55, w: 7.9, h: 0.35, fontSize: 11, italic: true, color: P.muted });
s.addNotes("These are the six studies from our literature review. GAs, CSP and temporal windows have all been used before, so none is novel alone. Our narrower contribution is a joint, stability-aware search over channels, bands and windows, evaluated against matched baselines. Note the Gao dataset paper reports 75.1 percent with EEGNet; we come back to that.");

// ================================================================ 4. dataset
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Dataset: Gao et al. 2026 visual-imagery EEG", { placeholder: "title" });
[["22", "participants"], ["42", "recordings (up to 2 sessions each)"], ["32 ch", "EEG at 1000 Hz"], ["4914", "trials kept (126 of 5040 rejected, 2.5 %)"]]
  .forEach(([b, l], i) => stat(s, MX + i * 3.1, 1.45, 2.9, b, l, i === 0 ? P.teal : P.navy));
T(s, "One trial (seconds)", { x: MX, y: 3.35, w: 5, h: 0.35, fontSize: 14, bold: true, color: P.navy });
const phases = [["Fixation", 0, 3, "C9D3E6"], ["Viewing", 3, 7, "9FC4EE"], ["Mask", 7, 9, "C9D3E6"], ["Imagery", 9, 13, P.teal], ["Rest", 13, 17, "C9D3E6"]];
const tx0 = MX, tw = W - 2 * MX, scale = tw / 17;
phases.forEach(([n, a, b, c]) => {
  s.addShape(pres.shapes.RECTANGLE, { x: tx0 + a * scale, y: 3.8, w: (b - a) * scale - 0.04, h: 0.7, fill: { color: c }, line: { color: c } });
  T(s, n, { x: tx0 + a * scale, y: 3.8, w: (b - a) * scale - 0.04, h: 0.7, fontSize: 14, bold: true, align: "center", valign: "middle", color: n === "Imagery" ? P.white : P.ink });
  T(s, String(a), { x: tx0 + a * scale - 0.15, y: 4.55, w: 0.3, h: 0.3, fontSize: 11, color: P.muted, align: "center" });
});
T(s, "17", { x: tx0 + 17 * scale - 0.3, y: 4.55, w: 0.3, h: 0.3, fontSize: 11, color: P.muted, align: "center" });
T(s, bullets([
  "120 trials per session: 40 per shape in 4 blocks of 30; we decode the 4-s imagery epoch (viewing period used as a sanity check)",
  "Preprocessing: 50 Hz notch, 1–40 Hz band-pass, common average reference, 250 Hz; reject trials > 150 µV on 8 visual channels",
], 15), { x: MX, y: 5.05, w: W - 2 * MX, h: 1.3, valign: "top" });
s.addNotes("We used the public Gao 2026 dataset: 22 participants, 42 recordings, 32 channels at 1000 Hz. Each trial has fixation, viewing the shape, a mask, then 4 seconds of imagery. We decode the imagery period; the viewing period is our sanity check. Standard preprocessing; only 2.5 percent of trials were rejected.");

// ================================================================ 5. pipeline
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Approach: break the signal into selectable components", { placeholder: "title" });
const steps = [["Raw EEG", "32 ch, 1000 Hz"], ["Preprocess", "filter, CAR, 250 Hz, reject"], ["Components", "8 ch × 4 bands × 7 windows"], ["FBCSP", "log-variance features"], ["Shrinkage LDA", "→ circle / square / pentagram"]];
const bw = 2.1, gap = (W - 2 * MX - 5 * bw) / 4;
steps.forEach(([a, b], i) => {
  const x = MX + i * (bw + gap);
  card(s, x, 1.5, bw, 1.25, i === 2 ? P.teal : P.tint, "step " + a);
  T(s, a, { x: x + 0.1, y: 1.6, w: bw - 0.2, h: 0.45, fontSize: 16, bold: true, align: "center", color: i === 2 ? P.white : P.navy });
  T(s, b, { x: x + 0.1, y: 2.05, w: bw - 0.2, h: 0.6, fontSize: 12, align: "center", color: i === 2 ? P.white : P.muted });
  if (i < 4) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.07, y: 1.95, w: gap - 0.14, h: 0.35, fill: { color: P.line }, line: { color: P.line } });
});
T(s, "A selection = 19-bit chromosome (example chosen by the GA for one fold)", { x: MX, y: 3.2, w: 9, h: 0.4, fontSize: 15, bold: true, color: P.navy });
const bits = "0001011001001000000".split("").map(Number);
const genes = ["O1", "Oz", "O2", "Pz", "P3", "P4", "P7", "P8", "θ", "α", "β", "γ", "0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"];
const PITCH = 0.43, SQ = 0.37, GG = 0.22;
const groupX = (i) => MX + i * PITCH + (i >= 8 ? GG : 0) + (i >= 12 ? GG : 0);
bits.forEach((b, i) => {
  const x = groupX(i);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 3.72, w: SQ, h: SQ, rectRadius: 0.05, fill: { color: b ? P.coral : P.white }, line: { color: b ? P.coral : P.line, width: 1.25 } });
  T(s, genes[i], { x: x - 0.04, y: 4.14, w: SQ + 0.08, h: 0.28, fontSize: 10, align: "center", color: P.muted });
});
[["8 channels", 0, 8], ["4 bands", 8, 4], ["7 windows (s)", 12, 7]].forEach(([l, i0, n]) =>
  T(s, l, { x: groupX(i0), y: 4.45, w: n * PITCH - 0.06, h: 0.3, fontSize: 12, bold: true, align: "center", color: P.navy }));
card(s, 9.5, 3.2, 3.23, 2.6, P.tint, "cost card");
T(s, "Cost of a selection", { x: 9.7, y: 3.35, w: 2.9, h: 0.35, fontSize: 15, bold: true, color: P.navy });
T(s, "c = (ch/8 + bands/4 + win/7) / 3", { x: 9.7, y: 3.8, w: 2.9, h: 0.4, fontSize: 14, bold: true, color: P.ink });
T(s, "All on = 168 features, cost 1.0. The example uses 4 channels, 1 band and 1 window.", { x: 9.7, y: 4.3, w: 2.9, h: 1.2, fontSize: 13, color: P.muted, valign: "top" });
T(s, "FBCSP: one-vs-rest CSP per selected band × window, 1 filter pair per class → 6 log-variance features each; LDA with Ledoit–Wolf shrinkage copes with ~90 training trials.",
  { x: MX, y: 5.15, w: 8.6, h: 0.8, fontSize: 14, color: P.ink, valign: "top" });
s.addNotes("Our pipeline: preprocess, then split the signal into components: 8 visual channels, 4 frequency bands and 7 one-second windows. Any selection is a 19-bit chromosome; coral means selected. For each selected band and window, filter-bank CSP learns spatial filters, and a shrinkage LDA classifies. The cost is the fraction of components used; all components give 168 features.");

// ================================================================ 6. algorithm
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Proposed algorithm: stability-aware NSGA-II", { placeholder: "title" });
const objs = [["Maximise", "inner-CV accuracy μ", P.teal], ["Minimise", "inner-CV SD σ (stability)", P.navy], ["Minimise", "cost c (compactness)", P.coral]];
objs.forEach(([a, b, c], i) => {
  const y = 1.5 + i * 1.2;
  card(s, MX, y, 5.4, 1.0, P.tint, "objective " + i);
  badge(s, MX + 0.25, y + 0.27, i + 1, c);
  T(s, a, { x: MX + 0.95, y: y + 0.12, w: 4.2, h: 0.4, fontSize: 14, color: P.muted });
  T(s, b, { x: MX + 0.95, y: y + 0.47, w: 4.3, h: 0.45, fontSize: 18, bold: true, color: P.ink });
});
T(s, "Each objective uses a 3-fold inner CV on the outer training blocks only", { x: MX, y: 5.2, w: 5.4, h: 0.7, fontSize: 13, italic: true, color: P.muted, valign: "top" });
T(s, "Search", { x: 6.5, y: 1.45, w: 6, h: 0.4, fontSize: 18, bold: true, color: P.navy });
T(s, bullets([
  "NSGA-II (pymoo): population 30 × 20 generations ≈ 600 evaluations per fold",
  "Two-point crossover, bit-flip mutation 1/19, repair of invalid chromosomes",
  "Seeds 0, 1, 2; results averaged over seeds",
], 15), { x: 6.5, y: 1.9, w: 6.2, h: 1.7, valign: "top" });
T(s, "Final choice: the “within 1 SD” rule", { x: 6.5, y: 3.65, w: 6, h: 0.4, fontSize: 18, bold: true, color: P.navy });
card(s, 6.5, 4.15, 6.23, 1.15, P.tealLight, "rule card");
T(s, "pick the cheapest Pareto solution with  μ ≥ μ(best) − σ(best)", { x: 6.7, y: 4.25, w: 5.9, h: 0.45, fontSize: 17, bold: true, color: P.teal });
T(s, "then refit on all training blocks and test once on the held-out block", { x: 6.7, y: 4.7, w: 5.9, h: 0.45, fontSize: 14, color: P.ink });
T(s, "Baselines: FBCSP-all (no selection, 168 features) · GA single-objective (accuracy only) · EEGNet CNN with 32 or 8 channels",
  { x: 6.5, y: 5.5, w: 6.23, h: 0.7, fontSize: 14, color: P.ink, valign: "top" });
s.addNotes("The proposed method is NSGA-II with three objectives: high inner cross-validation accuracy, low variability across inner folds, which is the stability objective, and low cost. After about 600 evaluations per fold, we pick the cheapest Pareto solution within one standard deviation of the best accuracy, refit it and test it once. Baselines: all components, a GA on accuracy only, and EEGNet.");

// ================================================================ 7. evaluation
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Evaluation: strict, nested and pre-registered", { placeholder: "title" });
const ev = [
  ["Leave-one-block-out", "4 folds per session; test trials are never time-neighbours of training trials"],
  ["Cross-session", "train on session 1, test on session 2 (20 subjects)"],
  ["Nested selection", "the GA sees only training blocks; the test block is used exactly once"],
  ["Balanced accuracy", "mean per-class recall; chance = 0.333; one score per subject (n = 22)"],
  ["Statistics", "Wilcoxon signed-rank over subjects, Holm correction, permutation tests per recording"],
  ["Pre-registration", "hypotheses and tests pushed before the results were analysed"],
];
ev.forEach(([a, b], i) => {
  const col = i % 3, row = Math.floor(i / 3);
  const x = MX + col * 4.1, y = 1.5 + row * 2.2;
  card(s, x, y, 3.9, 1.95, row === 0 ? P.tint : P.tealLight, "eval " + a);
  badge(s, x + 0.2, y + 0.2, i + 1, row === 0 ? P.navy : P.teal);
  T(s, a, { x: x + 0.8, y: y + 0.2, w: 2.95, h: 0.46, fontSize: 17, bold: true, color: P.navy, valign: "middle" });
  T(s, b, { x: x + 0.25, y: y + 0.8, w: 3.45, h: 1.05, fontSize: 14, color: P.ink, valign: "top" });
});
T(s, "Primary hypothesis: GA-NSGA > FBCSP-all (block, one-sided, α = 0.05). Secondary: vs GA single-obj., cross-session, EEGNet-32, EEGNet-8 (Holm).",
  { x: MX, y: 6.0, w: W - 2 * MX, h: 0.5, fontSize: 13, italic: true, color: P.muted });
s.addNotes("The evaluation is deliberately strict. Leave-one-block-out avoids temporal leakage, and cross-session tests transfer to a new day. Selection is nested: the GA never sees the test block. Our metric is balanced accuracy with one score per subject. The tests are Wilcoxon with Holm correction, and we wrote down the hypotheses and pushed them before analysing results.");

// ================================================================ 8. primary result
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Main result: the GA does not beat using everything", { placeholder: "title" });
stat(s, MX, 1.5, 2.5, "0.345", "GA-NSGA (proposed)", P.coral);
stat(s, MX, 2.9, 2.5, "0.353", "FBCSP-all baseline", P.navy);
stat(s, MX, 4.3, 2.5, "p = 0.79", "one-sided Wilcoxon, n = 22: primary hypothesis NOT supported", P.teal);
img(s, "F01_method_comparison_block.png", 3.35, 1.55, 9.4, 2.351, "Subject-level balanced accuracy for five methods, block protocol");
T(s, "Balanced accuracy per subject (block protocol). Dashed line = chance (0.333).", { x: 3.35, y: 5.65, w: 9.4, h: 0.35, fontSize: 11, color: P.muted });
s.addNotes("This is the pre-registered primary result. The proposed GA reaches 0.345 balanced accuracy, the full baseline 0.353. The difference is -0.007 with a one-sided p of 0.79, so the hypothesis is not supported. The figure shows every method hugging the chance line.");

// ================================================================ 9. all methods
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("All methods land near chance; the GA is just cheaper", { placeholder: "title" });
const rows = [
  ["FBCSP-all", "0.353", "0.337", "2 / 42", "1.00 / 168"],
  ["GA-NSGA (proposed)", "0.345", "0.340", "2 / 42", "0.30 / 13"],
  ["GA single-obj.", "0.353", "0.328", "3 / 42", "0.48 / 39"],
  ["EEGNet (32 ch)", "0.341", "0.325", "2 / 42", "– / 3107 params"],
  ["EEGNet (8 ch)", "0.347", "0.334", "1 / 42", "– / 2723 params"],
];
s.addTable([
  [hdr("Method"), hdr("Block BA"), hdr("Cross-session BA"), hdr("Recordings > chance"), hdr("Cost / features")],
  ...rows.map((r, i) => r.map((t) => cell(t, { fontSize: 14, bold: i === 1, fill: { color: i === 1 ? P.tealLight : P.white } }))),
], { x: MX, y: 1.5, w: 8.2, colW: [2.4, 1.2, 1.6, 1.5, 1.5], border: { type: "solid", pt: 0.5, color: P.line }, rowH: 0.5, valign: "middle" });
T(s, "Secondary tests (Holm): all p = 1.00", { x: MX, y: 4.7, w: 8.2, h: 0.4, fontSize: 15, bold: true, color: P.navy });
T(s, "All five methods are within 0.012 of each other and at most 0.02 above chance; only 1–3 of 42 recordings pass Holm.", { x: MX, y: 5.1, w: 8.2, h: 0.7, fontSize: 14, color: P.ink, valign: "top" });
card(s, 9.2, 1.5, 3.53, 4.3, P.navy, "features callout");
T(s, "13", { x: 9.4, y: 1.8, w: 3.1, h: 1.1, fontFace: "Cambria", fontSize: 66, bold: true, color: P.white, align: "center" });
T(s, "features instead of 168", { x: 9.4, y: 2.9, w: 3.1, h: 0.4, fontSize: 16, color: "C9D3E6", align: "center" });
T(s, "≈ 8 %", { x: 9.4, y: 3.55, w: 3.1, h: 0.7, fontFace: "Cambria", fontSize: 34, bold: true, color: "9FE3D6", align: "center" });
T(s, "of the features, same accuracy, smallest cross-session drop (+0.002)", { x: 9.4, y: 4.3, w: 3.1, h: 1.2, fontSize: 14, color: P.white, align: "center", valign: "top" });
s.addNotes("The full table: every method lands between 0.341 and 0.353, at most two points above chance, and only one to three recordings out of 42 are significant. The secondary tests all have corrected p of 1. What the GA does deliver is compactness: 13 features instead of 168, about 8 percent, for the same accuracy.");

// ================================================================ 10. random control
pres.addSection({ title: "Why" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Why" });
s.addText("Why? A random subset of the same size does as well", { placeholder: "title" });
img(s, "F06_random_control.png", MX, 1.45, 8.3, 2.5, "Percentile of GA among same-cost random chromosomes, and GA vs random per subject");
T(s, "50 random chromosomes per fold with the same numbers of channels, bands and windows", { x: MX, y: 4.85, w: 8.3, h: 0.35, fontSize: 11, color: P.muted });
stat(s, 9.3, 1.5, 3.4, "+0.004", "GA − random at the same cost (p = 0.42)", P.coral);
stat(s, 9.3, 2.9, 3.4, "0.51", "GA's percentile among random draws (0.5 = typical)", P.navy);
card(s, 9.3, 4.35, 3.43, 1.45, P.tealLight, "takeaway");
T(s, "The search adds nothing beyond the cost reduction", { x: 9.45, y: 4.45, w: 3.15, h: 1.25, fontSize: 16, bold: true, color: P.teal, valign: "middle" });
s.addNotes("To ask why, we compared the GA's choice with 50 random selections of exactly the same size, scored identically. The GA is a typical random draw: on average 0.004 better, not significant, at the 51st percentile. So the search finds nothing beyond the cost reduction.");

// ================================================================ 11. overfitting (native chart)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Why" });
s.addText("Every model memorises its training data", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Training", labels: ["FBCSP-all", "GA single-obj.", "GA-NSGA", "EEGNet (32 ch)"], values: [0.999, 0.751, 0.576, 0.639] },
  { name: "Inner CV / validation", labels: ["FBCSP-all", "GA single-obj.", "GA-NSGA", "EEGNet (32 ch)"], values: [0.342, 0.568, 0.512, 0.364] },
  { name: "Test", labels: ["FBCSP-all", "GA single-obj.", "GA-NSGA", "EEGNet (32 ch)"], values: [0.353, 0.353, 0.345, 0.350] },
], { x: MX, y: 1.4, w: 7.8, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: ["1B2A4A", "6DA7EC", "E4572E"],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelColor: "1F2937",
  valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisMajorUnit: 0.2, valAxisLabelFontSize: 11, catAxisLabelFontSize: 12,
  valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", valAxisLabelColor: "5B6472", catAxisLabelColor: "1F2937",
  valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", valAxisTitle: "Balanced accuracy", showValAxisTitle: true, valAxisTitleFontSize: 12,
  showTitle: true, title: "Block protocol, n = 22 (chance = 0.33)", titleFontSize: 13, titleColor: "1F2937", titleFontFace: "+mn-lt" });
T(s, "What it shows", { x: 8.85, y: 1.5, w: 3.9, h: 0.4, fontSize: 18, bold: true, color: P.navy });
T(s, bullets([
  "Training accuracy 0.58–1.00, test ≈ 0.35 for every model",
  "EEGNet without early stopping reaches 1.00 on training; validation stays at chance",
  "The GA rates its pick at ≈ 0.51 but it scores 0.35: picking the best of ~600 noisy scores selects luck (winner's curse, +0.17)",
], 15), { x: 8.85, y: 1.95, w: 3.9, h: 4.0, valign: "top" });
s.addNotes("This chart explains the result. Every model fits its training trials, up to 100 percent, yet tests at about 0.35. Without selection, FBCSP's inner cross-validation is honest at 0.34. The GA picks the best of about 600 noisy estimates, so its inner score is inflated by 0.17. That is the winner's curse.");

// ================================================================ 12. controls
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Why" });
s.addText("Controls rule out the usual pitfalls", { placeholder: "title" });
const ctr = [
  ["Trial-order confound", "0.397", "position-only classifier (no EEG) across sessions: block orders repeat for 10 of 20 subjects. The EEG methods do not follow it (κ ≈ 0)."],
  ["Random-split leakage", "−0.002 to −0.011", "random 5-fold vs leave-one-block-out: no inflation for any method."],
  ["Chooser rule", "0.338–0.351", "7 alternative rules (best, knee, min cost, 0.5/2 SD, SE) on the stored fronts: none beats the baseline."],
];
ctr.forEach(([a, b, c], i) => {
  const x = MX + i * 4.1;
  card(s, x, 1.5, 3.9, 3.55, i === 1 ? P.tealLight : P.tint, "control " + a);
  T(s, a, { x: x + 0.25, y: 1.7, w: 3.4, h: 0.45, fontSize: 18, bold: true, color: P.navy });
  T(s, b, { x: x + 0.25, y: 2.3, w: 3.4, h: 0.75, fontFace: "Cambria", fontSize: i === 0 ? 34 : 26, bold: true, color: P.coral, valign: "middle" });
  T(s, c, { x: x + 0.25, y: 3.2, w: 3.4, h: 1.7, fontSize: 15, color: P.ink, valign: "top" });
});
T(s, "All exploratory (not in the pre-registered plan)", { x: MX, y: 5.3, w: 8, h: 0.35, fontSize: 12, italic: true, color: P.muted });
s.addNotes("We also checked the usual pitfalls. First, trial order: the dataset reuses block orders across sessions, so position alone predicts the class at 0.40, but our EEG methods do not exploit it. Second, random splits do not inflate anything. Third, no alternative rule for picking the final solution helps.");

// ================================================================ 13. viewing (native chart)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Why" });
s.addText("When the signal is real, selection has a price", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Imagery (9–13 s)", labels: ["FBCSP-all", "GA single-obj.", "GA-NSGA"], values: [0.353, 0.353, 0.345] },
  { name: "Viewing (3–7 s)", labels: ["FBCSP-all", "GA single-obj.", "GA-NSGA"], values: [0.435, 0.428, 0.410] },
], { x: MX, y: 1.4, w: 7.3, h: 4.6, barDir: "col", barGrouping: "clustered", chartColors: ["9FC4EE", "128C7E"],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt", dataLabelColor: "1F2937",
  valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1, valAxisLabelFontSize: 11, catAxisLabelFontSize: 13,
  valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", valAxisLabelColor: "5B6472", catAxisLabelColor: "1F2937",
  valGridLine: { color: "E4E3DF", size: 0.5 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
  showTitle: true, title: "Balanced accuracy, block protocol, n = 22", titleFontSize: 13, titleColor: "1F2937", titleFontFace: "+mn-lt" });
stat(s, 8.4, 1.5, 4.3, "0.435", "FBCSP-all on the viewing period: the pipeline works when there is signal", P.teal);
stat(s, 8.4, 3.0, 4.3, "−0.025", "GA-NSGA vs FBCSP-all on viewing (p = 0.004): cost pressure throws signal away", P.coral);
T(s, "The chooser rule matters here (0.377–0.421 across rules), unlike on imagery.", { x: 8.4, y: 4.6, w: 4.3, h: 0.9, fontSize: 14, color: P.ink, valign: "top" });
s.addNotes("A sanity check on the viewing period, when the shape is on screen. All methods decode it well, up to 0.435, so the pipeline works when the signal exists. But there, the GA is significantly worse than the baseline, by 0.025 with p 0.004: its cost pressure discards real information. Feature selection is a cost-reduction tool with a price.");

// ================================================================ 14. takeaways
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
s.addText("Key takeaways", { placeholder: "title" });
const tk = [
  ["Negative result", "The stability-aware GA does not beat FBCSP on all components (0.345 vs 0.353, p = 0.79), nor the other baselines."],
  ["Data, not model", "Every model, from 13 features to a 3,000-parameter CNN, memorises its training data and ends at about chance + 0.02."],
  ["Selection = cost reduction", "Same accuracy with 8 % of the features, but the selection is no better than random and not reproducible (Kuncheva 0.08)."],
  ["Rigour pays off", "Pre-registered, nested, leakage-controlled and reproducible (bit-identical across 3 machines, 200 tests, v1.0)."],
];
tk.forEach(([a, b], i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = MX + col * 6.15, y = 1.6 + row * 2.05;
  card(s, x, y, 5.95, 1.8, col === row ? P.tint : P.tealLight, "takeaway " + a);
  badge(s, x + 0.25, y + 0.25, i + 1, i === 0 ? P.coral : P.navy);
  T(s, a, { x: x + 0.9, y: y + 0.25, w: 4.85, h: 0.46, fontSize: 19, bold: true, color: P.navy, valign: "middle" });
  T(s, b, { x: x + 0.3, y: y + 0.85, w: 5.4, h: 1.05, fontSize: 15, color: P.ink, valign: "top" });
});
s.addNotes("Four takeaways. One, a clear negative result for the proposed method. Two, the bottleneck is the imagery signal, not the classifier. Three, the GA buys a cost reduction, not accuracy, and its selections are not reproducible. Four, a strict, pre-registered and reproducible protocol is what lets us trust this conclusion.");

// ================================================================ 15. limitations & future work
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
s.addText("Limitations and future work", { placeholder: "title" });
card(s, MX, 1.5, 5.95, 4.0, P.tint, "limitations");
T(s, "Limitations", { x: MX + 0.3, y: 1.65, w: 5.4, h: 0.45, fontSize: 19, bold: true, color: P.navy });
T(s, bullets([
  "3 classes (pentagram), not the proposal's triangle and hexagon",
  "Weak signal; with n = 22 only differences ≥ 0.024 are detectable",
  "Fixed search space and budget; 3 small inner folds as GA fitness",
  "Dataset reuses trial orders across sessions",
  "Could not reproduce the 75.1 % EEGNet result reported by Gao et al. (our EEGNet: 0.34, also with random splits)",
], 15), { x: MX + 0.3, y: 2.15, w: 5.4, h: 3.6, valign: "top" });
card(s, MX + 6.15, 1.5, 5.95, 4.0, P.tealLight, "future work");
T(s, "Future work", { x: MX + 6.45, y: 1.65, w: 5.4, h: 0.45, fontSize: 19, bold: true, color: P.teal });
T(s, bullets([
  "Four-shape data (circle, square, triangle, hexagon) with more trials per class",
  "Reconcile our pipeline with the published 75.1 %",
  "Fitness robust to the winner's curse: repeated inner CV, stability selection",
  "Use selection where signal is strong and channels are costly: perception, feedback-trained imagery, low-channel wearable BCIs",
], 15), { x: MX + 6.45, y: 2.15, w: 5.4, h: 3.6, valign: "top" });
s.addNotes("Limitations: three classes instead of four, a weak signal with 22 subjects, a fixed search space, and dataset trial-order structure. We also could not reproduce the 75.1 percent reported by the dataset authors; our EEGNet stays at 0.34 even with their random-split protocol, which is the first thing we would investigate next. Future work: four-shape data, more robust fitness, and applying selection where the signal is strong.");

// ================================================================ 16. close
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
T(s, "Thank you", { x: MX, y: 1.6, w: W - 2 * MX, h: 1.0, fontFace: "Cambria", fontSize: 48, bold: true, color: P.white });
T(s, "Questions?", { x: MX, y: 2.6, w: W - 2 * MX, h: 0.7, fontSize: 28, color: "9FE3D6" });
T(s, "“A stability-aware GA buys compactness, not accuracy: on imagined shapes every model ends at about chance + 0.02, and when the signal is real, compactness has a price.”",
  { x: MX, y: 3.7, w: 10.5, h: 1.2, fontSize: 18, italic: true, color: "C9D3E6", valign: "top" });
team.forEach(([n, r], i) => T(s, `${n} (${r})`, { x: MX + i * 4.1, y: 5.6, w: 3.95, h: 0.4, fontSize: 13, color: P.white }));
T(s, "Code, data pipeline, results and report: release v1.0 (pre-registered, 200 tests, bit-identical reruns)", { x: MX, y: H - 0.75, w: 10, h: 0.35, fontSize: 12, color: "AAB6CC" });
s.addNotes("Thank you. In one sentence: the stability-aware GA buys compactness, not accuracy. We are happy to take questions.");

// ---------------------------------------------------------------- write
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
