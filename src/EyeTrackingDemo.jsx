import React, { useEffect, useRef, useState, useCallback } from "react";

// ─── LANDMARK INDICES ─────────────────────────────────────────────────────────
const LM = {
  LEFT_IRIS:  [468, 469, 470, 471, 472],
  RIGHT_IRIS: [473, 474, 475, 476, 477],
  L_INNER: 133, L_OUTER: 33,
  R_INNER: 362, R_OUTER: 263,
  // Two points per lid edge → averaged for stability
  L_LID_TOP: 159, L_LID_TOP2: 158,
  L_LID_BOT: 145, L_LID_BOT2: 153,
  R_LID_TOP: 386, R_LID_TOP2: 385,
  R_LID_BOT: 374, R_LID_BOT2: 380,
  // EAR helpers (kept exactly as before)
  L_TOP: [386, 374], L_BOT: [380, 373],
  R_TOP: [159, 158], R_BOT: [153, 145],
  NOSE_TIP: 1, CHIN: 152,
  L_CHEEK: 234, R_CHEEK: 454,
  FOREHEAD: 10,
};

// ─── 16-POINT CALIBRATION GRID ───────────────────────────────────────────────
const TARGETS = [
  { x:0.05, y:0.05 }, { x:0.35, y:0.05 }, { x:0.65, y:0.05 }, { x:0.95, y:0.05 },
  { x:0.05, y:0.35 }, { x:0.35, y:0.35 }, { x:0.65, y:0.35 }, { x:0.95, y:0.35 },
  { x:0.05, y:0.65 }, { x:0.35, y:0.65 }, { x:0.65, y:0.65 }, { x:0.95, y:0.65 },
  { x:0.05, y:0.95 }, { x:0.35, y:0.95 }, { x:0.65, y:0.95 }, { x:0.95, y:0.95 },
];

const COLLECT_MS   = 2000;
const COUNTDOWN_MS = 600;

// ─── MATH ─────────────────────────────────────────────────────────────────────
function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function irisCenter(lm, idxs) {
  let x = 0, y = 0;
  for (const i of idxs) { x += lm[i].x; y += lm[i].y; }
  return { x: x / idxs.length, y: y / idxs.length };
}

// ─── FEATURE EXTRACTION (vertical accuracy fixes here) ────────────────────────
function extractFeatures(lm) {
  const li = irisCenter(lm, LM.LEFT_IRIS);
  const ri = irisCenter(lm, LM.RIGHT_IRIS);

  const lIn = lm[LM.L_INNER], lOut = lm[LM.L_OUTER];
  const rIn = lm[LM.R_INNER], rOut = lm[LM.R_OUTER];
  const lw  = dist(lIn, lOut) || 0.001;
  const rw  = dist(rIn, rOut) || 0.001;
  const lCx = (lIn.x + lOut.x) / 2;
  const rCx = (rIn.x + rOut.x) / 2;

  // ── FIX 1: Average two lid points per edge for smoother vertical reference ──
  const lTopY = (lm[LM.L_LID_TOP].y + lm[LM.L_LID_TOP2].y) / 2;
  const lBotY = (lm[LM.L_LID_BOT].y + lm[LM.L_LID_BOT2].y) / 2;
  const rTopY = (lm[LM.R_LID_TOP].y + lm[LM.R_LID_TOP2].y) / 2;
  const rBotY = (lm[LM.R_LID_BOT].y + lm[LM.R_LID_BOT2].y) / 2;
  const lMidY = (lTopY + lBotY) / 2;
  const rMidY = (rTopY + rBotY) / 2;

  // ── Horizontal offset: normalized by eye width (was correct before) ─────────
  const lOffX = (li.x - lCx) / lw;
  const rOffX = (ri.x - rCx) / rw;

  // ── FIX 2: Vertical offset normalized by eye WIDTH not eye HEIGHT ────────────
  // Old: (li.y - lMidY) / lh  — bad because lh shrinks when you look down
  // New: (li.y - lMidY) / lw  — lw is stable regardless of gaze direction
  const lOffY = (li.y - lMidY) / lw;
  const rOffY = (ri.y - rMidY) / rw;

  // ── FIX 3: Absolute vertical anchor — iris Y relative to nose tip ────────────
  // Nose tip is a stable face landmark unaffected by eyelid position.
  // This gives the regression a non-lid-dependent vertical signal that
  // extrapolates naturally beyond the calibration target range (e.g. looking
  // above the top of the screen).
  const nose = lm[LM.NOSE_TIP];
  const lc   = lm[LM.L_CHEEK], rc = lm[LM.R_CHEEK];
  const fh_  = lm[LM.FOREHEAD], chin = lm[LM.CHIN];
  const fw   = dist(lc, rc) || 0.001;
  const fcx  = (lc.x + rc.x) / 2;
  const fvy  = (fh_.y + chin.y) / 2;
  const fvh  = dist(fh_, chin) || 0.001;
  const yaw   = (nose.x - fcx) / fw;
  const pitch = (nose.y - fvy) / fvh;

  // Signed vertical distance from each iris to the nose tip, face-height normalised.
  // When looking up: iris moves up → this value becomes more negative → model can
  // extrapolate a prediction of y < 0.05 reliably.
  const lIrisNoseY = (li.y - nose.y) / fvh;
  const rIrisNoseY = (ri.y - nose.y) / fvh;

  const faceScale = dist(li, ri);
  const avgIrisY  = (li.y + ri.y) / 2;

  // 12 features (was 8). Polynomial expansion will be ~91 terms (manageable).
  return [
    lOffX, lOffY, rOffX, rOffY,    // iris-in-socket offsets (fixed normalization)
    lIrisNoseY, rIrisNoseY,          // absolute vertical anchor (new — key for up/down range)
    yaw, pitch,                       // head pose
    faceScale, avgIrisY,             // global scale / absolute position
    lw, rw,                           // eye width (scale reference, helps cross-terms)
  ];
}

function polyFeatures(f) {
  const n = f.length;
  const out = [1, ...f];
  for (let i = 0; i < n; i++) out.push(f[i] * f[i]);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      out.push(f[i] * f[j]);
  return out;
}

function ridgeRegression(X, y, lambda = 0.01) {
  const n = X[0].length;
  const m = X.length;
  const XtX = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < m; k++)
        XtX[i][j] += X[k][i] * X[k][j];
  for (let i = 1; i < n; i++) XtX[i][i] += lambda;
  const Xty = new Float64Array(n);
  for (let i = 0; i < n; i++)
    for (let k = 0; k < m; k++)
      Xty[i] += X[k][i] * y[k];
  const aug = XtX.map((row, i) => [...Array.from(row), Xty[i]]);
  for (let col = 0; col < n; col++) {
    let max = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(aug[r][col]) > Math.abs(aug[max][col])) max = r;
    [aug[col], aug[max]] = [aug[max], aug[col]];
    const piv = aug[col][col];
    if (Math.abs(piv) < 1e-14) continue;
    for (let r = col + 1; r < n; r++) {
      const f2 = aug[r][col] / piv;
      for (let c = col; c <= n; c++) aug[r][c] -= f2 * aug[col][c];
    }
  }
  const b = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    b[i] = aug[i][n];
    for (let j = i + 1; j < n; j++) b[i] -= aug[i][j] * b[j];
    b[i] /= aug[i][i] || 1;
  }
  return b;
}

// ─── FIX 4: Higher lambda for Y regression ────────────────────────────────────
// Vertical needs stronger regularisation so it generalises beyond the
// y=0.05..0.95 calibration range (i.e. looking above/below the screen).
// Horizontal is fine with tighter fit (lambda=0.01).
function trainModel(samples) {
  const X  = samples.map(s => polyFeatures(s.features));
  const bx = ridgeRegression(X, samples.map(s => s.sx), 0.01);
  const by = ridgeRegression(X, samples.map(s => s.sy), 0.05);
  return { bx, by };
}

function predict(model, features) {
  const v  = polyFeatures(features);
  const sx = Array.from(model.bx).reduce((s, b, i) => s + b * v[i], 0);
  const sy = Array.from(model.by).reduce((s, b, i) => s + b * v[i], 0);
  return {
    x: Math.max(0.01, Math.min(0.99, sx)),
    // ── FIX 5: Allow slight over/under-shoot so extreme gaze hits the edge ──
    // Display code clamps to screen bounds separately.
    y: Math.max(-0.05, Math.min(1.05, sy)),
  };
}

// ─── VELOCITY-ADAPTIVE SMOOTHER ───────────────────────────────────────────────
class AdaptiveSmoother {
  constructor() {
    this.x = null; this.y = null;
    this.px = null; this.py = null;
  }
  push(x, y) {
    if (this.x === null) { this.x = x; this.y = y; this.px = x; this.py = y; return; }
    const vel   = Math.sqrt((x - this.x) ** 2 + (y - this.y) ** 2);
    const alpha = Math.min(0.95, Math.max(0.12, vel * 14));
    this.px = this.x; this.py = this.y;
    this.x  = alpha * x + (1 - alpha) * this.x;
    this.y  = alpha * y + (1 - alpha) * this.y;
  }
  get() {
    if (this.x === null) return null;
    return { x: this.x, y: this.y };
  }
  reset() { this.x = null; this.y = null; this.px = null; this.py = null; }
}

// ─── EAR ─────────────────────────────────────────────────────────────────────
function calcEAR(lm) {
  const e = (ti, bi, li, ri) => {
    const v1 = dist(lm[ti[0]], lm[bi[0]]);
    const v2 = dist(lm[ti[1]], lm[bi[1]]);
    const h  = dist(lm[li],    lm[ri]);
    return h < 0.0001 ? 0.3 : (v1 + v2) / (2 * h);
  };
  return (e(LM.L_TOP, LM.L_BOT, LM.L_INNER, LM.L_OUTER) +
          e(LM.R_TOP, LM.R_BOT, LM.R_INNER, LM.R_OUTER)) / 2;
}

// ─── ZONE ────────────────────────────────────────────────────────────────────
function getZone(x, y) {
  const col = x < 0.28 ? "LEFT" : x > 0.72 ? "RIGHT" : "CENTER";
  const row = y < 0.30 ? "TOP"  : y > 0.70 ? "BOTTOM" : "MID";
  if (col === "CENTER" && row === "MID") return "CENTER";
  if (row === "MID") return col;
  if (col === "CENTER") return row === "TOP" ? "UP" : "DOWN";
  return `${row === "TOP" ? "UP-" : "DOWN-"}${col}`;
}

const ZONE_COLOR = {
  CENTER:"#22c55e", LEFT:"#3b82f6", RIGHT:"#f59e0b",
  UP:"#8b5cf6", DOWN:"#ec4899",
  "UP-LEFT":"#06b6d4","UP-RIGHT":"#a855f7",
  "DOWN-LEFT":"#6366f1","DOWN-RIGHT":"#f97316",
  "OUT_OF_FRAME":"#ef4444",
};

// ─── CHUNK ───────────────────────────────────────────────────────────────────
function makeChunk() {
  return { start: Date.now(), zones: {}, trail: [], blinks: 0, earSum: 0, earN: 0 };
}
function finalizeChunk(c) {
  const total  = Object.values(c.zones).reduce((a, b) => a + b, 0) || 1;
  const dom    = Object.entries(c.zones).sort((a, b) => b[1]-a[1])[0]?.[0] || "CENTER";
  const dur    = Date.now() - c.start;
  const br     = dur > 0 ? +(c.blinks / (dur / 1000)).toFixed(3) : 0;
  const avgEAR = c.earN > 0 ? +(c.earSum / c.earN).toFixed(3) : 0.3;
  const rF  = ((c.zones["RIGHT"]||0)+(c.zones["UP-RIGHT"]||0)+(c.zones["DOWN-RIGHT"]||0)) / total;
  const lF  = ((c.zones["LEFT"] ||0)+(c.zones["UP-LEFT"] ||0)+(c.zones["DOWN-LEFT"] ||0)) / total;
  const dF  = ((c.zones["DOWN"] ||0)+(c.zones["DOWN-LEFT"]||0)+(c.zones["DOWN-RIGHT"]||0)) / total;
  const offF = (c.zones["OUT_OF_FRAME"]||0) / total;
  let susp = 0;
  if (dom !== "CENTER" && dom !== "OUT_OF_FRAME") susp += 30;
  if (rF > 0.35)  susp += 20;
  if (lF > 0.35)  susp += 20;
  if (dF > 0.30)  susp += 15;
  if (offF > 0.1) susp += 25;
  if (br < 0.05 && dur > 3000) susp += 15;
  return {
    ts: Date.now(), timeLabel: new Date().toLocaleTimeString(),
    duration: dur, zone: dom,
    zonePercent: Object.fromEntries(Object.entries(c.zones).map(([z,n]) => [z,+((n/total)*100).toFixed(1)])),
    gazeCenter: c.trail.length ? {
      x: +(c.trail.reduce((s,p) => s+p.x,0)/c.trail.length).toFixed(3),
      y: +(c.trail.reduce((s,p) => s+p.y,0)/c.trail.length).toFixed(3),
    } : null,
    blinks: c.blinks, blinkRate: br, avgEAR,
    suspicion: Math.min(Math.round(susp), 100),
  };
}

// ─── SUMMARY ─────────────────────────────────────────────────────────────────
function buildSummary(chunks) {
  if (!chunks.length) return null;
  const N = chunks.length;
  const zc = {};
  chunks.forEach(c => { zc[c.zone] = (zc[c.zone]||0)+1; });
  const dom      = Object.entries(zc).sort((a,b)=>b[1]-a[1])[0]?.[0]||"CENTER";
  const focusPct = +((( zc["CENTER"]||0)/N)*100).toFixed(1);
  const rightPct = +(((zc["RIGHT"]||0)+(zc["UP-RIGHT"]||0)+(zc["DOWN-RIGHT"]||0))/N*100).toFixed(1);
  const leftPct  = +(((zc["LEFT"] ||0)+(zc["UP-LEFT"] ||0)+(zc["DOWN-LEFT"] ||0))/N*100).toFixed(1);
  const downPct  = +(((zc["DOWN"] ||0)+(zc["DOWN-LEFT"]||0)+(zc["DOWN-RIGHT"]||0))/N*100).toFixed(1);
  const offPct   = +(((zc["OUT_OF_FRAME"]||0)/N)*100).toFixed(1);
  const avgSusp  = +(chunks.reduce((s,c)=>s+c.suspicion,0)/N).toFixed(1);
  const maxSusp  = Math.max(...chunks.map(c=>c.suspicion));
  const highRisk = chunks.filter(c=>c.suspicion>=60).length;
  const totalBlinks = chunks.reduce((s,c)=>s+c.blinks,0);
  const totalSec    = chunks.reduce((s,c)=>s+c.duration,0)/1000;
  const blinkRate   = totalSec>0 ? +(totalBlinks/totalSec).toFixed(3) : 0;
  let verdict="CLEAN", verdictColor="#22c55e";
  if (avgSusp>=60||highRisk>=3) { verdict="HIGH RISK"; verdictColor="#ef4444"; }
  else if (avgSusp>=35||highRisk>=1) { verdict="REVIEW"; verdictColor="#f59e0b"; }
  return { N, dom, focusPct, rightPct, leftPct, downPct, offPct, avgSusp, maxSusp, highRisk, totalBlinks, blinkRate, verdict, verdictColor, zc, totalSec:+totalSec.toFixed(0) };
}

// ─── COMPONENT ───────────────────────────────────────────────────────────────
export default function EyeTracker() {
  const videoRef        = useRef(null);
  const canvasRef       = useRef(null);
  const heatCanvasRef   = useRef(null);
  const heatDisplayRef  = useRef(null);
  const fmRef           = useRef(null);
  const camRef          = useRef(null);
  const smootherRef     = useRef(new AdaptiveSmoother());
  const modelRef        = useRef(null);
  const chunkRef        = useRef(makeChunk());
  const timerRef        = useRef(null);
  const blinkRef        = useRef({ consec:0, was:false });
  const heatAnimRef     = useRef(null);
  const heatPointsRef   = useRef([]);
  const sessionRef      = useRef(null);
  const calibSamplesRef = useRef([]);
  const calibBufRef     = useRef([]);
  const calibTimerRef   = useRef(null);
  const HEAT_RADIUS     = 45;

  const [mode,        setMode]        = useState("idle");
  const [calibIdx,    setCalibIdx]    = useState(0);
  const [progress,    setProgress]    = useState(0);
  const [countdown,   setCountdown]   = useState(false);
  const [mpReady,     setMpReady]     = useState(false);
  const [face,        setFace]        = useState(false);
  const [gaze,        setGaze]        = useState(null);
  const [zone,        setZone]        = useState("—");
  const [chunks,      setChunks]      = useState([]);
  const [err,         setErr]         = useState("");
  const [showHeat,    setShowHeat]    = useState(false);
  const [heatOpacity, setHeatOpacity] = useState(0.75);
  const [calibScore,  setCalibScore]  = useState(null);

  const modeRef     = useRef("idle");
  const calibIdxRef = useRef(0);
  modeRef.current     = mode;
  calibIdxRef.current = calibIdx;

  // ── Load MediaPipe ───────────────────────────────────────────────────────
  useEffect(() => {
    const scripts = [
      { id:"mp-fm", src:"https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js" },
      { id:"mp-cu", src:"https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" },
    ];
    let done = 0;
    scripts.forEach(({ id, src }) => {
      if (document.getElementById(id)) { if (++done === 2) setMpReady(true); return; }
      const s = document.createElement("script");
      s.id = id; s.src = src; s.crossOrigin = "anonymous";
      s.onload  = () => { if (++done === 2) setMpReady(true); };
      s.onerror = () => setErr("MediaPipe failed to load.");
      document.head.appendChild(s);
    });
  }, []);

  // ── Calibration step ────────────────────────────────────────────────────
  const stepCalib = useCallback((idx) => {
    calibBufRef.current = [];
    setCalibIdx(idx);
    setProgress(0);
    setCountdown(true);

    setTimeout(() => {
      setCountdown(false);
      setMode("collecting");
      const t0 = Date.now();

      const iv = setInterval(() => {
        const elapsed = Date.now() - t0;
        setProgress(Math.min((elapsed / COLLECT_MS) * 100, 100));

        if (elapsed >= COLLECT_MS) {
          clearInterval(iv);
          const buf = calibBufRef.current.filter(Boolean);
          if (buf.length > 5) {
            const tgt = TARGETS[idx];
            buf.forEach(f => calibSamplesRef.current.push({ features: f, sx: tgt.x, sy: tgt.y }));
          }
          const next = idx + 1;
          if (next >= TARGETS.length) {
            if (calibSamplesRef.current.length > 20) {
              const model = trainModel(calibSamplesRef.current);
              modelRef.current = model;
              smootherRef.current.reset();

              const pointErrors = TARGETS.map((tgt) => {
                const ptSamples = calibSamplesRef.current.filter(
                  s => Math.abs(s.sx - tgt.x) < 0.01 && Math.abs(s.sy - tgt.y) < 0.01
                );
                if (!ptSamples.length) return 0;
                const avgF = ptSamples[Math.floor(ptSamples.length / 2)];
                const pred = predict(model, avgF.features);
                return Math.sqrt((pred.x - tgt.x) ** 2 + (pred.y - tgt.y) ** 2);
              });
              const rmse    = pointErrors.reduce((a,b)=>a+b,0) / pointErrors.length;
              const quality = rmse < 0.05 ? "Excellent" : rmse < 0.10 ? "Good" : rmse < 0.18 ? "Fair — consider recalibrating" : "Poor — please recalibrate";
              setCalibScore({ rmse: +rmse.toFixed(3), quality });

              setMode("tracking");
              setCalibIdx(0); setProgress(0);
            } else {
              setErr("Not enough data — try again with better lighting.");
              setMode("idle");
            }
          } else {
            setMode("countdown");
            stepCalib(next);
          }
        }
      }, 50);
      calibTimerRef.current = iv;
    }, COUNTDOWN_MS);
  }, []);

  // ── onResults (30fps) ───────────────────────────────────────────────────
  const onResults = useCallback((results) => {
    const cv  = canvasRef.current;
    const vid = videoRef.current;
    if (cv && vid) {
      const ctx = cv.getContext("2d");
      ctx.save(); ctx.scale(-1, 1);
      ctx.drawImage(vid, -cv.width, 0, cv.width, cv.height);
      ctx.restore();
    }

    if (!results.multiFaceLandmarks?.length) {
      setFace(false);
      smootherRef.current.reset();
      if (modeRef.current === "tracking") {
        setZone("OUT_OF_FRAME");
        chunkRef.current.zones["OUT_OF_FRAME"] =
          (chunkRef.current.zones["OUT_OF_FRAME"] || 0) + 1;
      }
      return;
    }
    setFace(true);
    const lm  = results.multiFaceLandmarks[0];
    const ear = calcEAR(lm);

    if (cv) {
      const ctx = cv.getContext("2d");
      [LM.LEFT_IRIS, LM.RIGHT_IRIS].forEach(idxs => {
        const c  = irisCenter(lm, idxs);
        const px = (1 - c.x) * cv.width;
        const py = c.y * cv.height;
        ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#00ff88"; ctx.fill();
        ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.strokeStyle = "#00ff88"; ctx.lineWidth = 1.5; ctx.stroke();
      });
    }

    const features = extractFeatures(lm);

    if (modeRef.current === "collecting" && ear > 0.22) {
      calibBufRef.current.push(features);
    }

    if (modeRef.current === "tracking" && modelRef.current) {
      const raw = predict(modelRef.current, features);
      if (ear > 0.19) smootherRef.current.push(raw.x, raw.y);

      const smooth = smootherRef.current.get();
      if (!smooth) return;

      // Clamp for display only — allow the smoother to track near-edge values
      const displayX = Math.max(0.01, Math.min(0.99, smooth.x));
      const displayY = Math.max(0.01, Math.min(0.99, smooth.y));

      const z = getZone(displayX, displayY);

      const b = blinkRef.current;
      if (ear < 0.21) {
        b.consec++;
      } else {
        if (b.consec >= 2 && b.was) chunkRef.current.blinks++;
        b.was    = b.consec >= 2;
        b.consec = 0;
      }
      chunkRef.current.earSum += ear;
      chunkRef.current.earN   += 1;
      chunkRef.current.zones[z] = (chunkRef.current.zones[z]||0) + 1;
      if (chunkRef.current.earN % 8 === 0) {
        chunkRef.current.trail.push({ x:+displayX.toFixed(3), y:+displayY.toFixed(3) });
      }
      heatPointsRef.current.push({ x: displayX, y: displayY });
      setGaze({ x: displayX, y: displayY });
      setZone(z);
    }
  }, []);

  // ── Start / Recalibrate ──────────────────────────────────────────────────
  const startCalibration = useCallback(async () => {
    if (!videoRef.current) { setErr("Video not ready."); return; }
    if (!mpReady || !window.FaceMesh || !window.Camera) { setErr("MediaPipe not ready."); return; }

    setErr("");
    calibSamplesRef.current = [];
    calibBufRef.current     = [];
    modelRef.current        = null;
    smootherRef.current.reset();
    blinkRef.current        = { consec:0, was:false };
    heatPointsRef.current   = [];
    setChunks([]); setGaze(null); setZone("—");

    if (camRef.current) { camRef.current.stop(); camRef.current = null; }

    if (!fmRef.current) {
      fmRef.current = new window.FaceMesh({
        locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
      });
      fmRef.current.setOptions({
        maxNumFaces: 1, refineLandmarks: true,
        minDetectionConfidence: 0.75, minTrackingConfidence: 0.75,
      });
    }
    fmRef.current.onResults(onResults);

    camRef.current = new window.Camera(videoRef.current, {
      onFrame: async () => {
        if (fmRef.current) await fmRef.current.send({ image: videoRef.current });
      },
      width: 640, height: 480,
    });
    await camRef.current.start();

    setMode("countdown");
    stepCalib(0);
  }, [mpReady, onResults, stepCalib]);

  // ── Chunk timer ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (mode === "tracking") {
      chunkRef.current = makeChunk();
      timerRef.current = setInterval(() => {
        setChunks(prev => [...prev, finalizeChunk(chunkRef.current)]);
        chunkRef.current = makeChunk();
      }, 5000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [mode]);

  // ── Stop session ─────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    clearInterval(timerRef.current);
    clearInterval(calibTimerRef.current);
    if (camRef.current) { camRef.current.stop(); camRef.current = null; }
    modelRef.current = null;

    const savedHeatPoints = [...heatPointsRef.current];
    setChunks(prev => {
      const final = finalizeChunk(chunkRef.current);
      const all   = final.earN > 0 ? [...prev, final] : prev;
      sessionRef.current = { chunks: all, heatPoints: savedHeatPoints };
      return all;
    });
    setMode("results");
    setFace(false); setGaze(null); setZone("—");
  }, []);

  // ── Cleanup on unmount ───────────────────────────────────────────────────
  useEffect(() => () => {
    clearInterval(timerRef.current);
    clearInterval(calibTimerRef.current);
    cancelAnimationFrame(heatAnimRef.current);
    if (camRef.current) camRef.current.stop();
  }, []);

  // ── Heatmap render loop ───────────────────────────────────────────────────
  useEffect(() => {
    if (!showHeat || mode !== "tracking") {
      cancelAnimationFrame(heatAnimRef.current);
      return;
    }
    const STOPS = [
      [0,    0,   0, 255,   0],
      [0.25, 0, 255, 255, 120],
      [0.5,  0, 255,   0, 180],
      [0.75,255, 255,   0, 210],
      [1.0, 255,   0,   0, 255],
    ];
    function lerpColor(t) {
      t = Math.max(0, Math.min(1, t));
      for (let i = 1; i < STOPS.length; i++) {
        const [t0,r0,g0,b0,a0] = STOPS[i-1];
        const [t1,r1,g1,b1,a1] = STOPS[i];
        if (t <= t1) {
          const f = (t-t0)/(t1-t0);
          return [r0+f*(r1-r0), g0+f*(g1-g0), b0+f*(b1-b0), a0+f*(a1-a0)].map(Math.round);
        }
      }
      return [255,0,0,255];
    }
    function render() {
      const dc = heatDisplayRef.current;
      const oc = heatCanvasRef.current;
      if (!dc || !oc) return;
      const W = window.innerWidth, H = window.innerHeight;
      if (dc.width!==W||dc.height!==H) { dc.width=W; dc.height=H; }
      if (oc.width!==W||oc.height!==H) { oc.width=W; oc.height=H; }
      const dctx = dc.getContext("2d");
      const octx = oc.getContext("2d");
      octx.clearRect(0,0,W,H);
      const pts = heatPointsRef.current;
      if (!pts.length) { dctx.clearRect(0,0,W,H); heatAnimRef.current = requestAnimationFrame(render); return; }
      pts.forEach(pt => {
        const cx = pt.x*W, cy = pt.y*H;
        const g  = octx.createRadialGradient(cx,cy,0,cx,cy,HEAT_RADIUS);
        g.addColorStop(0,   "rgba(255,255,255,0.2)");
        g.addColorStop(0.5, "rgba(255,255,255,0.06)");
        g.addColorStop(1,   "rgba(255,255,255,0)");
        octx.fillStyle = g;
        octx.fillRect(cx-HEAT_RADIUS, cy-HEAT_RADIUS, HEAT_RADIUS*2, HEAT_RADIUS*2);
      });
      const id  = octx.getImageData(0,0,W,H);
      const out = dctx.createImageData(W,H);
      const src = id.data, dst = out.data;
      let maxA = 0;
      for (let i=3; i<src.length; i+=4) if (src[i]>maxA) maxA=src[i];
      if (!maxA) maxA = 1;
      for (let i=0; i<src.length; i+=4) {
        const intensity = src[i+3]/maxA;
        if (intensity < 0.01) { dst[i]=dst[i+1]=dst[i+2]=dst[i+3]=0; continue; }
        const [r,g,b,a] = lerpColor(intensity);
        dst[i]=r; dst[i+1]=g; dst[i+2]=b; dst[i+3]=Math.round(a*intensity);
      }
      dctx.clearRect(0,0,W,H);
      dctx.putImageData(out,0,0);
      heatAnimRef.current = requestAnimationFrame(render);
    }
    heatAnimRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(heatAnimRef.current);
  }, [showHeat, mode]);

  const color   = ZONE_COLOR[zone] || "#22c55e";
  const isCalib = mode === "countdown" || mode === "collecting";

  return (
    <>
      <video ref={videoRef} autoPlay muted playsInline
        style={{ position:"absolute", width:1, height:1, opacity:0, pointerEvents:"none" }} />
      <canvas ref={heatCanvasRef} style={{ display:"none" }} />

      <div style={{ width:"100vw", height:"100vh", background:"#fff", position:"relative", overflow:"hidden", fontFamily:"monospace" }}>

        {/* Heatmap overlay */}
        <canvas ref={heatDisplayRef} style={{
          position:"fixed", inset:0, pointerEvents:"none", zIndex:90,
          opacity: showHeat ? heatOpacity : 0, transition:"opacity 0.3s",
        }} />

        {/* ── CALIBRATION SCREEN ────────────────────────────────────────── */}
        {isCalib && (
          <div style={{ position:"fixed", inset:0, background:"#000", zIndex:999 }}>
            {TARGETS.map((t, i) => {
              const done = i < calibIdx, current = i === calibIdx;
              return (
                <div key={i} style={{
                  position:"absolute",
                  left:`${t.x*100}%`, top:`${t.y*100}%`,
                  transform:"translate(-50%,-50%)",
                  width: current ? 28 : 12, height: current ? 28 : 12,
                  borderRadius:"50%",
                  background: done ? "#22c55e" : current ? "#fff" : "rgba(255,255,255,0.12)",
                  transition:"all 0.25s",
                  boxShadow: current
                    ? `0 0 0 ${3+(progress/100)*10}px rgba(255,255,255,${0.1+(progress/100)*0.3}), 0 0 24px rgba(255,255,255,0.5)`
                    : "none",
                }} />
              );
            })}
            <div style={{ position:"absolute", bottom:"5%", left:"50%", transform:"translateX(-50%)", textAlign:"center", color:"#fff" }}>
              {countdown ? (
                <>
                  <div style={{ fontSize:22, fontWeight:700, marginBottom:8 }}>{calibIdx+1} / {TARGETS.length}</div>
                  <div style={{ fontSize:14, color:"#9ca3af" }}>
                    Look at the <b style={{color:"#fff"}}>white dot</b> — keep your <b style={{color:"#fff"}}>head still</b>
                  </div>
                </>
              ) : (
                <>
                  <div style={{ fontSize:22, fontWeight:700, marginBottom:8 }}>
                    {calibIdx+1} / {TARGETS.length} — Hold...
                  </div>
                  <div style={{ width:220, height:6, background:"rgba(255,255,255,0.1)", borderRadius:3, margin:"0 auto 6px" }}>
                    <div style={{ width:`${progress}%`, height:"100%", background:"#22c55e", borderRadius:3, transition:"width 0.05s" }} />
                  </div>
                </>
              )}
              {!face && <div style={{ marginTop:10, fontSize:12, color:"#ef4444" }}>⚠ No face — check camera</div>}
            </div>
          </div>
        )}

        {/* ── GAZE DOT ──────────────────────────────────────────────────── */}
        {mode === "tracking" && gaze && (
          <div style={{
            position:"fixed",
            left:`${gaze.x*100}%`, top:`${gaze.y*100}%`,
            transform:"translate(-50%,-50%)",
            pointerEvents:"none", zIndex:100,
          }}>
            <div style={{
              position:"absolute", width:50, height:50, borderRadius:"50%",
              border:`2px solid ${color}`, opacity:face ? 0.3 : 0,
              top:"50%", left:"50%", transform:"translate(-50%,-50%)",
              animation: face ? "gp 1.2s ease-out infinite" : "none",
            }} />
            <div style={{
              position:"absolute", width:26, height:26, borderRadius:"50%",
              border:`2px solid ${color}`, opacity: face ? 0.7 : 0.4,
              top:"50%", left:"50%", transform:"translate(-50%,-50%)",
            }} />
            <div style={{
              width:12, height:12, borderRadius:"50%",
              background: color, boxShadow:`0 0 10px ${color}`,
              opacity: face ? 1 : 0.5,
            }} />
            <div style={{
              position:"absolute", top:"120%", left:"50%", transform:"translateX(-50%)",
              fontSize:11, fontWeight:700, color, whiteSpace:"nowrap",
              background:"rgba(255,255,255,0.92)", padding:"2px 6px", borderRadius:4,
            }}>
              {zone}
            </div>
          </div>
        )}

        {mode === "idle" && (
          <div style={{ position:"fixed", top:"50%", left:"50%", transform:"translate(-50%,-50%)", textAlign:"center", pointerEvents:"none" }}>
            <div style={{ fontSize:16, color:"#9ca3af", marginBottom:6 }}>Press Calibrate — takes ~35 seconds</div>
            <div style={{ fontSize:12, color:"#d1d5db" }}>16-point calibration · works for any screen size</div>
          </div>
        )}
        {mode === "tracking" && !face && (
          <div style={{
            position:"fixed", top:20, left:"50%", transform:"translateX(-50%)",
            background:"rgba(239,68,68,0.95)", color:"#fff",
            fontSize:13, fontWeight:700, letterSpacing:1,
            padding:"8px 20px", borderRadius:8, zIndex:200, pointerEvents:"none",
          }}>
            ⚠ OUT OF FRAME — face not visible
          </div>
        )}

        {/* ── CONTROLS ──────────────────────────────────────────────────── */}
        <div style={{ position:"fixed", top:12, right:12, zIndex:200, display:"flex", flexDirection:"column", gap:8, alignItems:"flex-end" }}>
          <div style={{ display:"flex", gap:8 }}>
            {mode === "idle" ? (
              <Btn color="#22c55e" disabled={!mpReady} onClick={startCalibration}>
                {mpReady ? "▶ Calibrate & Start" : "Loading..."}
              </Btn>
            ) : mode === "tracking" ? (
              <>
                <Btn color="#f59e0b" onClick={startCalibration}>Recalibrate</Btn>
                <Btn color={showHeat ? "#ec4899" : "#6b7280"} onClick={() => setShowHeat(h => !h)}>
                  {showHeat ? "Heatmap ON" : "Heatmap"}
                </Btn>
                <Btn color="#ef4444" onClick={stop}>■ Stop</Btn>
              </>
            ) : isCalib ? (
              <Btn color="#ef4444" onClick={() => {
                clearInterval(calibTimerRef.current);
                if (camRef.current) { camRef.current.stop(); camRef.current = null; }
                setMode("idle");
              }}>Cancel</Btn>
            ) : null}
          </div>

          {/* Webcam preview */}
          {(mode === "tracking" || isCalib) && (
            <div style={{ borderRadius:8, overflow:"hidden", border:`2px solid ${face ? "#22c55e" : "#ef4444"}`, width:160 }}>
              <canvas ref={canvasRef} width={320} height={240} style={{ display:"block", width:"100%" }} />
              <div style={{ background: face ? "rgba(34,197,94,0.85)" : "rgba(239,68,68,0.85)", color:"#fff", fontSize:10, fontWeight:700, textAlign:"center", padding:"2px 0" }}>
                {face ? "● FACE DETECTED" : "○ NO FACE"}
              </div>
            </div>
          )}

          {/* Live readout */}
          {mode === "tracking" && face && gaze && (
            <div style={{ background:"rgba(0,0,0,0.05)", borderRadius:6, padding:"8px 10px", fontSize:11, color:"#374151", lineHeight:1.9, minWidth:150 }}>
              <Row label="zone"   val={zone}                              color={color} />
              <Row label="x"      val={`${(gaze.x*100).toFixed(1)}%`} />
              <Row label="y"      val={`${(gaze.y*100).toFixed(1)}%`} />
              <Row label="chunks" val={chunks.length} />
              {calibScore && (
                <div style={{ marginTop:4, paddingTop:4, borderTop:"1px solid rgba(0,0,0,0.06)", fontSize:10 }}>
                  <span style={{ color:"#9ca3af" }}>calib </span>
                  <b style={{ color: calibScore.rmse < 0.05 ? "#22c55e" : calibScore.rmse < 0.10 ? "#f59e0b" : "#ef4444" }}>
                    {calibScore.quality.split(" ")[0]}
                  </b>
                  <span style={{ color:"#9ca3af" }}> (rmse {calibScore.rmse})</span>
                </div>
              )}
            </div>
          )}

          {/* Heatmap controls */}
          {mode === "tracking" && showHeat && (
            <div style={{ background:"rgba(0,0,0,0.05)", borderRadius:6, padding:"8px 10px", fontSize:10, color:"#6b7280", width:160 }}>
              <div style={{ fontWeight:700, color:"#ec4899", marginBottom:5 }}>Heatmap</div>
              <div style={{ display:"flex", justifyContent:"space-between" }}>
                <span>Opacity</span>
                <span style={{ color:"#374151" }}>{Math.round(heatOpacity*100)}%</span>
              </div>
              <input type="range" min={0.1} max={1} step={0.05} value={heatOpacity}
                onChange={e => setHeatOpacity(+e.target.value)}
                style={{ width:"100%", margin:"2px 0" }} />
              <div style={{ display:"flex", justifyContent:"space-between", marginTop:4 }}>
                <span style={{ color:"#9ca3af" }}>{heatPointsRef.current.length} pts</span>
                <button onClick={() => { heatPointsRef.current = []; }}
                  style={{ fontSize:9, padding:"2px 6px", borderRadius:4, background:"rgba(239,68,68,0.1)", color:"#ef4444", border:"1px solid rgba(239,68,68,0.3)", cursor:"pointer", fontFamily:"monospace" }}>
                  Clear
                </button>
              </div>
              <div style={{ display:"flex", height:7, borderRadius:4, overflow:"hidden", marginTop:8 }}>
                {["#00f","#0ff","#0f0","#ff0","#f00"].map((c,i) => (
                  <div key={i} style={{ flex:1, background:c }} />
                ))}
              </div>
              <div style={{ display:"flex", justifyContent:"space-between", fontSize:8, color:"#9ca3af", marginTop:2 }}>
                <span>low</span><span>high</span>
              </div>
            </div>
          )}
        </div>

        {/* ── CHUNK BAR ─────────────────────────────────────────────────── */}
        {chunks.length > 0 && mode === "tracking" && (
          <div style={{ position:"fixed", bottom:0, left:0, right:0, background:"rgba(255,255,255,0.96)", borderTop:"1px solid #e5e7eb", padding:"8px 16px", zIndex:200 }}>
            <div style={{ display:"flex", gap:2, alignItems:"flex-end", height:28, marginBottom:4 }}>
              {chunks.slice(-80).map((c,i) => (
                <div key={c.ts} title={`${c.timeLabel}·${c.zone}·${c.suspicion}/100`} style={{
                  flex:1, minWidth:4, borderRadius:"2px 2px 0 0",
                  height:`${4+(c.suspicion/100)*24}px`,
                  background: c.suspicion>=60?"#ef4444":c.suspicion>=30?"#f59e0b":"#22c55e",
                  opacity: 0.4+(i/80)*0.6,
                }} />
              ))}
            </div>
            <div style={{ display:"flex", gap:14, fontSize:10, color:"#9ca3af", overflowX:"auto" }}>
              {[...chunks].reverse().slice(0,5).map(c => (
                <div key={c.ts} style={{ display:"flex", gap:5, alignItems:"center", whiteSpace:"nowrap" }}>
                  <span>{c.timeLabel}</span>
                  <b style={{ color:ZONE_COLOR[c.zone]||"#22c55e" }}>{c.zone}</b>
                  {c.gazeCenter&&<span>({(c.gazeCenter.x*100).toFixed(0)}%,{(c.gazeCenter.y*100).toFixed(0)}%)</span>}
                  <b style={{ color:c.suspicion>=60?"#ef4444":c.suspicion>=30?"#f59e0b":"#22c55e" }}>{c.suspicion}/100</b>
                </div>
              ))}
            </div>
          </div>
        )}

        {err && (
          <div style={{ position:"fixed", top:12, left:"50%", transform:"translateX(-50%)", background:"#fef2f2", border:"1px solid #fecaca", color:"#dc2626", borderRadius:6, padding:"6px 14px", fontSize:12, zIndex:300 }}>
            {err}
          </div>
        )}

        <style>{`
          @keyframes gp {
            0%   { transform:translate(-50%,-50%) scale(1);   opacity:0.3; }
            100% { transform:translate(-50%,-50%) scale(2.4); opacity:0;   }
          }
          @keyframes slideUp {
            from { transform:translateY(100%); opacity:0; }
            to   { transform:translateY(0);    opacity:1; }
          }
        `}</style>
      </div>

      {/* ── RESULTS PANEL ─────────────────────────────────────────────────── */}
      {mode === "results" && (
        <ResultsPanel
          chunks={chunks}
          heatPoints={sessionRef.current?.heatPoints || []}
          onRetry={() => { setMode("idle"); setChunks([]); heatPointsRef.current=[]; }}
          onDownload={() => {
            const a = document.createElement("a");
            a.href = URL.createObjectURL(new Blob([JSON.stringify({ chunks, summary:buildSummary(chunks) },null,2)],{type:"application/json"}));
            a.download = `eye-tracking-${Date.now()}.json`;
            a.click();
          }}
        />
      )}
    </>
  );
}

// ─── RESULTS PANEL ────────────────────────────────────────────────────────────
function ResultsPanel({ chunks, heatPoints = [], onRetry, onDownload }) {
  const s          = buildSummary(chunks);
  const heatResRef = useRef(null);
  const heatOffRef = useRef(null);

  useEffect(() => {
    const dc = heatResRef.current;
    const oc = heatOffRef.current;
    if (!dc || !oc || !heatPoints.length) return;
    const W = dc.width, H = dc.height;
    oc.width = W; oc.height = H;
    const octx = oc.getContext("2d");
    const dctx = dc.getContext("2d");
    octx.clearRect(0,0,W,H); dctx.clearRect(0,0,W,H);
    const R = 30;
    heatPoints.forEach(pt => {
      const cx = pt.x * W, cy = pt.y * H;
      const g = octx.createRadialGradient(cx,cy,0,cx,cy,R);
      g.addColorStop(0,  "rgba(255,255,255,0.22)");
      g.addColorStop(0.5,"rgba(255,255,255,0.07)");
      g.addColorStop(1,  "rgba(255,255,255,0)");
      octx.fillStyle = g;
      octx.fillRect(cx-R, cy-R, R*2, R*2);
    });
    const STOPS = [[0,0,0,255,0],[0.25,0,255,255,120],[0.5,0,255,0,180],[0.75,255,255,0,210],[1,255,0,0,255]];
    function lerp(t) {
      t = Math.max(0,Math.min(1,t));
      for(let i=1;i<STOPS.length;i++){
        const [t0,r0,g0,b0,a0]=STOPS[i-1],[t1,r1,g1,b1,a1]=STOPS[i];
        if(t<=t1){const f=(t-t0)/(t1-t0);return[r0+f*(r1-r0),g0+f*(g1-g0),b0+f*(b1-b0),a0+f*(a1-a0)].map(Math.round);}
      }
      return[255,0,0,255];
    }
    const id=octx.getImageData(0,0,W,H), out=dctx.createImageData(W,H);
    const src=id.data,dst=out.data;
    let maxA=0; for(let i=3;i<src.length;i+=4) if(src[i]>maxA) maxA=src[i];
    if(!maxA) return;
    for(let i=0;i<src.length;i+=4){
      const intensity=src[i+3]/maxA;
      if(intensity<0.01){dst[i]=dst[i+1]=dst[i+2]=dst[i+3]=0;continue;}
      const [r,g,b,a]=lerp(intensity);
      dst[i]=r;dst[i+1]=g;dst[i+2]=b;dst[i+3]=Math.round(a*intensity);
    }
    dctx.putImageData(out,0,0);
  }, [heatPoints]);

  if(!s) return null;
  return (
    <div style={{ position:"fixed",inset:0,zIndex:1000,background:"#fff",fontFamily:"monospace",animation:"slideUp 0.35s ease-out",overflowY:"auto" }}>
      <div style={{ padding:"14px 20px",borderBottom:"1px solid #e5e7eb",display:"flex",alignItems:"center",justifyContent:"space-between",position:"sticky",top:0,background:"#fff",zIndex:10 }}>
        <div>
          <b style={{ fontSize:14, letterSpacing:2 }}>SESSION REPORT</b>
          <span style={{ fontSize:11,color:"#9ca3af",marginLeft:10 }}>{s.N} chunks · {s.totalSec}s</span>
        </div>
        <div style={{ display:"flex",gap:8 }}>
          <Btn color="#22c55e" onClick={onRetry}>New session</Btn>
          <Btn color="#6366f1" onClick={onDownload}>↓ JSON</Btn>
        </div>
      </div>

      <div style={{ padding:"18px 20px",maxWidth:1100,margin:"0 auto" }}>
        <div style={{ display:"flex",gap:10,marginBottom:18,flexWrap:"wrap" }}>
          <div style={{ background:`${s.verdictColor}10`,border:`1px solid ${s.verdictColor}40`,borderRadius:10,padding:"14px 22px",textAlign:"center",minWidth:150 }}>
            <div style={{ fontSize:10,color:"#9ca3af",marginBottom:4,textTransform:"uppercase",letterSpacing:1 }}>Verdict</div>
            <div style={{ fontSize:22,fontWeight:700,color:s.verdictColor }}>{s.verdict}</div>
          </div>
          {[
            ["Avg suspicion", `${s.avgSusp}/100`, s.avgSusp>=60?"#ef4444":s.avgSusp>=35?"#f59e0b":"#22c55e"],
            ["Focus",         `${s.focusPct}%`,   s.focusPct>=70?"#22c55e":"#f59e0b"],
            ["Right gaze",    `${s.rightPct}%`,   s.rightPct>30?"#ef4444":"#374151"],
            ["Left gaze",     `${s.leftPct}%`,    s.leftPct>30?"#ef4444":"#374151"],
            ["Down gaze",     `${s.downPct}%`,    s.downPct>25?"#f59e0b":"#374151"],
            ["Out of frame",  `${s.offPct}%`,     s.offPct>5?"#ef4444":"#374151"],
            ["High-risk",     s.highRisk,          s.highRisk>2?"#ef4444":"#374151"],
            ["Blink rate",    `${s.blinkRate}/s`,  s.blinkRate<0.08?"#f59e0b":"#374151"],
          ].map(([lbl,val,col])=>(
            <div key={lbl} style={{ background:"#f9fafb",border:"1px solid #e5e7eb",borderRadius:10,padding:"10px 16px",flex:1,minWidth:100 }}>
              <div style={{ fontSize:9,color:"#9ca3af",marginBottom:3,textTransform:"uppercase",letterSpacing:1 }}>{lbl}</div>
              <div style={{ fontSize:19,fontWeight:700,color:col }}>{val}</div>
            </div>
          ))}
        </div>

        <div style={{ display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginBottom:14 }}>
          <div style={{ border:"1px solid #e5e7eb",borderRadius:10,padding:14 }}>
            <div style={{ fontSize:10,color:"#6b7280",textTransform:"uppercase",letterSpacing:1,marginBottom:10 }}>Zone breakdown</div>
            {Object.entries(s.zc).sort((a,b)=>b[1]-a[1]).map(([z,n])=>{
              const pct=+((n/s.N)*100).toFixed(1);
              const c=ZONE_COLOR[z]||"#6366f1";
              return (
                <div key={z} style={{ marginBottom:6 }}>
                  <div style={{ display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:2 }}>
                    <b style={{ color:c }}>{z}</b>
                    <span style={{ color:"#6b7280" }}>{pct}%</span>
                  </div>
                  <div style={{ height:5,background:"#f3f4f6",borderRadius:3,overflow:"hidden" }}>
                    <div style={{ width:`${pct}%`,height:"100%",background:c,borderRadius:3 }} />
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ border:"1px solid #e5e7eb",borderRadius:10,padding:14 }}>
            <div style={{ fontSize:10,color:"#6b7280",textTransform:"uppercase",letterSpacing:1,marginBottom:8 }}>
              Gaze heatmap — full session ({heatPoints.length} points)
            </div>
            {heatPoints.length > 0 ? (
              <div style={{ position:"relative", background:"#f8fafc", borderRadius:6, overflow:"hidden" }}>
                <svg style={{ position:"absolute",inset:0,width:"100%",height:"100%",pointerEvents:"none" }}>
                  <line x1="33.3%" y1="0" x2="33.3%" y2="100%" stroke="rgba(0,0,0,0.06)" strokeWidth="1"/>
                  <line x1="66.6%" y1="0" x2="66.6%" y2="100%" stroke="rgba(0,0,0,0.06)" strokeWidth="1"/>
                  <line x1="0" y1="30%" x2="100%" y2="30%" stroke="rgba(0,0,0,0.06)" strokeWidth="1"/>
                  <line x1="0" y1="70%" x2="100%" y2="70%" stroke="rgba(0,0,0,0.06)" strokeWidth="1"/>
                </svg>
                <canvas ref={heatOffRef} style={{ display:"none" }} />
                <canvas ref={heatResRef} width={380} height={220}
                  style={{ display:"block",width:"100%",borderRadius:6 }} />
                <div style={{ display:"flex",gap:0,height:6,margin:"6px 0 2px",borderRadius:3,overflow:"hidden" }}>
                  {["#00f","#0ff","#0f0","#ff0","#f00"].map((c,i)=>(
                    <div key={i} style={{ flex:1,background:c }} />
                  ))}
                </div>
                <div style={{ display:"flex",justifyContent:"space-between",fontSize:9,color:"#9ca3af" }}>
                  <span>low density</span><span>high density</span>
                </div>
              </div>
            ) : (
              <div style={{ color:"#d1d5db",fontSize:12,textAlign:"center",padding:"40px 0" }}>
                No heatmap data — enable heatmap during session
              </div>
            )}
          </div>
        </div>

        <div style={{ border:"1px solid #e5e7eb",borderRadius:10,padding:14,marginBottom:14 }}>
          <div style={{ fontSize:10,color:"#6b7280",textTransform:"uppercase",letterSpacing:1,marginBottom:8 }}>Suspicion timeline</div>
          <div style={{ display:"flex",gap:2,alignItems:"flex-end",height:52 }}>
            {chunks.map(c=>(
              <div key={c.ts} title={`${c.timeLabel}·${c.zone}·${c.suspicion}/100`} style={{
                flex:1,minWidth:4,borderRadius:"3px 3px 0 0",
                height:`${4+(c.suspicion/100)*48}px`,
                background:c.suspicion>=60?"#ef4444":c.suspicion>=30?"#f59e0b":"#22c55e",
              }} />
            ))}
          </div>
        </div>

        <div style={{ border:"1px solid #e5e7eb",borderRadius:10,padding:14 }}>
          <div style={{ fontSize:10,color:"#6b7280",textTransform:"uppercase",letterSpacing:1,marginBottom:8 }}>All chunks</div>
          <div style={{ maxHeight:240,overflowY:"auto" }}>
            <table style={{ width:"100%",borderCollapse:"collapse",fontSize:11 }}>
              <thead>
                <tr style={{ borderBottom:"1px solid #e5e7eb" }}>
                  {["Time","Zone","Gaze X","Gaze Y","Blinks","EAR","Suspicion"].map(h=>(
                    <th key={h} style={{ padding:"3px 8px",textAlign:"left",color:"#9ca3af",fontWeight:500 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {chunks.map((c,i)=>(
                  <tr key={c.ts} style={{ borderBottom:"1px solid #f3f4f6",background:i%2?"#fafafa":"#fff" }}>
                    <td style={{ padding:"3px 8px",color:"#6b7280" }}>{c.timeLabel}</td>
                    <td style={{ padding:"3px 8px",fontWeight:700,color:ZONE_COLOR[c.zone]||"#6366f1" }}>{c.zone}</td>
                    <td style={{ padding:"3px 8px" }}>{c.gazeCenter?`${(c.gazeCenter.x*100).toFixed(0)}%`:"—"}</td>
                    <td style={{ padding:"3px 8px" }}>{c.gazeCenter?`${(c.gazeCenter.y*100).toFixed(0)}%`:"—"}</td>
                    <td style={{ padding:"3px 8px" }}>{c.blinks}</td>
                    <td style={{ padding:"3px 8px",color:c.avgEAR<0.21?"#ef4444":"#374151" }}>{c.avgEAR}</td>
                    <td style={{ padding:"3px 8px",fontWeight:700,color:c.suspicion>=60?"#ef4444":c.suspicion>=30?"#f59e0b":"#22c55e" }}>{c.suspicion}/100</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function Btn({ color, disabled, onClick, children }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      padding:"7px 14px", borderRadius:6, fontFamily:"monospace",
      fontWeight:700, fontSize:11, cursor:disabled?"not-allowed":"pointer",
      background:`${color}18`, color, border:`1px solid ${color}50`,
      opacity:disabled?0.4:1,
    }}>
      {children}
    </button>
  );
}
function Row({ label, val, color }) {
  return (
    <div>
      <span style={{ color:"#9ca3af", marginRight:6 }}>{label}</span>
      <b style={{ color:color||"#374151" }}>{val}</b>
    </div>
  );
}