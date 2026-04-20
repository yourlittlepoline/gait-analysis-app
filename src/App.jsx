import React, { useEffect, useMemo, useRef, useState } from "react";
import { Upload, CheckCircle2 } from "lucide-react";

const LANDMARKS = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
  leftHeel: 29,
  rightHeel: 30,
  leftFootIndex: 31,
  rightFootIndex: 32,
};

const PHASES = {
  loadingResponse: {
    title: "Loading response",
    focus: "приём веса, начальный контакт, колено",
    norm: { hip: 25, knee: 15, ankle: 5, shank: 8 },
  },
  midStance: {
    title: "Mid stance",
    focus: "одиночная опора, контроль голени и колена",
    norm: { hip: 0, knee: 5, ankle: 5, shank: 5 },
  },
  terminalStance: {
    title: "Terminal stance",
    focus: "поздняя опора, tibial progression, push-off",
    norm: { hip: -10, knee: 0, ankle: 10, shank: 15 },
  },
  swingClearance: {
    title: "Swing clearance",
    focus: "перенос, clearance стопы, сгибание колена",
    norm: { hip: 20, knee: 60, ankle: 0, shank: 0 },
  },
};

const PHASE_ORDER = {
  loadingResponse: 0,
  midStance: 1,
  terminalStance: 2,
  swingClearance: 3,
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function radToDeg(rad) {
  return (rad * 180) / Math.PI;
}

function angle3(a, b, c) {
  if (!a || !b || !c) return null;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!mag) return null;
  return radToDeg(Math.acos(clamp(dot / mag, -1, 1)));
}

function signedAngleToVertical(top, bottom) {
  if (!top || !bottom) return null;
  const dx = bottom.x - top.x;
  const dy = bottom.y - top.y;
  return radToDeg(Math.atan2(dx, dy));
}

function signedAngle(a, b) {
  if (!a || !b) return null;
  return radToDeg(Math.atan2(-(b.y - a.y), b.x - a.x));
}

function angleBetweenVectors(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);
  if (!mag) return null;
  return radToDeg(Math.acos(clamp(dot / mag, -1, 1)));
}

function distance(a, b) {
  if (!a || !b) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function getPoint(landmarks, idx, width, height) {
  const p = landmarks?.[idx];
  if (!p) return null;
  return {
    x: p.x * width,
    y: p.y * height,
    z: p.z ?? 0,
    visibility: p.visibility ?? 0,
  };
}

function computeFootAngle(heel, toe) {
  if (!heel || !toe) return 0;
  return signedAngle(heel, toe) ?? 0;
}

function computeAnkleFromShankAndFoot(knee, ankle, heel, toe) {
  if (!knee || !ankle || !heel || !toe) return 0;

  const shank = { x: knee.x - ankle.x, y: knee.y - ankle.y };
  const foot = { x: toe.x - heel.x, y: toe.y - heel.y };

  const raw = angleBetweenVectors(shank, foot);
  if (raw == null) return 0;

  return 90 - raw;
}

function extractLeg(landmarks, side, width, height) {
  const shoulder = getPoint(
    landmarks,
    side === "left" ? LANDMARKS.leftShoulder : LANDMARKS.rightShoulder,
    width,
    height
  );
  const hip = getPoint(
    landmarks,
    side === "left" ? LANDMARKS.leftHip : LANDMARKS.rightHip,
    width,
    height
  );
  const knee = getPoint(
    landmarks,
    side === "left" ? LANDMARKS.leftKnee : LANDMARKS.rightKnee,
    width,
    height
  );
  const ankle = getPoint(
    landmarks,
    side === "left" ? LANDMARKS.leftAnkle : LANDMARKS.rightAnkle,
    width,
    height
  );
  const heel = getPoint(
    landmarks,
    side === "left" ? LANDMARKS.leftHeel : LANDMARKS.rightHeel,
    width,
    height
  );
  const toe = getPoint(
    landmarks,
    side === "left" ? LANDMARKS.leftFootIndex : LANDMARKS.rightFootIndex,
    width,
    height
  );

  const footPoint =
    heel && toe
      ? {
          x: (heel.x + toe.x) / 2,
          y: (heel.y + toe.y) / 2,
          z: ((heel.z ?? 0) + (toe.z ?? 0)) / 2,
        }
      : toe || heel;

  const rawKnee = angle3(hip, knee, ankle) ?? 180;
  const kneeFlexion = Math.max(0, 180 - rawKnee);

  const hipAngle = signedAngleToVertical(hip, knee) ?? 0;
  const shankAngle = signedAngleToVertical(knee, ankle) ?? 0;

  const footAngle = computeFootAngle(heel, toe);
  const ankleAngle = computeAnkleFromShankAndFoot(knee, ankle, heel, toe);

  const legSize =
    distance(hip, knee) + distance(knee, ankle) + distance(ankle, footPoint);

  const toeClearance =
    toe && heel ? Math.max(-40, Math.min(40, heel.y - toe.y)) : 0;

  const isFootOnGround =
    heel && toe ? Math.abs(heel.y - toe.y) < height * 0.02 : false;

  const depthPoints = [hip, knee, ankle, heel, toe].filter(Boolean);
  const meanZ = depthPoints.length
    ? depthPoints.reduce((s, p) => s + (p.z ?? 0), 0) / depthPoints.length
    : 0;

  return {
    side,
    points: { shoulder, hip, knee, ankle, heel, toe, footPoint },
    metrics: {
      hip: hipAngle,
      knee: kneeFlexion,
      ankle: ankleAngle,
      shankAngle,
      footAngle,
      toeClearance,
      legSize,
      meanZ,
      isFootOnGround,
    },
  };
}

function qualityScore(leg, width, height) {
  if (!leg) return 0;
  const pts = [
    leg.points.shoulder,
    leg.points.hip,
    leg.points.knee,
    leg.points.ankle,
    leg.points.footPoint,
  ].filter(Boolean);

  if (pts.length < 5) return 0;

  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const bodyHeight = Math.max(...ys) - Math.min(...ys);
  const centerX = xs.reduce((a, b) => a + b, 0) / xs.length;

  let score = 0;
  if (bodyHeight > height * 0.35) score += 40;
  if (centerX > width * 0.08 && centerX < width * 0.92) score += 20;
  if (leg.metrics.legSize > height * 0.22) score += 20;
  if (pts.length >= 5) score += 20;
  return score;
}

function chooseNearLeg(landmarks, width, height, previousSide = null) {
  const left = extractLeg(landmarks, "left", width, height);
  const right = extractLeg(landmarks, "right", width, height);
  const leftScore = qualityScore(left, width, height);
  const rightScore = qualityScore(right, width, height);

  if (leftScore < 60 && rightScore < 60) return null;

  let chosen = left.metrics.meanZ <= right.metrics.meanZ ? left : right;

  if (previousSide) {
    const prev = previousSide === "left" ? left : right;
    const alt = previousSide === "left" ? right : left;
    const prevGood = qualityScore(prev, width, height);
    const altGood = qualityScore(alt, width, height);

    if (
      prevGood >= 60 &&
      Math.abs(prev.metrics.meanZ - alt.metrics.meanZ) < 0.08
    ) {
      chosen = prev;
    } else if (
      prevGood >= 60 &&
      prev.metrics.meanZ <= alt.metrics.meanZ + 0.03
    ) {
      chosen = prev;
    } else if (altGood >= 60) {
      chosen = alt;
    }
  }

  return chosen;
}

function buildProgressionLine(samples) {
  const valid = samples.filter((s) => s?.points?.hip && s?.points?.footPoint);
  if (valid.length < 2) return null;
  const first = valid[0].points.hip;
  const last = valid[valid.length - 1].points.hip;
  return { start: first, end: last };
}

function progressionAngle(line) {
  if (!line) return 0;
  return signedAngle(line.start, line.end) ?? 0;
}

function footRelativeToProgression(heel, toe, line) {
  if (!heel || !toe) return 0;
  const footAxis = signedAngle(heel, toe) ?? 0;
  return footAxis - progressionAngle(line);
}

function phaseScore(metrics, phaseKey) {
  const shankForward = Math.max(0, metrics.shankAngle);

  if (phaseKey === "loadingResponse") {
    return (
      Math.abs(metrics.knee - 15) * 1.6 +
      Math.abs(metrics.ankle - 5) * 0.2 +
      Math.abs(shankForward - 8) * 1.0 +
      Math.abs(metrics.footProgression) * 0.08
    );
  }

  if (phaseKey === "midStance") {
    return (
      Math.abs(metrics.knee - 5) * 1.5 +
      Math.abs(shankForward - 5) * 1.6 +
      Math.abs(metrics.ankle - 5) * 0.2 +
      Math.abs(metrics.hip) * 0.8
    );
  }

  if (phaseKey === "terminalStance") {
    return (
      Math.abs(metrics.ankle - 10) * 0.25 +
      Math.abs(shankForward - 15) * 1.8 +
      Math.abs(metrics.knee) * 1.2 +
      Math.abs(metrics.hip + 10) * 0.9
    );
  }

  if (phaseKey === "swingClearance") {
    return (
      Math.abs(metrics.knee - 60) * 1.5 +
      Math.max(0, 12 - metrics.toeClearance) * 0.5 +
      Math.abs(metrics.footProgression) * 0.08 +
      Math.abs(metrics.hip - 20) * 0.8
    );
  }

  return 999;
}

function rankPhases(metrics) {
  if (!metrics.isFootOnGround) {
    return {
      bestPhase: "swingClearance",
      bestCost: 0,
      secondPhase: null,
      secondCost: 0,
      confidence: 0.9,
      ranked: [],
    };
  }

  const ranked = Object.keys(PHASES)
    .map((phaseKey) => ({
      phaseKey,
      cost: phaseScore(metrics, phaseKey),
    }))
    .sort((a, b) => a.cost - b.cost);

  const best = ranked[0];
  const second = ranked[1];
  const confidenceGap = second.cost - best.cost;
  const confidence = clamp(confidenceGap / 20, 0, 1);

  return {
    bestPhase: best.phaseKey,
    bestCost: best.cost,
    secondPhase: second.phaseKey,
    secondCost: second.cost,
    confidence,
    ranked,
  };
}

function detectFrameLevelPathology(metrics, ranked) {
  const flags = [];

  if (ranked.bestCost > 58) flags.push("poor_phase_fit");
  if (ranked.confidence < 0.08) flags.push("phase_ambiguity");

  if (metrics.knee > 90) flags.push("excessive_knee_flexion");

  if (
    ranked.bestPhase === "terminalStance" &&
    metrics.knee < 5 &&
    metrics.shankAngle > 14
  ) {
    flags.push("possible_knee_hyperextension_pattern");
  }

  if (ranked.bestPhase === "swingClearance" && metrics.toeClearance < -8) {
    flags.push("low_toe_clearance");
  }

  if (ranked.bestPhase === "swingClearance" && metrics.knee < 18) {
    flags.push("insufficient_knee_flexion_in_swing");
  }

  return flags;
}

function analyzeSequencePathology(candidates) {
  if (!candidates.length) {
    return {
      isPathological: false,
      pathologyScore: 0,
      reliabilityScore: 0,
      confidence: 0,
      reasons: ["no_candidates"],
      enriched: [],
      phaseReliability: "low",
    };
  }

  const enriched = candidates.map((c) => {
    const ranked = rankPhases(c.metrics);
    const frameFlags = detectFrameLevelPathology(c.metrics, ranked);

    return {
      ...c,
      ranked,
      frameFlags,
      assignedPhase: ranked.bestPhase,
      phaseIndex: PHASE_ORDER[ranked.bestPhase],
    };
  });

  let pathologyScore = 0;
  let reliabilityPenalty = 0;
  const reasons = [];

  const poorFitCount = enriched.filter((e) =>
    e.frameFlags.includes("poor_phase_fit")
  ).length;

  const ambiguousCount = enriched.filter((e) =>
    e.frameFlags.includes("phase_ambiguity")
  ).length;

  const mechanicalExtremeCount = enriched.filter((e) =>
    e.frameFlags.some((f) =>
      [
        "excessive_knee_flexion",
        "possible_knee_hyperextension_pattern",
        "insufficient_knee_flexion_in_swing",
        "low_toe_clearance",
      ].includes(f)
    )
  ).length;

  if (poorFitCount >= Math.ceil(enriched.length * 0.5)) {
    reliabilityPenalty += 2;
    reasons.push("many_frames_fit_no_normal_phase_well");
  }

  if (ambiguousCount >= Math.ceil(enriched.length * 0.45)) {
    reliabilityPenalty += 1;
    reasons.push("phase_assignment_is_ambiguous");
  }

  if (mechanicalExtremeCount >= Math.ceil(enriched.length * 0.3)) {
    pathologyScore += 2;
    reasons.push("multiple_frames_have_extreme_deviations");
  }

  let backwardsJumps = 0;
  for (let i = 1; i < enriched.length; i++) {
    const prev = enriched[i - 1].phaseIndex;
    const curr = enriched[i].phaseIndex;
    if (curr + 1 < prev) backwardsJumps += 1;
  }

  if (backwardsJumps >= 4) {
    reliabilityPenalty += 1;
    reasons.push("phase_progression_is_inconsistent");
  }

  const kneeValues = enriched.map((e) => e.metrics.knee);
  const range = (arr) => Math.max(...arr) - Math.min(...arr);

  if (range(kneeValues) > 95) {
    pathologyScore += 1;
    reasons.push("knee_motion_is_highly_irregular");
  }

  const meanConfidence =
    enriched.reduce((sum, e) => sum + (e.ranked?.confidence ?? 0), 0) /
    enriched.length;

  let phaseReliability = "high";
  if (reliabilityPenalty >= 3 || meanConfidence < 0.12) phaseReliability = "low";
  else if (reliabilityPenalty >= 1 || meanConfidence < 0.22)
    phaseReliability = "medium";

  const isPathological = pathologyScore >= 3;

  const confidence = clamp(
    0.55 * (pathologyScore / 4) +
      0.45 * (1 - Math.min(reliabilityPenalty / 4, 1)),
    0,
    1
  );

  return {
    isPathological,
    pathologyScore,
    reliabilityScore: reliabilityPenalty,
    confidence,
    reasons,
    enriched,
    phaseReliability,
  };
}

function footAssessment(metrics, phaseKey) {
  if (phaseKey === "swingClearance") {
    if (metrics.toeClearance < 8) {
      return `Стопа: низкий clearance, риск зацепа · clearance ≈ ${metrics.toeClearance.toFixed(
        0
      )} px`;
    }
    return `Стопа: clearance выглядит приемлемо · heel-to-toe угол ≈ ${metrics.footAngle.toFixed(
      0
    )}°`;
  }

  return `Стопа: heel-to-toe угол ≈ ${metrics.footAngle.toFixed(0)}°`;
}

function makeText(
  metrics,
  phaseKey,
  side,
  pathologySummary = null,
  frameFlags = [],
  ranked = null
) {
  const ref = PHASES[phaseKey].norm;

  let summary = "Паттерн ближе к нормотипичной интерпретации.";

  if (pathologySummary?.isPathological) {
    summary =
      "Есть признаки атипичной биомеханики; фазовая разметка может быть ограниченно надёжной.";
  } else if (pathologySummary?.phaseReliability === "low") {
    summary =
      "Фазовая разметка ненадёжна: кадр плохо укладывается в типичный цикл, но это ещё не значит, что походка патологическая.";
  } else if (ranked && ranked.confidence < 0.1) {
    summary = "Кадр неоднозначен: он плохо отделяется от соседних фаз.";
  }

  const warnings = [];
  if (frameFlags.includes("low_toe_clearance")) warnings.push("низкий clearance");
  if (frameFlags.includes("insufficient_knee_flexion_in_swing")) {
    warnings.push("недостаточное сгибание колена в переносе");
  }
  if (frameFlags.includes("possible_knee_hyperextension_pattern")) {
    warnings.push("возможный recurvatum / нестабильность колена");
  }
  if (frameFlags.includes("poor_phase_fit")) {
    warnings.push("кадр плохо укладывается в нормальные фазы");
  }
  if (frameFlags.includes("phase_ambiguity")) {
    warnings.push("фаза определяется неоднозначно");
  }

  return {
    phaseTitle: PHASES[phaseKey].title,
    focus: PHASES[phaseKey].focus,
    side: side === "left" ? "левая (ближняя)" : "правая (ближняя)",
    summary,
    warnings: warnings.length ? warnings.join(", ") : "грубых предупреждений нет",
    hip: `Таз: видео ≈ ${metrics.hip.toFixed(0)}°, норма ${ref.hip}°`,
    knee: `Колено: видео ≈ ${metrics.knee.toFixed(0)}°, норма ${ref.knee}°`,
    ankle: `Голеностоп: видео ≈ ${metrics.ankle.toFixed(0)}°, норма ${ref.ankle}°`,
    shank: `Голень: наклон вперёд ≈ ${Math.max(
      0,
      metrics.shankAngle
    ).toFixed(0)}°, норма ${ref.shank}°`,
    foot: footAssessment(metrics, phaseKey),
  };
}

async function loadPoseLandmarker() {
  const tasksVision = await import("@mediapipe/tasks-vision");
  const { FilesetResolver, PoseLandmarker } = tasksVision;

  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  try {
    return await PoseLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (gpuError) {
    console.warn("GPU delegate failed, fallback to CPU", gpuError);

    return await PoseLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  }
}

function drawPoint(ctx, p, color = "#22c55e", r = 5) {
  if (!p) return;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawLine(ctx, a, b, color = "#38bdf8", width = 3, dash = []) {
  if (!a || !b) return;
  ctx.save();
  ctx.setLineDash(dash);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function drawLegOverlay(ctx, frame, width, height) {
  const leg = frame?.leg;
  if (!leg) return;

  const { shoulder, hip, knee, ankle, heel, toe } = leg.points;

  drawLine(ctx, shoulder, hip, "#64748b", 3);
  drawLine(ctx, hip, knee, "#64748b", 4);
  drawLine(ctx, knee, ankle, "#64748b", 4);

  drawLine(ctx, heel, toe, "#22c55e", 4);

  drawPoint(ctx, shoulder, "#94a3b8", 4);
  drawPoint(ctx, hip, "#f59e0b", 5);
  drawPoint(ctx, knee, "#f59e0b", 5);
  drawPoint(ctx, ankle, "#f59e0b", 5);
  drawPoint(ctx, heel, "#22c55e", 5);
  drawPoint(ctx, toe, "#22c55e", 5);

  if (frame.progressionLine) {
    drawLine(
      ctx,
      frame.progressionLine.start,
      frame.progressionLine.end,
      "#60a5fa",
      2,
      [8, 6]
    );
  }

  ctx.fillStyle = "rgba(15,23,42,0.75)";
  ctx.fillRect(14, 14, 380, 120);

  ctx.fillStyle = "white";
  ctx.font = "600 18px Inter, Arial";
  ctx.fillText(`Фаза: ${frame.text.phaseTitle}`, 24, 40);

  ctx.font = "14px Inter, Arial";
  ctx.fillText(`Сторона: ${frame.text.side}`, 24, 64);
  ctx.fillText(`Колено: ${frame.leg.metrics.knee.toFixed(0)}°`, 24, 86);
  ctx.fillText(`Стопа heel→toe: ${frame.leg.metrics.footAngle.toFixed(0)}°`, 24, 108);

  if (frame.pathologySummary?.isPathological) {
    ctx.fillStyle = "#ef4444";
    ctx.fillRect(width - 240, 14, 220, 34);
    ctx.fillStyle = "white";
    ctx.font = "600 14px Inter, Arial";
    ctx.fillText("Возможна патологическая походка", width - 228, 36);
  } else if (frame.pathologySummary?.phaseReliability !== "high") {
    ctx.fillStyle = "#f59e0b";
    ctx.fillRect(width - 250, 14, 230, 34);
    ctx.fillStyle = "white";
    ctx.font = "600 14px Inter, Arial";
    ctx.fillText("Фазовая разметка ненадёжна", width - 238, 36);
  }

  if (frame.frameFlags?.length) {
    ctx.fillStyle = "rgba(15,23,42,0.78)";
    ctx.fillRect(14, height - 70, width - 28, 52);
    ctx.fillStyle = "#f8fafc";
    ctx.font = "13px Inter, Arial";
    ctx.fillText(`Предупреждения: ${frame.text.warnings}`, 24, height - 38);
  }
}

function FrameCard({ frame, isActive, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        border: isActive ? "2px solid #fff" : "1px solid #334155",
        background: "#0f172a",
        color: "white",
        borderRadius: 16,
        padding: 12,
        width: "100%",
        textAlign: "left",
        cursor: "pointer",
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 6 }}>{frame.step}</div>
      <div style={{ fontSize: 13, opacity: 0.9 }}>{frame.text.phaseTitle}</div>
      <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>
        колено {frame.leg.metrics.knee.toFixed(0)}° · стопа{" "}
        {frame.leg.metrics.footAngle.toFixed(0)}°
      </div>
    </button>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const poseLandmarkerRef = useRef(null);

  const [videoUrl, setVideoUrl] = useState("");
  const [isReady, setIsReady] = useState(false);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [status, setStatus] = useState("Загрузите видео");
  const [error, setError] = useState("");
  const [frames, setFrames] = useState([]);
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [pathologySummary, setPathologySummary] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        setIsLoadingModel(true);
        setError("");
        setStatus("Загрузка модели...");

        const poseLandmarker = await loadPoseLandmarker();

        if (cancelled) return;

        poseLandmarkerRef.current = poseLandmarker;
        setIsReady(true);
        setStatus("Модель готова");
      } catch (e) {
        console.error("Model init error:", e);
        if (!cancelled) {
          setError(`Не удалось загрузить модель: ${e?.message || "unknown error"}`);
          setStatus("Ошибка загрузки модели");
          setIsReady(false);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingModel(false);
        }
      }
    }

    init();

    return () => {
      cancelled = true;
    };
  }, []);

  const currentFrame = frames[selectedFrame] ?? null;
  const canAnalyze = !!videoUrl && isReady && isVideoReady && !isLoadingModel;

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !currentFrame) return;

    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 540;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    drawLegOverlay(ctx, currentFrame, canvas.width, canvas.height);
  }, [currentFrame, videoUrl]);

  const summaryText = useMemo(() => {
    if (!pathologySummary) return null;

    if (pathologySummary.isPathological) {
      return {
        title: "Есть признаки патологической / атипичной походки",
        tone: "#7f1d1d",
        bg: "#fee2e2",
      };
    }

    if (pathologySummary.phaseReliability !== "high") {
      return {
        title: "Фазовая разметка ограниченно надёжна",
        tone: "#78350f",
        bg: "#fef3c7",
      };
    }

    return {
      title: "Паттерн ближе к нормотипичному",
      tone: "#14532d",
      bg: "#dcfce7",
    };
  }, [pathologySummary]);

  async function analyzeVideo() {
    try {
      setError("");
      setFrames([]);
      setSelectedFrame(0);
      setPathologySummary(null);

      const video = videoRef.current;
      const poseLandmarker = poseLandmarkerRef.current;

      if (!poseLandmarker) {
        setError("Модель ещё не готова");
        return;
      }

      if (!video) {
        setError("Видео не найдено");
        return;
      }

      if (!isVideoReady || !video.videoWidth || !video.videoHeight) {
        setError("Видео ещё не загрузило метаданные");
        return;
      }

      const width = video.videoWidth;
      const height = video.videoHeight;
      const duration = video.duration || 0;

      if (!duration || !Number.isFinite(duration)) {
        setError("Не удалось получить длительность видео");
        return;
      }

      setStatus("Анализ видео...");

      const sampleCount = 8;
      const timePoints = Array.from({ length: sampleCount }, (_, i) =>
        (duration * (i + 1)) / (sampleCount + 1)
      );

      const collected = [];
      let previousSide = null;

      for (let i = 0; i < timePoints.length; i++) {
        const t = timePoints[i];

        await new Promise((resolve) => {
          const onSeeked = () => {
            video.removeEventListener("seeked", onSeeked);
            resolve();
          };
          video.addEventListener("seeked", onSeeked, { once: true });
          video.currentTime = t;
        });

        const nowMs = Math.round(t * 1000);
        const result = poseLandmarker.detectForVideo(video, nowMs);
        const landmarks = result?.landmarks?.[0];

        if (!landmarks) continue;

        const leg = chooseNearLeg(landmarks, width, height, previousSide);
        if (!leg) continue;

        previousSide = leg.side;

        collected.push({
          step: `Кадр ${collected.length + 1}`,
          time: t,
          leg,
        });
      }

      if (!collected.length) {
        setError("Не удалось стабильно распознать ногу на видео");
        setStatus("Нет данных");
        return;
      }

      const progressionLine = buildProgressionLine(collected.map((c) => c.leg));

      const candidates = collected.map((item) => {
        const footProgression = footRelativeToProgression(
          item.leg.points.heel,
          item.leg.points.toe,
          progressionLine
        );

        return {
          ...item,
          points: item.leg.points,
          metrics: {
            ...item.leg.metrics,
            footProgression,
          },
        };
      });

      const pathology = analyzeSequencePathology(candidates);

      const finalFrames = pathology.enriched.map((frame, idx) => {
        const text = makeText(
          frame.metrics,
          frame.assignedPhase,
          frame.side,
          pathology,
          frame.frameFlags,
          frame.ranked
        );

        return {
          ...frame,
          step: `Кадр ${idx + 1}`,
          text,
          progressionLine,
          pathologySummary: pathology,
        };
      });

      setFrames(finalFrames);
      setPathologySummary(pathology);
      setSelectedFrame(0);
      setStatus("Анализ завершён");

      if (video.duration) {
        video.currentTime = 0;
      }
    } catch (e) {
      console.error("Analyze error:", e);
      setError(`Ошибка анализа видео: ${e?.message || "unknown error"}`);
      setStatus("Ошибка");
    }
  }

  function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }

    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setFrames([]);
    setSelectedFrame(0);
    setPathologySummary(null);
    setStatus("Видео загружено");
    setError("");
    setIsVideoReady(false);
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#020617",
        color: "white",
        padding: 24,
        fontFamily: "Inter, Arial, sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: 1400,
          margin: "0 auto",
          display: "grid",
          gridTemplateColumns: "1.15fr 0.85fr",
          gap: 24,
        }}
      >
        <div
          style={{
            background: "#0f172a",
            border: "1px solid #1e293b",
            borderRadius: 24,
            padding: 20,
          }}
        >
          <div
            style={{
              display: "flex",
              gap: 12,
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              marginBottom: 18,
            }}
          >
            <div>
              <h1 style={{ margin: 0, fontSize: 28 }}>
                Анализ походки по видео
              </h1>
              <div style={{ opacity: 0.7, marginTop: 6 }}>
                Ближняя нога, heel-to-toe стопа, фазовая разметка и флаг патологичности
              </div>
            </div>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <label
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 8,
                  background: "#1d4ed8",
                  padding: "12px 16px",
                  borderRadius: 14,
                  cursor: "pointer",
                  fontWeight: 600,
                }}
              >
                <Upload size={18} />
                Загрузить видео
                <input
                  type="file"
                  accept="video/*"
                  onChange={onUpload}
                  style={{ display: "none" }}
                />
              </label>

              <button
                onClick={analyzeVideo}
                disabled={!canAnalyze}
                style={{
                  background: !canAnalyze ? "#334155" : "#16a34a",
                  color: "white",
                  border: "none",
                  padding: "12px 16px",
                  borderRadius: 14,
                  cursor: !canAnalyze ? "not-allowed" : "pointer",
                  fontWeight: 700,
                }}
              >
                Анализировать
              </button>
            </div>
          </div>

          <div
            style={{
              background: "#020617",
              borderRadius: 18,
              overflow: "hidden",
              border: "1px solid #1e293b",
            }}
          >
            {videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  playsInline
                  onLoadedMetadata={() => {
                    setIsVideoReady(true);
                    setStatus((prev) =>
                      prev === "Модель готова" ? "Видео и модель готовы" : prev
                    );
                  }}
                  style={{
                    width: "100%",
                    maxHeight: 420,
                    display: currentFrame ? "none" : "block",
                    background: "black",
                  }}
                />
                <canvas
                  ref={canvasRef}
                  style={{
                    width: "100%",
                    display: currentFrame ? "block" : "none",
                    background: "black",
                  }}
                />
              </>
            ) : (
              <div
                style={{
                  height: 420,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#94a3b8",
                }}
              >
                Сначала загрузи видео
              </div>
            )}
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginTop: 14,
              color: "#cbd5e1",
            }}
          >
            <CheckCircle2 size={16} />
            <span>
              {isLoadingModel
                ? "Модель загружается..."
                : isReady && isVideoReady
                ? "Видео и модель готовы"
                : isReady
                ? "Модель готова"
                : status}
            </span>
          </div>

          {error && (
            <div
              style={{
                marginTop: 14,
                background: "#7f1d1d",
                color: "white",
                padding: 14,
                borderRadius: 14,
              }}
            >
              {error}
            </div>
          )}
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 16,
          }}
        >
          {summaryText && (
            <div
              style={{
                background: summaryText.bg,
                color: summaryText.tone,
                borderRadius: 20,
                padding: 18,
              }}
            >
              <div style={{ fontWeight: 800, marginBottom: 8 }}>
                {summaryText.title}
              </div>

              {pathologySummary && (
                <div style={{ fontSize: 14, lineHeight: 1.5 }}>
                  confidence ≈ {(pathologySummary.confidence * 100).toFixed(0)}%
                  {" · "}
                  reliability: {pathologySummary.phaseReliability}
                  {" · "}
                  pathology score: {pathologySummary.pathologyScore}
                </div>
              )}
            </div>
          )}

          <div
            style={{
              background: "#0f172a",
              border: "1px solid #1e293b",
              borderRadius: 24,
              padding: 18,
            }}
          >
            <div style={{ fontWeight: 800, fontSize: 18, marginBottom: 14 }}>
              Кадры анализа
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 10,
              }}
            >
              {frames.map((frame, idx) => (
                <FrameCard
                  key={`${frame.step}-${idx}`}
                  frame={frame}
                  isActive={idx === selectedFrame}
                  onClick={() => setSelectedFrame(idx)}
                />
              ))}
            </div>

            {!frames.length && (
              <div style={{ color: "#94a3b8", fontSize: 14 }}>
                После анализа здесь появятся выбранные кадры.
              </div>
            )}
          </div>

          {currentFrame && (
            <div
              style={{
                background: "#0f172a",
                border: "1px solid #1e293b",
                borderRadius: 24,
                padding: 18,
                lineHeight: 1.55,
              }}
            >
              <div style={{ fontWeight: 800, fontSize: 20, marginBottom: 10 }}>
                {currentFrame.text.phaseTitle}
              </div>

              <div style={{ color: "#cbd5e1", marginBottom: 10 }}>
                {currentFrame.text.summary}
              </div>

              <div style={{ color: "#94a3b8", marginBottom: 12 }}>
                Фокус: {currentFrame.text.focus}
              </div>

              <div style={{ marginBottom: 8 }}>{currentFrame.text.hip}</div>
              <div style={{ marginBottom: 8 }}>{currentFrame.text.knee}</div>
              <div style={{ marginBottom: 8 }}>{currentFrame.text.ankle}</div>
              <div style={{ marginBottom: 8 }}>{currentFrame.text.shank}</div>
              <div style={{ marginBottom: 8 }}>{currentFrame.text.foot}</div>

              <div
                style={{
                  marginTop: 14,
                  padding: 12,
                  borderRadius: 14,
                  background: "#020617",
                  color: "#f8fafc",
                }}
              >
                <strong>Предупреждения:</strong> {currentFrame.text.warnings}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
