import React, { useEffect, useRef, useState } from "react";
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
    focus: "приём веса, контакт стопы, колено",
    norm: { hip: 25, knee: 15, ankle: 5, footProgression: 0 },
  },
  midStance: {
    title: "Mid stance",
    focus: "контроль голени и стабильность колена",
    norm: { hip: 0, knee: 5, ankle: 5, footProgression: 0 },
  },
  terminalStance: {
    title: "Terminal stance",
    focus: "tibia progression и push-off",
    norm: { hip: -10, knee: 0, ankle: 10, footProgression: 5 },
  },
  swingClearance: {
    title: "Swing clearance",
    focus: "clearance стопы и сгибание колена",
    norm: { hip: 20, knee: 60, ankle: 0, footProgression: 0 },
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

function extractLeg(landmarks, side, width, height) {
  const shoulder = getPoint(landmarks, side === "left" ? LANDMARKS.leftShoulder : LANDMARKS.rightShoulder, width, height);
  const hip = getPoint(landmarks, side === "left" ? LANDMARKS.leftHip : LANDMARKS.rightHip, width, height);
  const knee = getPoint(landmarks, side === "left" ? LANDMARKS.leftKnee : LANDMARKS.rightKnee, width, height);
  const ankle = getPoint(landmarks, side === "left" ? LANDMARKS.leftAnkle : LANDMARKS.rightAnkle, width, height);
  const heel = getPoint(landmarks, side === "left" ? LANDMARKS.leftHeel : LANDMARKS.rightHeel, width, height);
  const toe = getPoint(landmarks, side === "left" ? LANDMARKS.leftFootIndex : LANDMARKS.rightFootIndex, width, height);

  const footPoint = heel && toe
    ? { x: (heel.x + toe.x) / 2, y: (heel.y + toe.y) / 2, z: ((heel.z ?? 0) + (toe.z ?? 0)) / 2 }
    : toe || heel;

  const rawKnee = angle3(hip, knee, ankle) ?? 180;
  const kneeFlexion = Math.max(0, 180 - rawKnee);
  const rawAnkle = angle3(knee, ankle, footPoint) ?? 90;
  const ankleAngle = 90 - rawAnkle;
  const hipAngle = signedAngleToVertical(hip, knee) ?? 0;
  const shankAngle = signedAngleToVertical(knee, ankle) ?? 0;
  const legSize = distance(hip, knee) + distance(knee, ankle) + distance(ankle, footPoint);
  const toeClearance = toe && heel ? Math.max(-40, Math.min(40, heel.y - toe.y)) : 0;
  const depthPoints = [hip, knee, ankle, heel, toe].filter(Boolean);
  const meanZ = depthPoints.length ? depthPoints.reduce((s, p) => s + (p.z ?? 0), 0) / depthPoints.length : 0;

  return {
    side,
    points: { shoulder, hip, knee, ankle, heel, toe, footPoint },
    metrics: {
      hip: hipAngle,
      knee: kneeFlexion,
      ankle: ankleAngle,
      shankAngle,
      toeClearance,
      legSize,
      meanZ,
    },
  };
}

function qualityScore(leg, width, height) {
  if (!leg) return 0;
  const pts = [leg.points.shoulder, leg.points.hip, leg.points.knee, leg.points.ankle, leg.points.footPoint].filter(Boolean);
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
    if (prevGood >= 60 && Math.abs(prev.metrics.meanZ - alt.metrics.meanZ) < 0.08) {
      chosen = prev;
    } else if (prevGood >= 60 && prev.metrics.meanZ <= alt.metrics.meanZ + 0.03) {
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

function footRelativeToProgression(ankle, footPoint, line) {
  if (!ankle || !footPoint) return 0;
  const footAngle = signedAngle(ankle, footPoint) ?? 0;
  return footAngle - progressionAngle(line);
}

function phaseScore(metrics, phaseKey) {
  const shankForward = Math.max(0, metrics.shankAngle);

  if (phaseKey === "loadingResponse") {
    return (
      Math.abs(metrics.knee - 15) * 1.4 +
      Math.abs(metrics.ankle - 5) +
      Math.abs(shankForward - 8) * 0.8 +
      Math.abs(metrics.footProgression) * 0.6
    );
  }
  if (phaseKey === "midStance") {
    return (
      Math.abs(metrics.knee - 5) * 1.2 +
      Math.abs(shankForward - 5) * 1.5 +
      Math.abs(metrics.ankle - 5) +
      Math.abs(metrics.hip) * 0.7
    );
  }
  if (phaseKey === "terminalStance") {
    return (
      Math.abs(metrics.ankle - 10) * 1.4 +
      Math.abs(shankForward - 15) * 1.6 +
      Math.abs(metrics.knee) +
      Math.abs(metrics.hip + 10) * 0.8
    );
  }
  if (phaseKey === "swingClearance") {
    return (
      Math.abs(metrics.knee - 60) * 1.2 +
      Math.max(0, 12 - metrics.toeClearance) * 2.2 +
      Math.abs(metrics.footProgression) * 0.6 +
      Math.abs(metrics.hip - 20) * 0.7
    );
  }
  return 999;
}

function rankPhases(metrics) {
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

  // Было слишком чувствительно
  if (ranked.bestCost > 52) flags.push("poor_phase_fit");
  if (ranked.confidence < 0.1) flags.push("phase_ambiguity");

  // Оставляем только более грубые отклонения
  if (metrics.knee > 85) flags.push("excessive_knee_flexion");
  if (metrics.knee < 0 && metrics.shankAngle > 12) {
    flags.push("possible_knee_hyperextension_pattern");
  }

  if (metrics.ankle < -35) flags.push("marked_plantarflexion");
  if (metrics.ankle > 35) flags.push("marked_dorsiflexion");
  if (ranked.bestPhase === "swingClearance" && metrics.toeClearance < -8) {
  flags.push("low_toe_clearance");
}
  if (Math.abs(metrics.footProgression) > 28) {
    flags.push("marked_foot_progression_deviation");
  }

  if (ranked.bestPhase === "terminalStance" && metrics.ankle < 0) {
    flags.push("weak_tibial_progression_or_push_off");
  }

  if (ranked.bestPhase === "swingClearance" && metrics.knee < 20) {
    flags.push("insufficient_knee_flexion_in_swing");
  }

  if (ranked.bestPhase === "loadingResponse" && metrics.ankle < -15) {
    flags.push("forefoot_or_plantarflexed_contact");
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
        "marked_plantarflexion",
        "marked_dorsiflexion",
        "low_toe_clearance",
        "marked_foot_progression_deviation",
        "weak_tibial_progression_or_push_off",
        "insufficient_knee_flexion_in_swing",
        "forefoot_or_plantarflexed_contact",
      ].includes(f)
    )
  ).length;

  // Это больше про ненадёжность фаз, а не про патологию
  if (poorFitCount >= Math.ceil(enriched.length * 0.5)) {
    reliabilityPenalty += 2;
    reasons.push("many_frames_fit_no_normal_phase_well");
  }

  if (ambiguousCount >= Math.ceil(enriched.length * 0.45)) {
    reliabilityPenalty += 1;
    reasons.push("phase_assignment_is_ambiguous");
  }

  // Реальные грубые биомеханические отклонения
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
  const ankleValues = enriched.map((e) => e.metrics.ankle);
  const toeValues = enriched.map((e) => e.metrics.toeClearance);

  const range = (arr) => Math.max(...arr) - Math.min(...arr);

  if (range(kneeValues) > 95) {
    pathologyScore += 1;
    reasons.push("knee_motion_is_highly_irregular");
  }

  if (range(ankleValues) > 55) {
    pathologyScore += 1;
    reasons.push("ankle_motion_is_highly_irregular");
  }

  if (Math.min(...toeValues) < 0) {
    pathologyScore += 1;
    reasons.push("toe_clearance_drops_below_zero");
  }

  const meanConfidence =
    enriched.reduce((sum, e) => sum + (e.ranked?.confidence ?? 0), 0) /
    enriched.length;

  let phaseReliability = "high";
  if (reliabilityPenalty >= 3 || meanConfidence < 0.12) phaseReliability = "low";
  else if (reliabilityPenalty >= 1 || meanConfidence < 0.22) phaseReliability = "medium";

  // Главное: плохая фазовая определимость != патологическая походка
  const isPathological = pathologyScore >= 3;

  const confidence = clamp(
    0.55 * (pathologyScore / 4) + 0.45 * (1 - Math.min(reliabilityPenalty / 4, 1)),
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
    if (metrics.toeClearance < 8) return `Стопа: низкий clearance, риск зацепа · clearance ≈ ${metrics.toeClearance.toFixed(0)} px`;
    if (metrics.footProgression < -10) return `Стопа: стопа свисает вниз в swing · угол к линии шага ≈ ${metrics.footProgression.toFixed(0)}°`;
    return `Стопа: clearance выглядит приемлемо · clearance ≈ ${metrics.toeClearance.toFixed(0)} px`;
  }
  if (phaseKey === "loadingResponse") {
    if (metrics.footProgression < -12) return `Стопа: выраженная plantarflexed посадка · угол к линии шага ≈ ${metrics.footProgression.toFixed(0)}°`;
    if (metrics.footProgression > 10) return `Стопа: слишком dorsiflexed контакт · угол к линии шага ≈ ${metrics.footProgression.toFixed(0)}°`;
    return `Стопа: контакт ближе к ожидаемому · угол к линии шага ≈ ${metrics.footProgression.toFixed(0)}°`;
  }
  if (phaseKey === "terminalStance") {
    if (metrics.ankle < 4) return `Стопа: мало продвижения над стопой / слабый push-off · угол к линии шага ≈ ${metrics.footProgression.toFixed(0)}°`;
    return `Стопа: push-off выглядит приемлемо · угол к линии шага ≈ ${metrics.footProgression.toFixed(0)}°`;
  }
  return `Стопа: опорная стабильность / ориентация к линии шага ≈ ${metrics.footProgression.toFixed(0)}°`;
}

function makeText(metrics, phaseKey, side, pathologySummary = null, frameFlags = [], ranked = null) {
  const ref = PHASES[phaseKey].norm;

  let summary = "Паттерн ближе к нормотипичной интерпретации.";

  if (pathologySummary?.isPathological) {
    summary = "Есть признаки атипичной биомеханики; фазовая разметка может быть ограниченно надёжной.";
  } else if (pathologySummary?.phaseReliability === "low") {
    summary = "Фазовая разметка ненадёжна: кадр плохо укладывается в типичный цикл, но это ещё не значит, что походка патологическая.";
  } else if (ranked && ranked.confidence < 0.1) {
    summary = "Кадр неоднозначен: он плохо отделяется от соседних фаз.";
  }

  const warnings = [];
  if (frameFlags.includes("marked_plantarflexion")) warnings.push("выраженная подошвенная установка стопы");
  if (frameFlags.includes("forefoot_or_plantarflexed_contact")) warnings.push("контакт больше похож на передний отдел / plantarflexed contact");
  if (frameFlags.includes("low_toe_clearance")) warnings.push("низкий clearance");
  if (frameFlags.includes("marked_foot_progression_deviation")) warnings.push("выраженное отклонение угла шага");
  if (frameFlags.includes("insufficient_knee_flexion_in_swing")) warnings.push("недостаточное сгибание колена в переносе");
  if (frameFlags.includes("weak_tibial_progression_or_push_off")) warnings.push("слабая tibial progression / push-off");
  if (frameFlags.includes("possible_knee_hyperextension_pattern")) warnings.push("возможный recurvatum / нестабильность колена");
  if (frameFlags.includes("poor_phase_fit")) warnings.push("кадр плохо укладывается в нормальные фазы");
  if (frameFlags.includes("phase_ambiguity")) warnings.push("фаза определяется неоднозначно");

  return {
    phaseTitle: PHASES[phaseKey].title,
    focus: PHASES[phaseKey].focus,
    side: side === "left" ? "левая (ближняя)" : "правая (ближняя)",
    summary,
    warnings: warnings.length ? warnings.join(", ") : "грубых предупреждений нет",
    hip: `Таз: видео ≈ ${metrics.hip.toFixed(0)}°, норма ${ref.hip}°`,
    knee: `Колено: видео ≈ ${metrics.knee.toFixed(0)}°, норма ${ref.knee}°`,
    ankle: `Голеностоп: видео ≈ ${metrics.ankle.toFixed(0)}°, норма ${ref.ankle}°`,
    shank: `Голень: наклон вперёд ≈ ${Math.max(0, metrics.shankAngle).toFixed(0)}°`,
    foot: footAssessment(metrics, phaseKey),
  };
}

async function loadPoseLandmarker() {
  const tasksVision = await import("@mediapipe/tasks-vision");
  const { FilesetResolver, PoseLandmarker } = tasksVision;
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  return PoseLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
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
        textAlign: "center",
        cursor: "pointer",
      }}
    >
      {frame.step}
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
  const [status, setStatus] = useState("Загрузите видео");
  const [error, setError] = useState("");
  const [frames, setFrames] = useState([]);
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [progression, setProgression] = useState(null);
  const [pathologySummary, setPathologySummary] = useState(null);

  const currentFrame = frames[selectedFrame] || null;

  useEffect(() => {
    return () => {
      if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  async function ensureModel() {
    if (poseLandmarkerRef.current) return poseLandmarkerRef.current;
    setIsLoadingModel(true);
    try {
      const model = await loadPoseLandmarker();
      poseLandmarkerRef.current = model;
      return model;
    } finally {
      setIsLoadingModel(false);
    }
  }

  function drawCurrent(frame) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !frame) return;

    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const leg = chooseNearLeg(frame.landmarks, canvas.width, canvas.height, frame.side);
    if (!leg) return;
    const p = leg.points;

    const line = (a, b, color = "#38bdf8", width = 5) => {
      if (!a || !b) return;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.stroke();
    };

    const dot = (a, color = "#38bdf8") => {
      if (!a) return;
      ctx.beginPath();
      ctx.arc(a.x, a.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    };

    if (progression?.start && progression?.end) {
      line(progression.start, progression.end, "#ef4444", 2);
    }

    line(p.shoulder, p.hip);
    line(p.hip, p.knee);
    line(p.knee, p.ankle);
    line(p.ankle, p.footPoint);

    [p.shoulder, p.hip, p.knee, p.ankle, p.footPoint].forEach((pt) => dot(pt));

    ctx.fillStyle = "white";
    ctx.font = "700 14px sans-serif";
    if (p.hip) ctx.fillText(`${frame.metrics.hip.toFixed(0)}°`, p.hip.x + 8, p.hip.y - 8);
    if (p.knee) ctx.fillText(`${frame.metrics.knee.toFixed(0)}°`, p.knee.x + 8, p.knee.y - 8);
    if (p.ankle) ctx.fillText(`${frame.metrics.ankle.toFixed(0)}°`, p.ankle.x + 8, p.ankle.y - 8);
    if (p.knee) ctx.fillText(`голень ${Math.max(0, frame.metrics.shankAngle).toFixed(0)}°`, p.knee.x + 12, p.knee.y + 18);

    ctx.fillStyle = "rgba(2, 6, 23, 0.8)";
    ctx.fillRect(12, 12, 320, 48);
    ctx.fillStyle = "white";
    ctx.font = "700 14px sans-serif";
    ctx.fillText(`Фаза: ${frame.text.phaseTitle}`, 22, 32);
    ctx.font = "600 12px sans-serif";
    ctx.fillText(
      pathologySummary?.isPathological ? "Атипичная походка вероятна" : "Фазовая оценка допустима",
      22,
      50
    );
  }

  async function showFrame(frame) {
    const video = videoRef.current;
    if (!video || !frame) return;

    await new Promise((resolve) => {
      const done = () => {
        video.removeEventListener("seeked", done);
        resolve();
      };
      video.addEventListener("seeked", done);
      video.currentTime = frame.time;
    });

    drawCurrent(frame);
  }

  async function analyze() {
    const video = videoRef.current;
    if (!video || !videoUrl) return;
    if (video.readyState < 2) {
      setError("Видео ещё грузится");
      return;
    }

    setError("");
    setStatus("Идёт анализ...");
    setPathologySummary(null);

    const model = await ensureModel();
    const width = video.videoWidth || 720;
    const height = video.videoHeight || 1280;
    const checkpoints = Array.from({ length: 18 }, (_, i) => 0.05 + i * 0.05).map((p) =>
      Math.min((video.duration || 1) * p, Math.max((video.duration || 1) - 0.2, 0))
    );

    const tracked = [];
    let lockedSide = null;

    for (const time of checkpoints) {
      await new Promise((resolve) => {
        const done = () => {
          video.removeEventListener("seeked", done);
          resolve();
        };
        video.addEventListener("seeked", done);
        video.currentTime = time;
      });

      const result = model.detectForVideo(video, performance.now());
      const landmarks = result?.landmarks?.[0];
      if (!landmarks) continue;

      const leg = chooseNearLeg(landmarks, width, height, lockedSide);
      if (!leg) continue;
      const score = qualityScore(leg, width, height);
      if (score < 70) continue;

      lockedSide = leg.side;
      tracked.push({
        landmarks,
        side: leg.side,
        points: leg.points,
        metrics: { ...leg.metrics },
        time,
        score,
      });
    }

    if (tracked.length < 4) {
      setError("Не удалось стабильно отследить ближнюю ногу");
      setStatus("Попробуйте видео, где одна сторона тела видна чище и дольше");
      return;
    }

    const prog = buildProgressionLine(tracked);
    setProgression(prog);

    const candidates = tracked.map((item) => {
      const footProgression = footRelativeToProgression(item.points.ankle, item.points.footPoint, prog);
      return {
        ...item,
        metrics: {
          ...item.metrics,
          footProgression,
          shankAngle: signedAngleToVertical(item.points.knee, item.points.ankle) ?? 0,
        },
      };
    });

    const pathology = analyzeSequencePathology(candidates);
    setPathologySummary(pathology);

    const usedTimes = [];
    const takeBestForPhase = (phaseKey) => {
      const sorted = [...candidates]
        .map((c) => {
          const ranked = rankPhases(c.metrics);
          const frameFlags = detectFrameLevelPathology(c.metrics, ranked);
          return {
            ...c,
            ranked,
            frameFlags,
            phaseCost: phaseScore(c.metrics, phaseKey),
          };
        })
        .sort((a, b) => a.phaseCost - b.phaseCost);

      const chosen =
        sorted.find((item) => !usedTimes.some((t) => Math.abs(t - item.time) < 0.08)) || sorted[0];

      usedTimes.push(chosen.time);

      return {
        ...chosen,
        targetPhase: phaseKey,
        text: makeText(
          chosen.metrics,
          phaseKey,
          chosen.side,
          pathology,
          chosen.frameFlags,
          chosen.ranked
        ),
      };
    };

    const selected = [
      takeBestForPhase("loadingResponse"),
      takeBestForPhase("midStance"),
      takeBestForPhase("terminalStance"),
      takeBestForPhase("swingClearance"),
    ]
      .sort((a, b) => a.time - b.time)
      .map((f, i) => ({ ...f, step: String(i + 1) }));

    setFrames(selected);

        setSelectedFrame(0);
setStatus(
  pathology.isPathological
    ? "Анализ завершён: есть признаки атипичной биомеханики"
    : pathology.phaseReliability === "low"
    ? "Анализ завершён: фазы определяются ненадёжно"
    : "Анализ завершён"
);

    if (selected.length) {
      await showFrame(selected[0]);
    }
  }

  useEffect(() => {
    if (currentFrame) {
      showFrame(currentFrame);
    }
  }, [selectedFrame, progression]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#020617",
        color: "white",
        padding: 24,
        fontFamily:
          'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      }}
    >
      <div style={{ maxWidth: 1280, margin: "0 auto" }}>
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ fontSize: 32, fontWeight: 800, marginBottom: 8 }}>
            Анализ походки
          </h1>
          <p style={{ color: "#94a3b8", fontSize: 16, lineHeight: 1.5 }}>
            Загрузите видео с сагиттальным видом. Алгоритм отслеживает ближнюю
            ногу, оценивает фазоподобные кадры и дополнительно проверяет,
            похожа ли последовательность на патологическую / атипичную походку.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(320px, 1.4fr) minmax(320px, 0.9fr)",
            gap: 24,
            alignItems: "start",
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
            <label
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 12,
                border: "1.5px dashed #334155",
                borderRadius: 20,
                padding: 18,
                marginBottom: 16,
                cursor: "pointer",
                background: "#020617",
              }}
            >
              <Upload size={20} />
              <span>Загрузить видео</span>
              <input
                type="file"
                accept="video/*"
                style={{ display: "none" }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;

                  setError("");
                  setFrames([]);
                  setSelectedFrame(0);
                  setProgression(null);
                  setPathologySummary(null);
                  setStatus("Видео загружено");

                  setVideoUrl((prev) => {
                    if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
                    return URL.createObjectURL(file);
                  });
                }}
              />
            </label>

            <div
              style={{
                position: "relative",
                width: "100%",
                aspectRatio: "9 / 16",
                background: "#000",
                borderRadius: 20,
                overflow: "hidden",
                border: "1px solid #1e293b",
              }}
            >
              {videoUrl ? (
                <>
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    playsInline
                    preload="auto"
                    controls
                    onLoadedData={() => {
                      setIsReady(true);
                      setStatus("Видео готово к анализу");
                    }}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                      display: "block",
                    }}
                  />
                  <canvas
                    ref={canvasRef}
                    style={{
                      position: "absolute",
                      inset: 0,
                      width: "100%",
                      height: "100%",
                      pointerEvents: "none",
                    }}
                  />
                </>
              ) : (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#64748b",
                    textAlign: "center",
                    padding: 24,
                  }}
                >
                  Сначала загрузите видео
                </div>
              )}
            </div>

            <div
              style={{
                display: "flex",
                gap: 12,
                marginTop: 16,
                flexWrap: "wrap",
              }}
            >
              <button
                onClick={analyze}
                disabled={!videoUrl || !isReady || isLoadingModel}
                style={{
                  background: !videoUrl || !isReady || isLoadingModel ? "#334155" : "#2563eb",
                  color: "white",
                  border: "none",
                  borderRadius: 16,
                  padding: "12px 18px",
                  fontWeight: 700,
                  cursor:
                    !videoUrl || !isReady || isLoadingModel ? "not-allowed" : "pointer",
                }}
              >
                {isLoadingModel ? "Загрузка модели..." : "Запустить анализ"}
              </button>

              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  color: "#cbd5e1",
                  fontSize: 14,
                }}
              >
                <CheckCircle2 size={16} />
                <span>{status}</span>
              </div>
            </div>

            {error ? (
              <div
                style={{
                  marginTop: 14,
                  background: "rgba(239,68,68,0.12)",
                  color: "#fecaca",
                  border: "1px solid rgba(239,68,68,0.3)",
                  padding: 12,
                  borderRadius: 14,
                }}
              >
                {error}
              </div>
            ) : null}
          </div>

          <div style={{ display: "grid", gap: 16 }}>
            <div
              style={{
                background: "#0f172a",
                border: "1px solid #1e293b",
                borderRadius: 24,
                padding: 20,
              }}
            >
              <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 12 }}>
                Сводка
              </h2>

              {pathologySummary ? (
                <div style={{ display: "grid", gap: 12 }}>
                  <div
                    style={{
                      borderRadius: 16,
                      padding: 14,
                      background: pathologySummary.isPathological
                        ? "rgba(245, 158, 11, 0.14)"
                        : "rgba(34, 197, 94, 0.14)",
                      border: pathologySummary.isPathological
                        ? "1px solid rgba(245, 158, 11, 0.35)"
                        : "1px solid rgba(34, 197, 94, 0.35)",
                    }}
                  >
                    <div style={{ fontWeight: 800, marginBottom: 6 }}>
                      {pathologySummary.isPathological
                        ? "Вероятна атипичная / патологическая походка"
                        : "Грубых признаков патологической походки не найдено"}
                    </div>
                    <div style={{ color: "#cbd5e1", fontSize: 14, lineHeight: 1.5 }}>
                      Индекс патологии: {pathologySummary.pathologyScore} · уверенность{" "}
                      {(pathologySummary.confidence * 100).toFixed(0)}%
                    </div>
                  </div>

                  <div>
                    <div
                      style={{
                        fontWeight: 700,
                        marginBottom: 8,
                        color: "#e2e8f0",
                      }}
                    >
                      Причины
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 18, color: "#cbd5e1" }}>
                      {pathologySummary.reasons?.length ? (
                        pathologySummary.reasons.map((reason, idx) => (
                          <li key={idx} style={{ marginBottom: 6 }}>
                            {reason === "many_frames_fit_no_normal_phase_well" &&
                              "Многие кадры плохо укладываются в нормальные фазы"}
                            {reason === "phase_assignment_is_ambiguous" &&
                              "Фазовая классификация часто неоднозначна"}
                            {reason === "multiple_frames_have_extreme_deviations" &&
                              "Во многих кадрах есть выраженные отклонения"}
                            {reason === "phase_progression_is_inconsistent" &&
                              "Последовательность фаз выглядит нелогичной"}
                            {reason === "knee_motion_is_highly_irregular" &&
                              "Движение колена сильно нерегулярно"}
                            {reason === "ankle_motion_is_highly_irregular" &&
                              "Движение голеностопа сильно нерегулярно"}
                          </li>
                        ))
                      ) : (
                        <li>Явных причин не выделено</li>
                      )}
                    </ul>
                  </div>
                </div>
              ) : (
                <div style={{ color: "#94a3b8" }}>
                  После анализа здесь появится общая оценка последовательности.
                </div>
              )}
            </div>

            <div
              style={{
                background: "#0f172a",
                border: "1px solid #1e293b",
                borderRadius: 24,
                padding: 20,
              }}
            >
              <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 12 }}>
                Выбранные кадры
              </h2>

              {frames.length ? (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
                    gap: 10,
                  }}
                >
                  {frames.map((frame, idx) => (
                    <FrameCard
                      key={`${frame.step}-${frame.time}`}
                      frame={frame}
                      isActive={idx === selectedFrame}
                      onClick={() => setSelectedFrame(idx)}
                    />
                  ))}
                </div>
              ) : (
                <div style={{ color: "#94a3b8" }}>
                  После анализа тут появятся 4 ключевых кадра.
                </div>
              )}
            </div>

            <div
              style={{
                background: "#0f172a",
                border: "1px solid #1e293b",
                borderRadius: 24,
                padding: 20,
              }}
            >
              <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 12 }}>
                Интерпретация кадра
              </h2>

              {currentFrame ? (
                <div style={{ display: "grid", gap: 10 }}>
                  <div style={{ fontSize: 18, fontWeight: 800 }}>
                    {currentFrame.text.phaseTitle}
                  </div>
                  <div style={{ color: "#cbd5e1" }}>
                    Сторона: {currentFrame.text.side}
                  </div>
                  <div style={{ color: "#cbd5e1" }}>
                    Фокус: {currentFrame.text.focus}
                  </div>
                  <div style={{ color: "#e2e8f0", lineHeight: 1.5 }}>
                    {currentFrame.text.summary}
                  </div>

                  <div
                    style={{
                      marginTop: 6,
                      borderRadius: 16,
                      padding: 12,
                      background: "rgba(148,163,184,0.08)",
                    }}
                  >
                    <div style={{ marginBottom: 6 }}>{currentFrame.text.hip}</div>
                    <div style={{ marginBottom: 6 }}>{currentFrame.text.knee}</div>
                    <div style={{ marginBottom: 6 }}>{currentFrame.text.ankle}</div>
                    <div style={{ marginBottom: 6 }}>{currentFrame.text.shank}</div>
                    <div>{currentFrame.text.foot}</div>
                  </div>

                  <div
                    style={{
                      color: "#fbbf24",
                      background: "rgba(251,191,36,0.08)",
                      border: "1px solid rgba(251,191,36,0.22)",
                      borderRadius: 14,
                      padding: 12,
                    }}
                  >
                    <strong>Предупреждения:</strong> {currentFrame.text.warnings}
                  </div>

                  <div style={{ color: "#94a3b8", fontSize: 13 }}>
                    Время кадра: {currentFrame.time.toFixed(2)} c · confidence:{" "}
                    {(currentFrame.ranked?.confidence * 100 || 0).toFixed(0)}%
                  </div>
                </div>
              ) : (
                <div style={{ color: "#94a3b8" }}>
                  Выбери кадр после анализа.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
