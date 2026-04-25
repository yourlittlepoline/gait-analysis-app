import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

const LANDMARKS = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
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

const CONNECTIONS = [
  ["nose", "midShoulder"],
  ["leftShoulder", "rightShoulder"],
  ["leftShoulder", "leftHip"],
  ["rightShoulder", "rightHip"],
  ["leftHip", "rightHip"],
  ["midShoulder", "midHip"],
  ["leftShoulder", "leftElbow"],
  ["leftElbow", "leftWrist"],
  ["rightShoulder", "rightElbow"],
  ["rightElbow", "rightWrist"],
  ["leftHip", "leftKnee"],
  ["leftKnee", "leftAnkle"],
  ["leftAnkle", "leftHeel"],
  ["leftHeel", "leftFootIndex"],
  ["leftAnkle", "leftFootIndex"],
  ["rightHip", "rightKnee"],
  ["rightKnee", "rightAnkle"],
  ["rightAnkle", "rightHeel"],
  ["rightHeel", "rightFootIndex"],
  ["rightAnkle", "rightFootIndex"],
];

const COLORS = {
  head: "#fb7185",
  trunk: "#3b82f6",
  leftArm: "#8b5cf6",
  rightArm: "#38bdf8",
  leftLeg: "#22c55e",
  rightLeg: "#f59e0b",
  weak: "rgba(255,255,255,0.28)",
};

const GAIT_PHASES = [
  {
    id: "loading_response",
    label: "Loading response",
    focus: "контакт, амортизация, стабильность колена",
    norms: { knee: { min: 5, max: 25 }, ankle: { min: 80, max: 120 } },
  },
  {
    id: "mid_stance",
    label: "Mid stance",
    focus: "опора, перенос тела над стопой",
    norms: { knee: { min: 0, max: 15 }, ankle: { min: 85, max: 120 } },
  },
  {
    id: "terminal_stance",
    label: "Terminal stance",
    focus: "перекат через стопу, пятка поднимается",
    norms: { knee: { min: 0, max: 20 }, ankle: { min: 80, max: 125 } },
  },
  {
    id: "pre_swing",
    label: "Pre-swing",
    focus: "отрыв, подготовка к переносу",
    norms: { knee: { min: 20, max: 50 }, ankle: { min: 70, max: 120 } },
  },
  {
    id: "initial_swing",
    label: "Initial swing",
    focus: "перенос, clearance стопы",
    norms: { knee: { min: 40, max: 75 }, ankle: { min: 75, max: 125 } },
  },
  {
    id: "mid_swing",
    label: "Mid swing",
    focus: "перенос ноги вперёд",
    norms: { knee: { min: 25, max: 65 }, ankle: { min: 80, max: 125 } },
  },
  {
    id: "terminal_swing",
    label: "Terminal swing",
    focus: "подготовка к контакту пяткой",
    norms: { knee: { min: 0, max: 25 }, ankle: { min: 80, max: 125 } },
  },
];

const PHOTO_VIEWS = [
  { id: "front", label: "Фото спереди" },
  { id: "side", label: "Фото сбоку" },
  { id: "back", label: "Фото сзади" },
];

const DEFAULT_PHASE = "mid_swing";
const MIN_CONFIDENCE = 55;

function getPhaseById(id) {
  return GAIT_PHASES.find((phase) => phase.id === id) ?? GAIT_PHASES.find((phase) => phase.id === DEFAULT_PHASE);
}

function midpoint(a, b, name) {
  if (!a || !b) return null;
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
    name,
  };
}

function angleAt(a, b, c) {
  if (!a || !b || !c) return null;
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const l1 = Math.hypot(v1.x, v1.y);
  const l2 = Math.hypot(v2.x, v2.y);
  if (!l1 || !l2) return null;
  const cos = Math.max(-1, Math.min(1, dot / (l1 * l2)));
  return Math.round((Math.acos(cos) * 180) / Math.PI);
}

function segmentAngle(a, b) {
  if (!a || !b) return null;
  return Math.round((Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI);
}

function buildPoints(landmarks, width, height) {
  const points = {};
  Object.entries(LANDMARKS).forEach(([name, index]) => {
    const lm = landmarks[index];
    if (!lm) return;
    points[name] = {
      x: lm.x * width,
      y: lm.y * height,
      visibility: lm.visibility ?? 1,
      name,
    };
  });
  points.midShoulder = midpoint(points.leftShoulder, points.rightShoulder, "midShoulder");
  points.midHip = midpoint(points.leftHip, points.rightHip, "midHip");
  return points;
}

function connectionColor(a, b) {
  if (a === "nose" || b === "nose") return COLORS.head;
  if (
    a.includes("Shoulder") ||
    b.includes("Shoulder") ||
    a.includes("Hip") ||
    b.includes("Hip") ||
    a.startsWith("mid") ||
    b.startsWith("mid")
  )
    return COLORS.trunk;
  if (a.startsWith("left") && (a.includes("Elbow") || a.includes("Wrist") || b.includes("Elbow") || b.includes("Wrist")))
    return COLORS.leftArm;
  if (a.startsWith("right") && (a.includes("Elbow") || a.includes("Wrist") || b.includes("Elbow") || b.includes("Wrist")))
    return COLORS.rightArm;
  if (a.startsWith("left") || b.startsWith("left")) return COLORS.leftLeg;
  return COLORS.rightLeg;
}

function pointColor(name) {
  if (name === "nose") return COLORS.head;
  if (name.includes("Shoulder") || name.includes("Hip")) return COLORS.trunk;
  if (name.startsWith("left") && (name.includes("Elbow") || name.includes("Wrist"))) return COLORS.leftArm;
  if (name.startsWith("right") && (name.includes("Elbow") || name.includes("Wrist"))) return COLORS.rightArm;
  if (name.startsWith("left")) return COLORS.leftLeg;
  return COLORS.rightLeg;
}

function drawSkeleton(canvas, img, landmarks) {
  const ctx = canvas.getContext("2d");
  const width = img.naturalWidth || img.videoWidth || img.width;
  const height = img.naturalHeight || img.videoHeight || img.height;

  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(img, 0, 0, width, height);

  if (!landmarks?.length) return null;

  const points = buildPoints(landmarks, width, height);
  const scale = Math.max(0.7, width / 900);

  CONNECTIONS.forEach(([a, b]) => {
    const p1 = points[a];
    const p2 = points[b];
    if (!p1 || !p2) return;
    const conf = Math.min(p1.visibility ?? 1, p2.visibility ?? 1);
    ctx.strokeStyle = conf < 0.45 ? COLORS.weak : connectionColor(a, b);
    ctx.lineWidth = Math.max(2, 4 * scale);
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  });

  Object.keys(LANDMARKS).forEach((name) => {
    const p = points[name];
    if (!p) return;
    ctx.fillStyle = (p.visibility ?? 1) < 0.45 ? COLORS.weak : pointColor(name);
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(4, 6 * scale), 0, Math.PI * 2);
    ctx.fill();
  });

  const confidence = Math.round(
    (Object.keys(LANDMARKS).reduce((sum, key) => sum + (points[key]?.visibility ?? 0), 0) / Object.keys(LANDMARKS).length) * 100
  );

  const torsoFromVertical =
    points.midHip && points.midShoulder ? Math.abs(90 - Math.abs(segmentAngle(points.midHip, points.midShoulder))) : null;

  return {
    points,
    confidence,
    leftKnee: angleAt(points.leftHip, points.leftKnee, points.leftAnkle),
    rightKnee: angleAt(points.rightHip, points.rightKnee, points.rightAnkle),
    leftAnkle: angleAt(points.leftKnee, points.leftAnkle, points.leftFootIndex),
    rightAnkle: angleAt(points.rightKnee, points.rightAnkle, points.rightFootIndex),
    leftFootAngle: segmentAngle(points.leftHeel, points.leftFootIndex),
    rightFootAngle: segmentAngle(points.rightHeel, points.rightFootIndex),
    pelvisTilt: segmentAngle(points.leftHip, points.rightHip),
    torsoTilt: segmentAngle(points.midHip, points.midShoulder),
    torsoFromVertical,
  };
}

function isOutsideNorm(value, range) {
  if (value === null || value === undefined || !range) return false;
  return value < range.min || value > range.max;
}

function describeNorm(value, range) {
  if (value === null || value === undefined || !range) return "нет данных";
  if (value < range.min) return "ниже ориентира";
  if (value > range.max) return "выше ориентира";
  return "в пределах ориентира";
}

function makeZone(area, level, note, source, priority = 1) {
  return { area, level, note, source, priority };
}

function classifyVideoFrame(metrics, phaseId = DEFAULT_PHASE) {
  const phase = getPhaseById(phaseId);
  const zones = [];

  if (!metrics) {
    return {
      type: "video",
      phase,
      score: 0,
      verdict: "нет данных",
      zones: [makeZone("Качество кадра", "warning", "тело не найдено", "video", 3)],
    };
  }

  if (metrics.confidence < MIN_CONFIDENCE) {
    zones.push(makeZone("Качество кадра", "warning", "модель плохо видит тело — лучше заменить кадр", "video", 3));
  }

  [
    ["Левое колено", metrics.leftKnee, phase.norms.knee],
    ["Правое колено", metrics.rightKnee, phase.norms.knee],
  ].forEach(([label, value, norm]) => {
    if (isOutsideNorm(value, norm)) zones.push(makeZone(label, "attention", `${describeNorm(value, norm)} для фазы ${phase.label}`, "video", 2));
  });

  [
    ["Левый голеностоп/стопа", metrics.leftAnkle, phase.norms.ankle],
    ["Правый голеностоп/стопа", metrics.rightAnkle, phase.norms.ankle],
  ].forEach(([label, value, norm]) => {
    if (isOutsideNorm(value, norm)) zones.push(makeZone(label, "attention", `${describeNorm(value, norm)} для фазы ${phase.label}`, "video", 2));
  });

  if ((metrics.torsoFromVertical ?? 0) > 12) {
    zones.push(makeZone("Корпус/спина", "attention", "заметный наклон корпуса — проверь компенсацию", "video", 2));
  }

  if (Math.abs(metrics.pelvisTilt ?? 0) > 12) {
    zones.push(makeZone("Таз", "attention", "таз заметно наклонён — проверь перекос или наклон камеры", "video", 2));
  }

  const footDiff =
    metrics.leftFootAngle !== null && metrics.rightFootAngle !== null ? Math.abs(metrics.leftFootAngle - metrics.rightFootAngle) : 0;

  if (footDiff > 35) {
    zones.push(makeZone("Стопа/голеностоп", "attention", "стопы выглядят асимметрично по направлению пятка-носок", "video", 2));
  }

  if (!zones.length) zones.push(makeZone("Общий вид", "ok", "грубых флагов по выбранной фазе не видно", "video", 0));

  const score = zones.reduce((sum, z) => sum + (z.level === "warning" ? 2 : z.level === "attention" ? 1 : 0), 0);

  return {
    type: "video",
    phase,
    score,
    verdict: score >= 4 ? "несколько зон внимания" : score >= 1 ? "есть отдельные зоны внимания" : "грубых отклонений не видно",
    zones,
  };
}

function classifyStaticImage(metrics, view) {
  const zones = [];

  if (!metrics) {
    return {
      type: "photo",
      view,
      score: 0,
      verdict: "нет данных",
      zones: [makeZone("Качество фото", "warning", "тело не найдено на фото", `photo:${view}`, 3)],
    };
  }

  if (metrics.confidence < MIN_CONFIDENCE) {
    zones.push(makeZone("Качество фото", "warning", "модель плохо видит тело — фото ненадёжно", `photo:${view}`, 3));
  }

  if (view === "side") {
    if ((metrics.torsoFromVertical ?? 0) > 10) {
      zones.push(makeZone("Корпус/спина", "attention", "наклон корпуса в статике сбоку", "photo:side", 2));
    }

    const kneeAverage = [metrics.leftKnee, metrics.rightKnee].filter(Boolean).reduce((a, b) => a + b, 0) / [metrics.leftKnee, metrics.rightKnee].filter(Boolean).length;
    if (Number.isFinite(kneeAverage) && kneeAverage < 160) {
      zones.push(makeZone("Колени", "attention", "колено выглядит заметно согнутым в стойке сбоку", "photo:side", 2));
    }
  }

  if (view === "front" || view === "back") {
    if (Math.abs(metrics.pelvisTilt ?? 0) > 8) {
      zones.push(makeZone("Таз", "attention", "асимметрия линии таза в стойке", `photo:${view}`, 2));
    }

    const kneeDiff = metrics.leftKnee && metrics.rightKnee ? Math.abs(metrics.leftKnee - metrics.rightKnee) : 0;
    if (kneeDiff > 12) {
      zones.push(makeZone("Колени", "attention", "асимметрия положения коленей в стойке", `photo:${view}`, 2));
    }

    const ankleDiff = metrics.leftAnkle && metrics.rightAnkle ? Math.abs(metrics.leftAnkle - metrics.rightAnkle) : 0;
    if (ankleDiff > 18) {
      zones.push(makeZone("Стопа/голеностоп", "attention", "асимметрия стоп или голеностопа в стойке", `photo:${view}`, 2));
    }
  }

  if (!zones.length) zones.push(makeZone("Статика", "ok", "грубых статических флагов не видно", `photo:${view}`, 0));

  const score = zones.reduce((sum, z) => sum + (z.level === "warning" ? 2 : z.level === "attention" ? 1 : 0), 0);

  return {
    type: "photo",
    view,
    score,
    verdict: score >= 3 ? "есть зоны внимания в статике" : score >= 1 ? "есть отдельные статические флаги" : "грубых статических флагов не видно",
    zones,
  };
}

function buildHintEngine({ videoResults, photoResults }) {
  const allZones = [...videoResults.flatMap((r) => r.analysis.zones), ...photoResults.flatMap((r) => r.analysis.zones)];
  const attentionZones = allZones.filter((z) => z.level !== "ok");

  const byArea = attentionZones.reduce((acc, zone) => {
    acc[zone.area] = acc[zone.area] ?? [];
    acc[zone.area].push(zone);
    return acc;
  }, {});

  const hints = [];

  if (byArea["Качество кадра"]?.length || byArea["Качество фото"]?.length) {
    hints.push({
      area: "Качество данных",
      level: "warning",
      title: "Сначала проверь качество входных данных",
      checks: [
        "снимай человека целиком, без обрезанных стоп и головы",
        "камера должна быть примерно на уровне таза или середины тела",
        "лучше однотонный фон и контрастная одежда",
      ],
    });
  }

  if (byArea["Корпус/спина"]?.length) {
    hints.push({
      area: "Корпус/спина",
      level: "attention",
      title: "Есть наклон или компенсация корпусом",
      checks: [
        "проверь, не заваливается ли корпус при переносе",
        "сравни видео с фото сбоку",
        "проверь, не компенсирует ли человек слабость ноги корпусом",
      ],
    });
  }

  if (byArea["Таз"]?.length) {
    hints.push({
      area: "Таз",
      level: "attention",
      title: "Есть флаг по тазу",
      checks: ["проверь наклон камеры", "сравни фото спереди/сзади", "проверь, не уходит ли таз при опоре на одну ногу"],
    });
  }

  if (Object.keys(byArea).some((area) => area.includes("Колено"))) {
    hints.push({
      area: "Колено",
      level: "attention",
      title: "Есть флаг по колену",
      checks: ["смотри фазу кадра: опора или перенос", "проверь, хватает ли сгибания в переносе", "проверь, не блокирует ли ортез движение"],
    });
  }

  if (Object.keys(byArea).some((area) => area.includes("Голеностоп") || area.includes("стопа") || area.includes("Стопа"))) {
    hints.push({
      area: "Стопа/голеностоп",
      level: "attention",
      title: "Есть флаг по стопе или голеностопу",
      checks: ["проверь clearance носка", "сравни пятка-носок в фазе переноса", "проверь, не цепляет ли носок из-за колена или таза"],
    });
  }

  if (!hints.length) {
    hints.push({
      area: "Общий вывод",
      level: "ok",
      title: "Грубых зон внимания не найдено",
      checks: ["проверь выбранные фазы", "проверь качество кадров", "сравни до/после ортеза, если есть оба видео"],
    });
  }

  return hints;
}

function UploadScreen({ status, videoUrl, staticImages, onUploadVideo, onUploadImage, onContinue }) {
  const hasAnyInput = Boolean(videoUrl || staticImages.front || staticImages.side || staticImages.back);

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="rounded-3xl bg-slate-900/80 border border-slate-800 p-6 shadow-xl">
          <p className="text-sm text-slate-400">MVP gait analytics</p>
          <h1 className="text-3xl font-bold mt-2">Анализ походки: фото + видео</h1>
          <p className="text-slate-300 mt-3 max-w-3xl">
            Загрузи видео походки и/или статические фото спереди, сбоку и сзади. Система не ставит диагноз, а ищет зоны внимания и даёт подсказки для проверки ортеза, стопы, колена, таза и компенсаций.
          </p>
          <div className="mt-4 text-sm text-slate-400">Статус: {status}</div>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="rounded-3xl bg-slate-900 border border-slate-800 p-5">
            <h2 className="text-xl font-semibold">1. Видео походки</h2>
            <p className="text-sm text-slate-400 mt-2">Лучше боковая съёмка, человек целиком, камера неподвижна.</p>
            <label className="mt-4 block rounded-2xl border border-dashed border-slate-600 p-5 cursor-pointer hover:bg-slate-800/60 transition">
              <input type="file" accept="video/*" className="hidden" onChange={(e) => onUploadVideo(e.target.files?.[0])} />
              <span className="text-slate-200">Выбрать видео</span>
            </label>
            {videoUrl && <video src={videoUrl} controls className="mt-4 w-full rounded-2xl border border-slate-700" />}
          </div>

          <div className="rounded-3xl bg-slate-900 border border-slate-800 p-5">
            <h2 className="text-xl font-semibold">2. Фото стойки</h2>
            <p className="text-sm text-slate-400 mt-2">Можно загрузить одно фото или все три: спереди, сбоку, сзади.</p>
            <div className="mt-4 grid gap-3">
              {PHOTO_VIEWS.map((view) => (
                <label key={view.id} className="rounded-2xl border border-dashed border-slate-600 p-4 cursor-pointer hover:bg-slate-800/60 transition">
                  <input type="file" accept="image/*" className="hidden" onChange={(e) => onUploadImage(view.id, e.target.files?.[0])} />
                  <span>{view.label}</span>
                  {staticImages[view.id] && <span className="ml-2 text-emerald-400">✓ загружено</span>}
                </label>
              ))}
            </div>
          </div>
        </div>

        <button
          disabled={!hasAnyInput}
          onClick={onContinue}
          className="w-full rounded-2xl bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-400 text-slate-950 font-bold py-4 hover:bg-emerald-400 transition"
        >
          Продолжить к анализу
        </button>
      </div>
    </div>
  );
}

function MetricPill({ label, value }) {
  return (
    <div className="rounded-2xl bg-slate-800/80 border border-slate-700 px-3 py-2">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="font-semibold">{value ?? "—"}</div>
    </div>
  );
}

function ZoneBadge({ level }) {
  const label = level === "warning" ? "качество" : level === "attention" ? "внимание" : "ок";
  const cls = level === "warning" ? "bg-amber-400 text-slate-950" : level === "attention" ? "bg-rose-400 text-slate-950" : "bg-emerald-400 text-slate-950";
  return <span className={`rounded-full px-2 py-1 text-xs font-bold ${cls}`}>{label}</span>;
}

function ResultCard({ title, result }) {
  const metrics = result.metrics;
  return (
    <div className="rounded-3xl bg-slate-900 border border-slate-800 p-5 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-bold">{title}</h3>
          <p className="text-sm text-slate-400">{result.analysis.verdict}</p>
        </div>
        <div className="text-right text-sm text-slate-400">score: {result.analysis.score}</div>
      </div>

      {result.imageUrl && <img src={result.imageUrl} alt={title} className="w-full rounded-2xl border border-slate-700" />}
      {result.canvasUrl && <img src={result.canvasUrl} alt={title} className="w-full rounded-2xl border border-slate-700" />}

      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <MetricPill label="confidence" value={`${metrics.confidence}%`} />
          <MetricPill label="левое колено" value={metrics.leftKnee ? `${metrics.leftKnee}°` : null} />
          <MetricPill label="правое колено" value={metrics.rightKnee ? `${metrics.rightKnee}°` : null} />
          <MetricPill label="таз" value={metrics.pelvisTilt ? `${metrics.pelvisTilt}°` : null} />
        </div>
      )}

      <div className="space-y-2">
        {result.analysis.zones.map((zone, index) => (
          <div key={`${zone.area}-${index}`} className="rounded-2xl bg-slate-800/70 border border-slate-700 p-3">
            <div className="flex items-center justify-between gap-2">
              <b>{zone.area}</b>
              <ZoneBadge level={zone.level} />
            </div>
            <p className="text-sm text-slate-300 mt-1">{zone.note}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function HintPanel({ hints }) {
  return (
    <div className="rounded-3xl bg-slate-900 border border-slate-800 p-5">
      <h2 className="text-2xl font-bold">Движок подсказок</h2>
      <p className="text-sm text-slate-400 mt-2">Это не диагноз. Это список вещей, которые стоит проверить глазами и руками.</p>
      <div className="mt-4 grid md:grid-cols-2 gap-3">
        {hints.map((hint, index) => (
          <div key={`${hint.area}-${index}`} className="rounded-2xl bg-slate-800/70 border border-slate-700 p-4">
            <div className="flex items-center justify-between gap-2">
              <h3 className="font-bold">{hint.title}</h3>
              <ZoneBadge level={hint.level} />
            </div>
            <ul className="mt-3 space-y-2 text-sm text-slate-300 list-disc pl-5">
              {hint.checks.map((check) => (
                <li key={check}>{check}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

function AnalysisScreen({ status, videoUrl, staticImages, phaseId, setPhaseId, videoResults, photoResults, hints, onAnalyze, onBack }) {
  return (
    <div className="min-h-screen bg-slate-950 text-white p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="rounded-3xl bg-slate-900/80 border border-slate-800 p-6 shadow-xl flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <p className="text-sm text-slate-400">MVP gait analytics</p>
            <h1 className="text-3xl font-bold mt-2">Единый анализ фото + видео</h1>
            <p className="text-slate-300 mt-2">Статус: {status}</p>
          </div>
          <div className="flex flex-col sm:flex-row gap-3">
            <select value={phaseId} onChange={(e) => setPhaseId(e.target.value)} className="rounded-2xl bg-slate-800 border border-slate-700 px-4 py-3">
              {GAIT_PHASES.map((phase) => (
                <option key={phase.id} value={phase.id}>
                  {phase.label}
                </option>
              ))}
            </select>
            <button onClick={onAnalyze} className="rounded-2xl bg-emerald-500 text-slate-950 font-bold px-5 py-3 hover:bg-emerald-400 transition">
              Запустить анализ
            </button>
            <button onClick={onBack} className="rounded-2xl bg-slate-800 border border-slate-700 px-5 py-3 hover:bg-slate-700 transition">
              Назад
            </button>
          </div>
        </div>

        <div className="rounded-3xl bg-slate-900 border border-slate-800 p-5">
          <h2 className="text-xl font-bold">Выбранная фаза видео</h2>
          <p className="text-slate-300 mt-1">{getPhaseById(phaseId).focus}</p>
          <p className="text-sm text-slate-500 mt-2">Важно: пока фаза выбирается вручную. Это честнее, чем пытаться угадывать фазу по патологическим компенсаторным углам.</p>
        </div>

        {hints.length > 0 && <HintPanel hints={hints} />}

        <div className="grid lg:grid-cols-2 gap-4">
          {videoResults.map((result, index) => (
            <ResultCard key={`video-${index}`} title={`Видео-кадр ${index + 1}`} result={result} />
          ))}
          {photoResults.map((result) => {
            const viewLabel = PHOTO_VIEWS.find((v) => v.id === result.view)?.label ?? result.view;
            return <ResultCard key={result.view} title={viewLabel} result={result} />;
          })}
        </div>

        {!videoResults.length && !photoResults.length && (
          <div className="rounded-3xl bg-slate-900 border border-slate-800 p-8 text-center text-slate-400">
            Нажми «Запустить анализ», чтобы увидеть разметку, зоны внимания и подсказки.
          </div>
        )}
      </div>
    </div>
  );
}

function createImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function createVideoFromUrl(url) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    video.onloadedmetadata = () => resolve(video);
    video.onerror = reject;
  });
}

function seekVideo(video, time) {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = Math.min(Math.max(time, 0), Math.max(video.duration - 0.05, 0));
  });
}

function canvasToUrl(canvas) {
  return canvas.toDataURL("image/jpeg", 0.92);
}

export default function GaitPhotoVideoAnalyzer() {
  const [status, setStatus] = useState("загрузка модели...");
  const [landmarker, setLandmarker] = useState(null);
  const [screen, setScreen] = useState("upload");
  const [videoUrl, setVideoUrl] = useState("");
  const [staticImages, setStaticImages] = useState({ front: "", side: "", back: "" });
  const [phaseId, setPhaseId] = useState(DEFAULT_PHASE);
  const [videoResults, setVideoResults] = useState([]);
  const [photoResults, setPhotoResults] = useState([]);
  const [hints, setHints] = useState([]);
  const tempCanvasRef = useRef(null);

  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "GPU",
          },
          runningMode: "IMAGE",
          numPoses: 1,
        });

        if (!mounted) return;
        setLandmarker(poseLandmarker);
        setStatus("модель готова");
      } catch (error) {
        console.error(error);
        if (!mounted) return;
        setStatus("ошибка загрузки модели");
      }
    }

    init();
    return () => {
      mounted = false;
    };
  }, []);

  function handleUploadVideo(file) {
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setVideoResults([]);
    setHints([]);
  }

  function handleUploadImage(view, file) {
    if (!file) return;
    setStaticImages((prev) => {
      if (prev[view]) URL.revokeObjectURL(prev[view]);
      return { ...prev, [view]: URL.createObjectURL(file) };
    });
    setPhotoResults([]);
    setHints([]);
  }

  async function analyzeImageUrl(url, view) {
    const img = await createImageFromUrl(url);
    const result = landmarker.detect(img);
    const canvas = tempCanvasRef.current ?? document.createElement("canvas");
    const metrics = drawSkeleton(canvas, img, result.landmarks?.[0]);
    const analysis = classifyStaticImage(metrics, view);
    return {
      type: "photo",
      view,
      imageUrl: canvasToUrl(canvas),
      metrics,
      analysis,
    };
  }

  async function analyzeVideoUrl(url) {
    const video = await createVideoFromUrl(url);
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
    const frameTimes = [0.2, 0.4, 0.6, 0.8].map((k) => duration * k);
    const canvas = tempCanvasRef.current ?? document.createElement("canvas");
    const results = [];

    for (const time of frameTimes) {
      await seekVideo(video, time);
      const pose = landmarker.detect(video);
      const metrics = drawSkeleton(canvas, video, pose.landmarks?.[0]);
      const analysis = classifyVideoFrame(metrics, phaseId);
      results.push({
        type: "video",
        time,
        canvasUrl: canvasToUrl(canvas),
        metrics,
        analysis,
      });
    }

    return results;
  }

  async function runAnalysis() {
    if (!landmarker) {
      setStatus("модель ещё не готова");
      return;
    }

    setStatus("анализирую...");
    setVideoResults([]);
    setPhotoResults([]);
    setHints([]);

    try {
      const nextVideoResults = videoUrl ? await analyzeVideoUrl(videoUrl) : [];

      const imageEntries = Object.entries(staticImages).filter(([, url]) => Boolean(url));
      const nextPhotoResults = [];
      for (const [view, url] of imageEntries) {
        const result = await analyzeImageUrl(url, view);
        nextPhotoResults.push(result);
      }

      const nextHints = buildHintEngine({ videoResults: nextVideoResults, photoResults: nextPhotoResults });
      setVideoResults(nextVideoResults);
      setPhotoResults(nextPhotoResults);
      setHints(nextHints);
      setStatus("анализ готов");
    } catch (error) {
      console.error(error);
      setStatus("ошибка анализа — проверь файл или перезагрузи страницу");
    }
  }

  return (
    <>
      <canvas ref={tempCanvasRef} className="hidden" />
      {screen === "upload" ? (
        <UploadScreen
          status={status}
          videoUrl={videoUrl}
          staticImages={staticImages}
          onUploadVideo={handleUploadVideo}
          onUploadImage={handleUploadImage}
          onContinue={() => setScreen("analysis")}
        />
      ) : (
        <AnalysisScreen
          status={status}
          videoUrl={videoUrl}
          staticImages={staticImages}
          phaseId={phaseId}
          setPhaseId={setPhaseId}
          videoResults={videoResults}
          photoResults={photoResults}
          hints={hints}
          onAnalyze={runAnalysis}
          onBack={() => setScreen("upload")}
        />
      )}
    </>
  );
}
