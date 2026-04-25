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
  { id: "loading_response", label: "Loading response", focus: "контакт, амортизация, стабильность колена", norms: { knee: { min: 5, max: 25 }, ankle: { min: 80, max: 120 } } },
  { id: "mid_stance", label: "Mid stance", focus: "опора, перенос тела над стопой", norms: { knee: { min: 0, max: 15 }, ankle: { min: 85, max: 120 } } },
  { id: "terminal_stance", label: "Terminal stance", focus: "перекат через стопу, пятка поднимается", norms: { knee: { min: 0, max: 20 }, ankle: { min: 80, max: 125 } } },
  { id: "pre_swing", label: "Pre-swing", focus: "отрыв, подготовка к переносу", norms: { knee: { min: 20, max: 50 }, ankle: { min: 70, max: 120 } } },
  { id: "initial_swing", label: "Initial swing", focus: "перенос, clearance стопы", norms: { knee: { min: 40, max: 75 }, ankle: { min: 75, max: 125 } } },
  { id: "mid_swing", label: "Mid swing", focus: "перенос ноги вперёд", norms: { knee: { min: 25, max: 65 }, ankle: { min: 80, max: 125 } } },
  { id: "terminal_swing", label: "Terminal swing", focus: "подготовка к контакту пяткой", norms: { knee: { min: 0, max: 25 }, ankle: { min: 80, max: 125 } } },
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
  if (a.includes("Shoulder") || b.includes("Shoulder") || a.includes("Hip") || b.includes("Hip") || a.startsWith("mid") || b.startsWith("mid")) return COLORS.trunk;
  if (a.startsWith("left") && (a.includes("Elbow") || a.includes("Wrist") || b.includes("Elbow") || b.includes("Wrist"))) return COLORS.leftArm;
  if (a.startsWith("right") && (a.includes("Elbow") || a.includes("Wrist") || b.includes("Elbow") || b.includes("Wrist"))) return COLORS.rightArm;
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
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;

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
    (Object.keys(LANDMARKS).reduce((sum, key) => sum + (points[key]?.visibility ?? 0), 0) /
      Object.keys(LANDMARKS).length) *
      100
  );

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
  };
}

function isOutsideNorm(value, range) {
  if (value === null || value === undefined || !range) return false;
  return value < range.min || value > range.max;
}

function describeNorm(value, range) {
  if (value === null || value === undefined || !range) return "нет данных";
  if (value < range.min) return `ниже нормы на ${range.min - value}°`;
  if (value > range.max) return `выше нормы на ${value - range.max}°`;
  return "в пределах ориентира";
}

function classifyFrame(metrics, phaseId = DEFAULT_PHASE) {
  const phase = getPhaseById(phaseId);
  if (!metrics) {
    return {
      verdict: "нет данных",
      score: 0,
      flags: ["тело не найдено"],
      phase,
      deviations: [],
      zones: [],
    };
  }

  const flags = [];
  const deviations = [];
  const zones = [];
  let score = 0;

  if (metrics.confidence < MIN_CONFIDENCE) {
    flags.push("кадр ненадёжный: модель плохо видит тело, не делай вывод по этому кадру");
    deviations.push({ label: "Качество", value: `${metrics.confidence}%`, norm: `≥${MIN_CONFIDENCE}%`, comment: "низкая уверенность модели" });
    zones.push({ area: "Качество кадра", level: "warning", note: "лучше заменить кадр" });
    score += 2;
  }

  const kneeItems = [
    ["L колено", metrics.leftKnee, phase.norms.knee],
    ["R колено", metrics.rightKnee, phase.norms.knee],
  ];

  const ankleItems = [
    ["L голеностоп", metrics.leftAnkle, phase.norms.ankle],
    ["R голеностоп", metrics.rightAnkle, phase.norms.ankle],
  ];

  kneeItems.forEach(([label, value, norm]) => {
    if (isOutsideNorm(value, norm)) {
      const comment = describeNorm(value, norm);
      flags.push(`${label}: ${comment} для фазы ${phase.label}`);
      deviations.push({ label, value: `${value}°`, norm: `${norm.min}–${norm.max}°`, comment });
      zones.push({ area: label, level: "attention", note: comment });
      score += 1;
    }
  });

  ankleItems.forEach(([label, value, norm]) => {
    if (isOutsideNorm(value, norm)) {
      const comment = describeNorm(value, norm);
      flags.push(`${label}: ${comment} для фазы ${phase.label}`);
      deviations.push({ label, value: `${value}°`, norm: `${norm.min}–${norm.max}°`, comment });
      zones.push({ area: label, level: "attention", note: comment });
      score += 1;
    }
  });

  const torsoFromVertical = metrics.torsoTilt === null || metrics.torsoTilt === undefined
    ? null
    : Math.abs(90 - Math.abs(metrics.torsoTilt));

  if (torsoFromVertical !== null && torsoFromVertical > 12) {
    flags.push(`корпус заметно наклонён: около ${torsoFromVertical}° от вертикали, проверь компенсацию корпусом`);
    deviations.push({ label: "Корпус", value: `${torsoFromVertical}°`, norm: "до 12° от вертикали", comment: "возможная компенсация" });
    zones.push({ area: "Корпус/спина", level: "attention", note: "наклон корпуса" });
    score += 1;
  }

  if (Math.abs(metrics.pelvisTilt ?? 0) > 12) {
    flags.push("таз заметно наклонён относительно кадра: проверь, это реальный перекос или наклон камеры");
    deviations.push({ label: "Таз", value: `${metrics.pelvisTilt}°`, norm: "ближе к 0°", comment: "перекос/наклон кадра" });
    zones.push({ area: "Таз", level: "attention", note: "наклон таза" });
    score += 1;
  }

  if (!flags.length) {
    flags.push(`для выбранной фазы ${phase.label} грубых отклонений по этому кадру не видно`);
    zones.push({ area: "Общий вид", level: "ok", note: "без грубых флагов" });
  }

  return {
    score,
    verdict: score >= 4 ? "есть несколько зон внимания" : score >= 1 ? "есть отдельные отклонения" : "грубых отклонений не видно",
    flags,
    phase,
    deviations,
    zones,
  };
}

function Metric({ label, value, norm, bad }) {
  return (
    <div className={`rounded-xl p-3 ${bad ? "bg-rose-950/50" : "bg-slate-950"}`}>
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-base font-semibold">{value ?? "—"}</div>
      {norm && <div className="mt-1 text-[11px] text-slate-500">норма: {norm}</div>}
    </div>
  );
}

function UploadScreen({ status, onUpload }) {
  return (
    <div className="min-h-screen bg-slate-950 p-6 text-white flex items-center justify-center">
      <div className="w-full max-w-xl rounded-3xl border border-slate-700 bg-slate-900 p-8 text-center shadow-2xl">
        <div className="mb-3 text-sm text-slate-400">1. Основной экран</div>
        <h1 className="text-3xl md:text-4xl font-bold">Анализ походки</h1>
        <p className="mt-3 text-slate-300">Загрузите видео. Дальше приложение нарежет его на кадры для выбора.</p>

        <label className="mt-8 inline-block cursor-pointer rounded-2xl bg-blue-600 px-6 py-3 font-semibold hover:bg-blue-500">
          Загрузить видео
          <input type="file" accept="video/*" className="hidden" onChange={onUpload} />
        </label>

        <div className="mt-6 rounded-2xl bg-slate-950 p-3 text-sm text-slate-300">{status}</div>
      </div>
    </div>
  );
}

function FrameSelectionScreen({
  status,
  videoUrl,
  videoRef,
  frames,
  selectedCount,
  selectedFrameId,
  isBusy,
  onExtractFrames,
  onToggleFrame,
  onAnalyzeSelected,
  onBack,
}) {
  return (
    <div className="min-h-screen bg-slate-950 p-4 md:p-8 text-white">
      <div className="mx-auto max-w-7xl space-y-5">
        <header className="flex flex-col md:flex-row md:items-end md:justify-between gap-3">
          <div>
            <div className="text-sm text-slate-400">2. Выбор кадров</div>
            <h1 className="text-2xl md:text-3xl font-bold">Анализ походки</h1>
            <p className="mt-1 text-slate-300">Выберите кадры, где человек полностью виден сбоку. Клик по картинке = выбрать/снять.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button onClick={onBack} className="rounded-xl bg-slate-800 px-4 py-2 font-semibold hover:bg-slate-700">Назад</button>
            <button disabled={!videoUrl || isBusy} onClick={onExtractFrames} className="rounded-xl bg-emerald-600 px-4 py-2 font-semibold hover:bg-emerald-500 disabled:opacity-50">
              {isBusy ? "Нарезаю…" : frames.length ? "Нарезать заново" : "Нарезать кадры"}
            </button>
            <button disabled={!selectedCount || isBusy} onClick={onAnalyzeSelected} className="rounded-xl bg-fuchsia-600 px-4 py-2 font-semibold hover:bg-fuchsia-500 disabled:opacity-50">
              Анализировать выбранные кадры
            </button>
          </div>
        </header>

        <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3 text-sm text-slate-200">{status}</div>
        <video ref={videoRef} src={videoUrl ?? undefined} muted playsInline className="hidden" />

        <section className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold">Выберите кадры</h2>
            <div className="text-sm text-slate-300">Выбрано: {selectedCount} / {frames.length}</div>
          </div>

          {frames.length ? (
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
              {frames.map((frame) => (
                <button
                  key={frame.id}
                  type="button"
                  onClick={() => onToggleFrame(frame)}
                  className={`relative overflow-hidden rounded-xl border-2 bg-slate-950 ${
                    frame.selected ? "border-emerald-500" : selectedFrameId === frame.id ? "border-blue-500" : "border-slate-700"
                  }`}
                >
                  <img src={frame.dataUrl} alt={`Кадр ${frame.id + 1}`} className="aspect-video w-full object-cover" />
                  <div className="absolute left-1 top-1 rounded bg-black/75 px-1.5 py-0.5 text-[10px]">{frame.id + 1}</div>
                  <div className="absolute right-1 top-1 rounded bg-black/75 px-1.5 py-0.5 text-[10px]">{frame.confidence === null ? "—" : `${frame.confidence}%`}</div>
                  <div className="absolute left-1 bottom-6 rounded bg-black/75 px-1.5 py-0.5 text-[10px]">{getPhaseById(frame.phaseId)?.label}</div>
                  <div className={`absolute inset-x-0 bottom-0 py-1 text-center text-xs font-bold ${frame.selected ? "bg-emerald-500" : "bg-black/70"}`}>
                    {frame.selected ? "✓ выбрано" : "клик = выбрать"}
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className="rounded-2xl bg-slate-950 p-10 text-center text-slate-400">Нажми “Нарезать кадры”.</div>
          )}
        </section>
      </div>
    </div>
  );
}

function AnalysisScreen({
  status,
  canvasRef,
  results,
  currentMetrics,
  currentFrame,
  currentPhase,
  currentAnalysis,
  onSetCurrentResult,
  onUpdateFramePhase,
  onBackToFrames,
}) {
  const attentionCount = results.filter((r) => r.analysis.score > 0).length;

  return (
    <div className="min-h-screen bg-[#050816] p-4 md:p-6 text-white">
      <div className="mx-auto max-w-[1120px]">
        <div className="grid grid-cols-1 lg:grid-cols-[560px_360px] gap-4 items-start justify-center">
          <section className="rounded-2xl border border-slate-700/80 bg-slate-900/90 p-4 shadow-2xl">
            <header className="mb-3">
              <h1 className="text-xl md:text-2xl font-bold">Анализ походки по видео</h1>
              <p className="mt-1 text-xs md:text-sm text-slate-400">
                Ближняя нога, heel-to-toe стопа, фазовая разметка и флаги по зонам
              </p>
            </header>

            <div className="mb-3 flex flex-wrap gap-2">
              <button onClick={onBackToFrames} className="rounded-xl bg-blue-600 px-3 py-2 text-sm font-semibold hover:bg-blue-500">
                ← К выбору кадров
              </button>
              <div className="rounded-xl bg-emerald-600 px-3 py-2 text-sm font-semibold">
                Кадров: {results.length}
              </div>
            </div>

            <div className="relative overflow-hidden rounded-xl border border-slate-700 bg-black">
              <canvas ref={canvasRef} className="block w-full h-auto" />
              {!currentMetrics && (
                <div className="flex aspect-[9/16] items-center justify-center text-slate-400">
                  Выбери кадр справа
                </div>
              )}

              {currentFrame && (
                <div className="absolute left-3 top-3 max-w-[72%] rounded-xl bg-slate-950/80 p-3 text-xs backdrop-blur">
                  <div className="font-bold">Фаза: {currentPhase?.label}</div>
                  <div>Кадр: {currentFrame.id + 1}</div>
                  <div>Колено: {currentMetrics?.leftKnee ?? "—"}° / {currentMetrics?.rightKnee ?? "—"}°</div>
                  <div>Стопа: {currentMetrics?.leftFootAngle ?? "—"}° / {currentMetrics?.rightFootAngle ?? "—"}°</div>
                </div>
              )}

              {currentAnalysis.score > 0 && (
                <div className="absolute right-3 top-3 rounded bg-rose-500 px-3 py-2 text-xs font-bold text-white">
                  Есть зоны внимания
                </div>
              )}
            </div>
          </section>

          <aside className="space-y-3">
            <div className={`rounded-2xl border p-3 ${currentAnalysis.score > 0 ? "border-rose-200 bg-rose-100 text-slate-950" : "border-emerald-700 bg-emerald-950/30"}`}>
              <div className="text-sm font-bold">
                {currentAnalysis.verdict}
              </div>
              <div className="mt-1 text-xs opacity-80">
                confidence ≈ {currentMetrics?.confidence ?? "—"}% · attention score: {currentAnalysis.score}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700/80 bg-slate-900/90 p-4">
              <h2 className="mb-3 font-semibold">Кадры анализа</h2>
              <div className="grid grid-cols-2 gap-2">
                {results.map((r, index) => (
                  <button
                    key={r.id}
                    onClick={() => onSetCurrentResult(r)}
                    className={`rounded-xl border p-3 text-left text-xs transition ${
                      currentFrame?.id === r.id
                        ? "border-white bg-slate-800"
                        : "border-slate-700 bg-slate-950 hover:bg-slate-800"
                    }`}
                  >
                    <div className="font-bold">Кадр {index + 1}</div>
                    <div className="mt-1 text-slate-400">{r.analysis.phase?.label}</div>
                    <div className="text-slate-500">score {r.analysis.score}</div>
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700/80 bg-slate-900/90 p-4">
              <h2 className="text-lg font-bold">{currentPhase?.label}</h2>
              <p className="mt-2 text-sm text-slate-300">
                Этот кадр не называется “патологичным целиком”. Смотрим зоны: корпус, таз, колено, стопа.
              </p>

              <select
                value={currentFrame?.phaseId ?? DEFAULT_PHASE}
                disabled={!currentFrame}
                onChange={(e) => onUpdateFramePhase(currentFrame.id, e.target.value)}
                className="mt-3 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white disabled:opacity-50"
              >
                {GAIT_PHASES.map((phase) => (
                  <option key={phase.id} value={phase.id}>{phase.label}</option>
                ))}
              </select>

              <div className="mt-4 space-y-2 text-sm text-slate-200">
                <div><b>Фокус:</b> {currentPhase?.focus}</div>
                <div><b>Корпус:</b> наклон от вертикали ≈ {currentMetrics?.torsoTilt === null || currentMetrics?.torsoTilt === undefined ? "—" : Math.abs(90 - Math.abs(currentMetrics.torsoTilt))}°</div>
                <div><b>Таз:</b> видео {currentMetrics?.pelvisTilt ?? "—"}°, ориентир ближе к 0°</div>
                <div><b>Колено:</b> видео {currentMetrics?.leftKnee ?? "—"}° / {currentMetrics?.rightKnee ?? "—"}°, норма {currentPhase?.norms.knee.min}–{currentPhase?.norms.knee.max}°</div>
                <div><b>Голеностоп:</b> видео {currentMetrics?.leftAnkle ?? "—"}° / {currentMetrics?.rightAnkle ?? "—"}°, норма {currentPhase?.norms.ankle.min}–{currentPhase?.norms.ankle.max}°</div>
              </div>

              <div className="mt-4 rounded-xl bg-slate-950 p-3 text-sm">
                <div className="font-semibold">Зоны внимания:</div>
                <ul className="mt-2 space-y-1 text-slate-300">
                  {currentAnalysis.zones?.map((zone, i) => (
                    <li key={i}>• <b>{zone.area}:</b> {zone.note}</li>
                  ))}
                </ul>
              </div>

              <div className="mt-3 rounded-xl bg-slate-950 p-3 text-sm">
                <div className="font-semibold">Подсказки:</div>
                <ul className="mt-2 space-y-1 text-slate-300">
                  {currentAnalysis.flags.map((flag, i) => <li key={i}>• {flag}</li>)}
                </ul>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700/80 bg-slate-900/90 p-4 text-sm text-slate-300">
              <div className="font-semibold text-white">Сводка</div>
              <div className="mt-2">Кадры с зонами внимания: {attentionCount}/{results.length}</div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}

export default function FullSkeletonGaitAnalyzer() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);

  const [step, setStep] = useState(1);
  const [status, setStatus] = useState("Загружается модель MediaPipe…");
  const [videoUrl, setVideoUrl] = useState(null);
  const [frames, setFrames] = useState([]);
  const [results, setResults] = useState([]);
  const [selectedFrameId, setSelectedFrameId] = useState(null);
  const [currentResultId, setCurrentResultId] = useState(null);
  const [currentMetrics, setCurrentMetrics] = useState(null);
  const [isBusy, setIsBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "IMAGE",
          numPoses: 1,
          minPoseDetectionConfidence: 0.45,
          minPosePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setStatus("Модель готова. Загрузите видео.");
        }
      } catch (error) {
        console.error(error);
        setStatus("Ошибка загрузки MediaPipe. Проверь @mediapipe/tasks-vision.");
      }
    }

    init();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
    };
  }, []);

  function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (videoUrl) URL.revokeObjectURL(videoUrl);
    const url = URL.createObjectURL(file);

    setVideoUrl(url);
    setFrames([]);
    setResults([]);
    setSelectedFrameId(null);
    setCurrentResultId(null);
    setCurrentMetrics(null);
    setStep(2);
    setStatus("Видео загружено. Нажми “Нарезать кадры”.");
  }

  async function extractFrames() {
    const video = videoRef.current;
    if (!video || !videoUrl) return;

    setIsBusy(true);
    setStatus("Нарезаю видео на кадры…");

    await new Promise((resolve) => {
      if (video.readyState >= 2) resolve();
      else video.onloadedmetadata = resolve;
    });

    const duration = video.duration || 1;
    const count = Math.min(40, Math.max(12, Math.floor(duration * 8)));
    const temp = document.createElement("canvas");
    const ctx = temp.getContext("2d");

    temp.width = video.videoWidth;
    temp.height = video.videoHeight;

    const nextFrames = [];

    for (let i = 0; i < count; i += 1) {
      const time = Math.min(duration - 0.05, (duration / count) * i);
      video.currentTime = time;
      await new Promise((resolve) => {
        video.onseeked = resolve;
      });

      ctx.drawImage(video, 0, 0, temp.width, temp.height);
      nextFrames.push({
        id: i,
        time,
        dataUrl: temp.toDataURL("image/jpeg", 0.88),
        selected: false,
        confidence: null,
        phaseId: DEFAULT_PHASE,
      });
    }

    setFrames(nextFrames);
    setResults([]);
    setSelectedFrameId(null);
    setCurrentResultId(null);
    setCurrentMetrics(null);
    setIsBusy(false);
    setStatus("Кадры готовы. Выбирай кадры кликом по картинкам.");
  }

  async function detectFrame(frame) {
    const landmarker = landmarkerRef.current;
    if (!landmarker || !frame) return null;

    const img = new Image();
    img.src = frame.dataUrl;
    await img.decode();

    const detected = landmarker.detect(img);
    return { img, metrics: drawSkeleton(canvasRef.current ?? document.createElement("canvas"), img, detected.landmarks?.[0]) };
  }

  async function toggleFrame(frame) {
    const nextSelected = !frame.selected;
    setFrames((prev) => prev.map((f) => (f.id === frame.id ? { ...f, selected: nextSelected } : f)));
    setSelectedFrameId(frame.id);

    if (frame.confidence === null) {
      const detected = await detectFrame(frame);
      const confidence = detected?.metrics?.confidence ?? 0;
      setFrames((prev) => prev.map((f) => (f.id === frame.id ? { ...f, confidence } : f)));
    }
  }

  async function analyzeSelectedFrames() {
    const selected = frames.filter((f) => f.selected);
    if (!selected.length) {
      setStatus("Сначала выбери кадры кликом по картинкам.");
      return;
    }

    setIsBusy(true);
    setStatus(`Анализирую выбранные кадры: ${selected.length} шт.`);

    const nextResults = [];
    for (const frame of selected) {
      const detected = await detectFrame(frame);
      const metrics = detected?.metrics ?? null;
      const analysis = classifyFrame(metrics, frame.phaseId ?? DEFAULT_PHASE);
      nextResults.push({ ...frame, metrics, analysis });
    }

    setResults(nextResults);
    setIsBusy(false);
    setStep(3);
    setStatus("Готово. Смотри выбранные кадры, норму/патологию, градусы и подсказки.");

    if (nextResults[0]) {
      await setCurrentResult(nextResults[0]);
    }
  }

  async function setCurrentResult(result) {
    if (!result) return;

    setCurrentResultId(result.id);
    setCurrentMetrics(result.metrics);

    const img = new Image();
    img.src = result.dataUrl;
    await img.decode();

    const landmarker = landmarkerRef.current;
    const detected = landmarker.detect(img);
    drawSkeleton(canvasRef.current, img, detected.landmarks?.[0]);
  }

  function updateFramePhase(frameId, phaseId) {
    setFrames((prev) => prev.map((f) => (f.id === frameId ? { ...f, phaseId } : f)));
    setResults((prev) =>
      prev.map((r) => {
        if (r.id !== frameId) return r;
        const analysis = classifyFrame(r.metrics, phaseId);
        return { ...r, phaseId, analysis };
      })
    );
    setStatus(`Фаза кадра обновлена: ${getPhaseById(phaseId)?.label}.`);
  }

  const selectedCount = frames.filter((f) => f.selected).length;
  const currentFrame = results.find((r) => r.id === currentResultId) ?? results[0] ?? null;
  const currentPhase = getPhaseById(currentFrame?.phaseId ?? DEFAULT_PHASE);
  const currentAnalysis = currentFrame?.analysis ?? classifyFrame(currentMetrics, currentFrame?.phaseId ?? DEFAULT_PHASE);

  if (step === 1) {
    return <UploadScreen status={status} onUpload={handleUpload} />;
  }

  if (step === 2) {
    return (
      <FrameSelectionScreen
        status={status}
        videoUrl={videoUrl}
        videoRef={videoRef}
        frames={frames}
        selectedCount={selectedCount}
        selectedFrameId={selectedFrameId}
        isBusy={isBusy}
        onExtractFrames={extractFrames}
        onToggleFrame={toggleFrame}
        onAnalyzeSelected={analyzeSelectedFrames}
        onBack={() => setStep(1)}
      />
    );
  }

  return (
    <AnalysisScreen
      status={status}
      canvasRef={canvasRef}
      results={results}
      currentMetrics={currentMetrics}
      currentFrame={currentFrame}
      currentPhase={currentPhase}
      currentAnalysis={currentAnalysis}
      onSetCurrentResult={setCurrentResult}
      onUpdateFramePhase={updateFramePhase}
      onBackToFrames={() => setStep(2)}
    />
  );
}
