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
    norms: {
      knee: { min: 5, max: 25, label: "колено слегка сгибается" },
      ankle: { min: 80, max: 120, label: "стопа принимает опору" },
    },
  },
  {
    id: "mid_stance",
    label: "Mid stance",
    focus: "опора, перенос тела над стопой",
    norms: {
      knee: { min: 0, max: 15, label: "колено близко к разгибанию" },
      ankle: { min: 85, max: 120, label: "контроль голеностопа" },
    },
  },
  {
    id: "terminal_stance",
    label: "Terminal stance",
    focus: "перекат через стопу, пятка поднимается",
    norms: {
      knee: { min: 0, max: 20, label: "колено почти разогнуто" },
      ankle: { min: 80, max: 125, label: "стопа уходит в перекат" },
    },
  },
  {
    id: "pre_swing",
    label: "Pre-swing",
    focus: "отрыв, подготовка к переносу",
    norms: {
      knee: { min: 20, max: 50, label: "начинается сгибание колена" },
      ankle: { min: 70, max: 120, label: "отталкивание/отрыв" },
    },
  },
  {
    id: "initial_swing",
    label: "Initial swing",
    focus: "перенос, clearance стопы",
    norms: {
      knee: { min: 40, max: 75, label: "колено сгибается для clearance" },
      ankle: { min: 75, max: 125, label: "стопа не должна цеплять пол" },
    },
  },
  {
    id: "mid_swing",
    label: "Mid swing",
    focus: "перенос ноги вперёд",
    norms: {
      knee: { min: 25, max: 65, label: "колено начинает разгибаться" },
      ankle: { min: 80, max: 125, label: "стопа удерживается от провиса" },
    },
  },
  {
    id: "terminal_swing",
    label: "Terminal swing",
    focus: "подготовка к контакту пяткой",
    norms: {
      knee: { min: 0, max: 25, label: "колено разгибается" },
      ankle: { min: 80, max: 125, label: "стопа готовится к контакту" },
    },
  },
];

const DEFAULT_PHASE = "mid_swing";

const NORMS = {
  confidence: { min: 55, max: 100, label: "качество распознавания" },
};

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
  return Math.round((Math.acos(Math.max(-1, Math.min(1, dot / (l1 * l2)))) * 180) / Math.PI);
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
    points[name] = { x: lm.x * width, y: lm.y * height, visibility: lm.visibility ?? 1, name };
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
      Object.keys(LANDMARKS).length) * 100
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
    };
  }

  const flags = [];
  const deviations = [];
  let score = 0;

  const leftKneeBad = isOutsideNorm(metrics.leftKnee, phase.norms.knee);
  const rightKneeBad = isOutsideNorm(metrics.rightKnee, phase.norms.knee);
  const leftAnkleBad = isOutsideNorm(metrics.leftAnkle, phase.norms.ankle);
  const rightAnkleBad = isOutsideNorm(metrics.rightAnkle, phase.norms.ankle);

  if (metrics.confidence < NORMS.confidence.min) {
    flags.push("низкое качество распознавания: кадр лучше не использовать для вывода");
    deviations.push({ label: "Confidence", value: `${metrics.confidence}%`, norm: `≥${NORMS.confidence.min}%`, comment: "низкая уверенность модели" });
    score += 2;
  }

  if (leftKneeBad) {
    const comment = describeNorm(metrics.leftKnee, phase.norms.knee);
    flags.push(`левое колено: ${comment} для фазы ${phase.label}`);
    deviations.push({ label: "L колено", value: `${metrics.leftKnee}°`, norm: `${phase.norms.knee.min}–${phase.norms.knee.max}°`, comment });
    score += 1;
  }

  if (rightKneeBad) {
    const comment = describeNorm(metrics.rightKnee, phase.norms.knee);
    flags.push(`правое колено: ${comment} для фазы ${phase.label}`);
    deviations.push({ label: "R колено", value: `${metrics.rightKnee}°`, norm: `${phase.norms.knee.min}–${phase.norms.knee.max}°`, comment });
    score += 1;
  }

  if (leftAnkleBad) {
    const comment = describeNorm(metrics.leftAnkle, phase.norms.ankle);
    flags.push(`левый голеностоп/стопа: ${comment} для фазы ${phase.label}`);
    deviations.push({ label: "L голеностоп", value: `${metrics.leftAnkle}°`, norm: `${phase.norms.ankle.min}–${phase.norms.ankle.max}°`, comment });
    score += 1;
  }

  if (rightAnkleBad) {
    const comment = describeNorm(metrics.rightAnkle, phase.norms.ankle);
    flags.push(`правый голеностоп/стопа: ${comment} для фазы ${phase.label}`);
    deviations.push({ label: "R голеностоп", value: `${metrics.rightAnkle}°`, norm: `${phase.norms.ankle.min}–${phase.norms.ankle.max}°`, comment });
    score += 1;
  }

  if (Math.abs(metrics.pelvisTilt ?? 0) > 12) {
    flags.push("таз заметно наклонён относительно кадра: проверь, это реальный перекос или наклон камеры");
    deviations.push({ label: "Таз", value: `${metrics.pelvisTilt}°`, norm: "ближе к 0°", comment: "перекос/наклон кадра" });
    score += 1;
  }

  if (!flags.length) flags.push(`для выбранной фазы ${phase.label} грубых отклонений по этому кадру не видно`);

  return {
    score,
    verdict: score >= 4 ? "патологичность/плохой кадр" : score >= 1 ? "есть отклонения" : "ближе к норме",
    flags,
    phase,
    deviations,
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

export default function FullSkeletonGaitAnalyzer() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);

  const [step, setStep] = useState(1);
  const [status, setStatus] = useState("Загружаю модель…");
  const [videoUrl, setVideoUrl] = useState(null);
  const [frames, setFrames] = useState([]);
  const [selectedFrameId, setSelectedFrameId] = useState(null);
  const [currentMetrics, setCurrentMetrics] = useState(null);
  const [results, setResults] = useState([]);
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
          setStatus("Модель готова. Шаг 1: загрузи видео.");
        }
      } catch (e) {
        console.error(e);
        setStatus("Ошибка загрузки MediaPipe / @mediapipe/tasks-vision.");
      }
    }
    init();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
    };
  }, []);

  function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setFrames([]);
    setResults([]);
    setCurrentMetrics(null);
    setSelectedFrameId(null);
    setStep(2);
    setStatus("Видео загружено. Шаг 2: нарежь кадры и выбери хорошие кликом по картинкам.");
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

    const next = [];
    for (let i = 0; i < count; i += 1) {
      video.currentTime = Math.min(duration - 0.05, (duration / count) * i);
      await new Promise((resolve) => (video.onseeked = resolve));
      ctx.drawImage(video, 0, 0, temp.width, temp.height);
      next.push({
        id: i,
        time: video.currentTime,
        dataUrl: temp.toDataURL("image/jpeg", 0.88),
        selected: false,
        confidence: null,
        phaseId: DEFAULT_PHASE,
      });
    }

    setFrames(next);
    setSelectedFrameId(next[0]?.id ?? null);
    setIsBusy(false);
    setStatus("Кадры готовы. Выбирай нужные кадры кликом по миниатюре, затем нажми “Анализировать выбранные”.");
    if (next[0]) previewFrame(next[0]);
  }

  async function previewFrame(frame) {
    const landmarker = landmarkerRef.current;
    const canvas = canvasRef.current;
    if (!landmarker || !canvas || !frame) return;

    setSelectedFrameId(frame.id);
    const img = new Image();
    img.src = frame.dataUrl;
    await img.decode();
    const detected = landmarker.detect(img);
    const metrics = drawSkeleton(canvas, img, detected.landmarks?.[0]);
    setCurrentMetrics(metrics);

    const confidence = metrics?.confidence ?? 0;
    setFrames((prev) => prev.map((f) => (f.id === frame.id ? { ...f, confidence } : f)));
  }

  function toggleFrame(frame) {
    setFrames((prev) => prev.map((f) => (f.id === frame.id ? { ...f, selected: !f.selected } : f)));
    previewFrame(frame);
  }

  function updateFramePhase(frameId, phaseId) {
    setFrames((prev) => prev.map((f) => (f.id === frameId ? { ...f, phaseId } : f)));

    if (selectedFrameId === frameId) {
      const nextAnalysis = classifyFrame(currentMetrics, phaseId);
      setStatus(`Фаза кадра обновлена: ${nextAnalysis.phase.label}. Теперь норма считается именно для неё.`);
    }
  }

  async function analyzeSelectedFrames() {
    const selected = frames.filter((f) => f.selected);
    if (!selected.length) {
      setStatus("Сначала выбери кадры кликом по картинкам.");
      return;
    }

    setIsBusy(true);
    setStep(3);
    setResults([]);
    setStatus(`Анализирую ${selected.length} выбранных кадров…`);

    const nextResults = [];
    for (const frame of selected) {
      await previewFrame(frame);
      const canvas = canvasRef.current;
      const landmarker = landmarkerRef.current;
      const img = new Image();
      img.src = frame.dataUrl;
      await img.decode();
      const detected = landmarker.detect(img);
      const metrics = drawSkeleton(canvas, img, detected.landmarks?.[0]);
      const analysis = classifyFrame(metrics, frame.phaseId ?? DEFAULT_PHASE);
      nextResults.push({ ...frame, metrics, analysis });
      setResults([...nextResults]);
      await new Promise((resolve) => setTimeout(resolve, 60));
    }

    setCurrentMetrics(nextResults[0]?.metrics ?? null);
    setSelectedFrameId(nextResults[0]?.id ?? null);
    setIsBusy(false);
    setStatus("Готово. Шаг 3: смотри норму/патологию, градусы и подсказки по выбранным кадрам.");
  }

  function selectAllVisible() {
    setFrames((prev) => prev.map((f) => ({ ...f, selected: (f.confidence ?? 0) >= 55 })));
  }

  const selectedCount = frames.filter((f) => f.selected).length;
  const currentFrame = frames.find((f) => f.id === selectedFrameId);
  const currentPhaseId = currentFrame?.phaseId ?? DEFAULT_PHASE;
  const currentPhase = getPhaseById(currentPhaseId);
  const currentAnalysis = classifyFrame(currentMetrics, currentPhaseId);
  const pathologyCount = results.filter((r) => r.analysis.score >= 3).length;
  const deviationCount = results.filter((r) => r.analysis.score > 0).length;

  return (
    <div className="min-h-screen bg-slate-950 p-4 md:p-8 text-white">
      <div className="mx-auto max-w-7xl space-y-5">
        <header>
          <h1 className="text-2xl md:text-3xl font-bold">Анализ походки</h1>
          <p className="mt-1 text-slate-300">UX: загрузка → выбор кадров → анализ выбранных кадров с нормой/патологией и подсказками.</p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className={`rounded-2xl border p-4 ${step === 1 ? "border-blue-500 bg-blue-950/40" : "border-slate-700 bg-slate-900"}`}>
            <div className="text-sm text-slate-400">1. Основной экран</div>
            <div className="mt-2 font-semibold">Загрузить видео</div>
            <label className="mt-3 inline-block cursor-pointer rounded-xl bg-blue-600 px-4 py-2 font-semibold hover:bg-blue-500">
              Загрузить видео
              <input type="file" accept="video/*" className="hidden" onChange={handleUpload} />
            </label>
          </div>

          <div className={`rounded-2xl border p-4 ${step === 2 ? "border-blue-500 bg-blue-950/40" : "border-slate-700 bg-slate-900"}`}>
            <div className="text-sm text-slate-400">2. Выбор кадров</div>
            <div className="mt-2 font-semibold">Выбрать кадры кликом</div>
            <button disabled={!videoUrl || isBusy} onClick={extractFrames} className="mt-3 rounded-xl bg-emerald-600 px-4 py-2 font-semibold hover:bg-emerald-500 disabled:opacity-50">
              {isBusy ? "Жди…" : "Нарезать кадры"}
            </button>
          </div>

          <div className={`rounded-2xl border p-4 ${step === 3 ? "border-blue-500 bg-blue-950/40" : "border-slate-700 bg-slate-900"}`}>
            <div className="text-sm text-slate-400">3. Анализ</div>
            <div className="mt-2 font-semibold">Норма / патология</div>
            <button disabled={!selectedCount || isBusy} onClick={analyzeSelectedFrames} className="mt-3 rounded-xl bg-fuchsia-600 px-4 py-2 font-semibold hover:bg-fuchsia-500 disabled:opacity-50">
              Анализировать выбранные
            </button>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3 text-sm text-slate-200">{status}</div>
        <video ref={videoRef} src={videoUrl ?? undefined} muted playsInline className="hidden" />

        {step === 2 && (
          <section className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-xl font-semibold">Выбор кадров</h2>
                <p className="text-sm text-slate-400">Клик по картинке = выбрать/снять. Мини-кнопок больше нет.</p>
              </div>
              <div className="flex gap-2">
                <button disabled={!frames.length} onClick={selectAllVisible} className="rounded-xl bg-slate-800 px-3 py-2 text-sm font-semibold hover:bg-slate-700 disabled:opacity-50">Выбрать качественные</button>
                <button disabled={!selectedCount} onClick={analyzeSelectedFrames} className="rounded-xl bg-fuchsia-600 px-3 py-2 text-sm font-semibold hover:bg-fuchsia-500 disabled:opacity-50">Анализировать</button>
              </div>
            </div>

            {frames.length ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
                {frames.map((frame) => (
                  <button key={frame.id} onClick={() => toggleFrame(frame)} className={`relative overflow-hidden rounded-xl border-2 ${frame.selected ? "border-emerald-500" : selectedFrameId === frame.id ? "border-blue-500" : "border-slate-700"}`}>
                    <img src={frame.dataUrl} alt={`Кадр ${frame.id + 1}`} className="aspect-video w-full object-cover" />
                    <div className="absolute left-1 top-1 rounded bg-black/75 px-1.5 py-0.5 text-[10px]">{frame.id + 1}</div>
                    <div className="absolute right-1 top-1 rounded bg-black/75 px-1.5 py-0.5 text-[10px]">{frame.confidence === null ? "—" : `${frame.confidence}%`}</div>
                    <div className="absolute left-1 bottom-6 rounded bg-black/75 px-1.5 py-0.5 text-[10px]">
                      {getPhaseById(frame.phaseId)?.label}
                    </div>
                    <div className={`absolute inset-x-0 bottom-0 py-1 text-center text-xs font-bold ${frame.selected ? "bg-emerald-500" : "bg-black/70"}`}>{frame.selected ? "✓ выбрано" : "клик = выбрать"}</div>
                  </button>
                ))}
              </div>
            ) : (
              <div className="rounded-xl bg-slate-950 p-8 text-center text-slate-400">Нажми “Нарезать кадры”.</div>
            )}
          </section>
        )}

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_420px] gap-4">
          <main className="rounded-2xl border border-slate-700 bg-black overflow-hidden">
            <canvas ref={canvasRef} className="block w-full h-auto" />
            {!currentMetrics && <div className="flex aspect-video items-center justify-center text-slate-400">Здесь будет кадр со скелетом</div>}
          </main>

          <aside className="space-y-4">
            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <h2 className="text-lg font-semibold">Фаза кадра</h2>
              <p className="mt-1 text-sm text-slate-400">Сначала выбери фазу. Норма/патология считается только относительно неё.</p>
              <select
                value={currentPhaseId}
                disabled={selectedFrameId === null}
                onChange={(e) => updateFramePhase(selectedFrameId, e.target.value)}
                className="mt-3 w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white disabled:opacity-50"
              >
                {GAIT_PHASES.map((phase) => (
                  <option key={phase.id} value={phase.id}>{phase.label}</option>
                ))}
              </select>
              <div className="mt-3 rounded-xl bg-slate-950 p-3 text-sm text-slate-300">
                <div className="font-semibold text-white">Фокус: {currentPhase?.focus}</div>
                <div className="mt-1">Колено: {currentPhase?.norms.knee.min}–{currentPhase?.norms.knee.max}°</div>
                <div>Голеностоп: {currentPhase?.norms.ankle.min}–{currentPhase?.norms.ankle.max}°</div>
              </div>
            </div>
            <div className={`rounded-2xl border p-4 ${currentAnalysis.score >= 3 ? "border-rose-600 bg-rose-950/40" : currentAnalysis.score > 0 ? "border-amber-600 bg-amber-950/30" : "border-emerald-700 bg-emerald-950/30"}`}>
              <h2 className="text-lg font-semibold">Норма / патология</h2>
              <div className="mt-2 text-2xl font-bold">{currentAnalysis.verdict}</div>
              <div className="mt-1 text-sm text-slate-300">pathology score: {currentAnalysis.score}</div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <h2 className="mb-3 text-lg font-semibold">Отклонения от нормы: {currentPhase?.label}</h2>
              <div className="grid grid-cols-2 gap-2">
                <Metric label="Confidence" value={currentMetrics ? `${currentMetrics.confidence}%` : null} norm="≥55%" bad={(currentMetrics?.confidence ?? 100) < 55} />
                <Metric label="L колено" value={currentMetrics?.leftKnee ? `${currentMetrics.leftKnee}°` : null} norm={`${currentPhase?.norms.knee.min}–${currentPhase?.norms.knee.max}°`} bad={isOutsideNorm(currentMetrics?.leftKnee, currentPhase?.norms.knee)} />
                <Metric label="R колено" value={currentMetrics?.rightKnee ? `${currentMetrics.rightKnee}°` : null} norm={`${currentPhase?.norms.knee.min}–${currentPhase?.norms.knee.max}°`} bad={isOutsideNorm(currentMetrics?.rightKnee, currentPhase?.norms.knee)} />
                <Metric label="L голеностоп" value={currentMetrics?.leftAnkle ? `${currentMetrics.leftAnkle}°` : null} norm={`${currentPhase?.norms.ankle.min}–${currentPhase?.norms.ankle.max}°`} bad={isOutsideNorm(currentMetrics?.leftAnkle, currentPhase?.norms.ankle)} />
                <Metric label="R голеностоп" value={currentMetrics?.rightAnkle ? `${currentMetrics.rightAnkle}°` : null} norm={`${currentPhase?.norms.ankle.min}–${currentPhase?.norms.ankle.max}°`} bad={isOutsideNorm(currentMetrics?.rightAnkle, currentPhase?.norms.ankle)} />
                <Metric label="Таз" value={currentMetrics?.pelvisTilt ? `${currentMetrics.pelvisTilt}°` : null} norm="ближе к 0°" bad={Math.abs(currentMetrics?.pelvisTilt ?? 0) > 12} />
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <h2 className="mb-3 text-lg font-semibold">Подсказки</h2>
              <ul className="space-y-2 text-sm text-slate-200">
                {currentAnalysis.flags.map((flag, i) => <li key={i}>• {flag}</li>)}
              </ul>
            </div>

            {results.length > 0 && (
              <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
                <h2 className="mb-3 text-lg font-semibold">Все выбранные кадры</h2>
                <div className="mb-3 text-sm text-slate-300">Отклонения: {deviationCount}/{results.length}; патологичность/плохой кадр: {pathologyCount}/{results.length}</div>
                <div className="grid grid-cols-2 gap-2">
                  {results.map((r) => (
                    <button key={r.id} onClick={() => { setCurrentMetrics(r.metrics); setSelectedFrameId(r.id); }} className={`rounded-xl border p-2 text-left text-xs ${r.analysis.score >= 3 ? "border-rose-600 bg-rose-950/40" : r.analysis.score > 0 ? "border-amber-600 bg-amber-950/30" : "border-emerald-700 bg-emerald-950/30"}`}>
                      <div className="font-semibold">Кадр {r.id + 1}</div>
                      <div>{r.analysis.verdict}</div>
                      <div className="text-slate-400">score {r.analysis.score}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
