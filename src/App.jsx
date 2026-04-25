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

const SKELETON_CONNECTIONS = [
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
  center: "#ffffff",
  weak: "rgba(255,255,255,0.28)",
};

function midpoint(a, b, name) {
  if (!a || !b) return null;
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
    name,
  };
}

function getColorForConnection(a, b) {
  if (a.includes("Shoulder") || b.includes("Shoulder") || a.includes("Hip") || b.includes("Hip") || a.startsWith("mid") || b.startsWith("mid")) return COLORS.trunk;
  if (a.startsWith("left") && (a.includes("Elbow") || a.includes("Wrist") || b.includes("Elbow") || b.includes("Wrist"))) return COLORS.leftArm;
  if (a.startsWith("right") && (a.includes("Elbow") || a.includes("Wrist") || b.includes("Elbow") || b.includes("Wrist"))) return COLORS.rightArm;
  if (a.startsWith("left") || b.startsWith("left")) return COLORS.leftLeg;
  if (a.startsWith("right") || b.startsWith("right")) return COLORS.rightLeg;
  return COLORS.head;
}

function drawLine(ctx, a, b, color, scale) {
  if (!a || !b) return;
  const conf = Math.min(a.visibility ?? 1, b.visibility ?? 1);
  ctx.save();
  ctx.strokeStyle = conf < 0.45 ? COLORS.weak : color;
  ctx.lineWidth = Math.max(2, 4 * scale);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function drawPoint(ctx, p, color, scale) {
  if (!p) return;
  ctx.save();
  ctx.fillStyle = (p.visibility ?? 1) < 0.45 ? COLORS.weak : color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, Math.max(4, 6 * scale), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
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

function drawSkeletonOnCanvas(canvas, image, landmarks) {
  const ctx = canvas.getContext("2d");
  const width = image.naturalWidth || image.videoWidth || image.width;
  const height = image.naturalHeight || image.videoHeight || image.height;

  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(image, 0, 0, width, height);

  if (!landmarks?.length) return null;

  const points = buildPoints(landmarks, width, height);
  const scale = Math.max(0.7, width / 900);

  SKELETON_CONNECTIONS.forEach(([a, b]) => {
    drawLine(ctx, points[a], points[b], getColorForConnection(a, b), scale);
  });

  Object.keys(LANDMARKS).forEach((name) => {
    const color = name === "nose"
      ? COLORS.head
      : name.includes("Shoulder") || name.includes("Hip")
        ? COLORS.trunk
        : name.startsWith("left") && (name.includes("Elbow") || name.includes("Wrist"))
          ? COLORS.leftArm
          : name.startsWith("right") && (name.includes("Elbow") || name.includes("Wrist"))
            ? COLORS.rightArm
            : name.startsWith("left")
              ? COLORS.leftLeg
              : COLORS.rightLeg;
    drawPoint(ctx, points[name], color, scale);
  });

  drawPoint(ctx, points.midShoulder, COLORS.center, scale);
  drawPoint(ctx, points.midHip, COLORS.center, scale);

  return {
    leftKnee: angleAt(points.leftHip, points.leftKnee, points.leftAnkle),
    rightKnee: angleAt(points.rightHip, points.rightKnee, points.rightAnkle),
    leftAnkle: angleAt(points.leftKnee, points.leftAnkle, points.leftFootIndex),
    rightAnkle: angleAt(points.rightKnee, points.rightAnkle, points.rightFootIndex),
    leftElbow: angleAt(points.leftShoulder, points.leftElbow, points.leftWrist),
    rightElbow: angleAt(points.rightShoulder, points.rightElbow, points.rightWrist),
    confidence: Math.round(
      (Object.keys(LANDMARKS).reduce((sum, key) => sum + (points[key]?.visibility ?? 0), 0) /
        Object.keys(LANDMARKS).length) *
        100
    ),
  };
}

function Metric({ label, value }) {
  return (
    <div className="rounded-xl bg-slate-950 p-3">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="text-base font-semibold">{value ?? "—"}</div>
    </div>
  );
}

export default function FullSkeletonGaitAnalyzer() {
  const hiddenVideoRef = useRef(null);
  const mainCanvasRef = useRef(null);
  const landmarkerRef = useRef(null);

  const [status, setStatus] = useState("Загружаю модель…");
  const [videoUrl, setVideoUrl] = useState(null);
  const [frames, setFrames] = useState([]);
  const [selectedFrameId, setSelectedFrameId] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isExtracting, setIsExtracting] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "GPU",
          },
          runningMode: "IMAGE",
          numPoses: 1,
          minPoseDetectionConfidence: 0.45,
          minPosePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });

        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setStatus("Модель готова. Загрузи видео, потом нарежь кадры.");
        }
      } catch (err) {
        console.error(err);
        setStatus("Ошибка загрузки MediaPipe. Проверь @mediapipe/tasks-vision.");
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
    setSelectedFrameId(null);
    setMetrics(null);
    setStatus("Видео загружено. Теперь нажми “Нарезать кадры”.");
  }

  async function extractFrames() {
    const video = hiddenVideoRef.current;
    if (!video || !videoUrl) return;

    setIsExtracting(true);
    setStatus("Нарезаю видео на кадры…");

    await new Promise((resolve) => {
      if (video.readyState >= 2) resolve();
      else video.onloadedmetadata = resolve;
    });

    const duration = video.duration;
    const count = Math.min(36, Math.max(12, Math.floor(duration * 8))); // примерно 8 кадров/сек, но без безумия
    const step = duration / count;
    const tempCanvas = document.createElement("canvas");
    const ctx = tempCanvas.getContext("2d");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;

    const nextFrames = [];

    for (let i = 0; i < count; i += 1) {
      const time = Math.min(duration - 0.05, i * step);
      video.currentTime = time;
      await new Promise((resolve) => {
        video.onseeked = resolve;
      });

      ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
      const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.86);
      nextFrames.push({ id: i, time, dataUrl, selected: false, analyzed: false, confidence: null });
    }

    setFrames(nextFrames);
    setSelectedFrameId(nextFrames[0]?.id ?? null);
    setIsExtracting(false);
    setStatus("Кадры готовы. Выбери кадры, где человек полностью виден сбоку.");

    if (nextFrames[0]) analyzeFrame(nextFrames[0]);
  }

  function toggleFrame(id) {
    setFrames((prev) => prev.map((f) => (f.id === id ? { ...f, selected: !f.selected } : f)));
  }

  async function analyzeFrame(frame) {
    const landmarker = landmarkerRef.current;
    const canvas = mainCanvasRef.current;
    if (!landmarker || !canvas || !frame) return;

    setSelectedFrameId(frame.id);
    setStatus(`Размечаю кадр ${frame.id + 1}…`);

    const img = new Image();
    img.src = frame.dataUrl;
    await img.decode();

    const result = landmarker.detect(img);
    const landmarks = result.landmarks?.[0];
    const nextMetrics = drawSkeletonOnCanvas(canvas, img, landmarks);

    if (!landmarks || !nextMetrics) {
      setMetrics(null);
      setFrames((prev) => prev.map((f) => (f.id === frame.id ? { ...f, analyzed: true, confidence: 0 } : f)));
      setStatus("На этом кадре тело не найдено. Не выбирай его для анализа.");
      return;
    }

    setMetrics(nextMetrics);
    setFrames((prev) =>
      prev.map((f) =>
        f.id === frame.id ? { ...f, analyzed: true, confidence: nextMetrics.confidence } : f
      )
    );
    setStatus(`Кадр размечен. Confidence: ${nextMetrics.confidence}%.`);
  }

  function selectAllGoodFrames() {
    setFrames((prev) =>
      prev.map((f) => ({
        ...f,
        selected: (f.confidence ?? 0) >= 55,
      }))
    );
  }

  const selectedFrame = frames.find((f) => f.id === selectedFrameId);
  const selectedCount = frames.filter((f) => f.selected).length;

  return (
    <div className="min-h-screen bg-slate-950 p-4 md:p-8 text-white">
      <div className="mx-auto max-w-7xl space-y-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold">Анализ походки по выбранным кадрам</h1>
          <p className="mt-1 text-slate-300">
            Сначала режем видео на картинки, потом размечаем только хорошие кадры. Так скелет меньше “пролетает”.
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <label className="cursor-pointer rounded-2xl bg-blue-600 px-4 py-2 font-semibold shadow hover:bg-blue-500">
            Загрузить видео
            <input type="file" accept="video/*" className="hidden" onChange={handleUpload} />
          </label>

          <button
            type="button"
            disabled={!videoUrl || isExtracting}
            onClick={extractFrames}
            className="rounded-2xl bg-emerald-600 px-4 py-2 font-semibold shadow hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isExtracting ? "Нарезаю…" : "Нарезать кадры"}
          </button>

          <button
            type="button"
            disabled={!frames.length}
            onClick={selectAllGoodFrames}
            className="rounded-2xl bg-slate-800 px-4 py-2 font-semibold shadow hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Выбрать хорошие автоматически
          </button>
        </div>

        <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3 text-sm text-slate-200">
          {status}
        </div>

        <video ref={hiddenVideoRef} src={videoUrl ?? undefined} className="hidden" muted playsInline />

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-4">
          <div className="space-y-4">
            <div className="overflow-hidden rounded-2xl border border-slate-700 bg-black">
              {frames.length ? (
                <canvas ref={mainCanvasRef} className="block w-full h-auto" />
              ) : (
                <div className="flex aspect-video items-center justify-center text-slate-400">
                  Здесь появится выбранный кадр с разметкой скелета
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <h2 className="font-semibold">Раскадровка</h2>
                <div className="text-sm text-slate-300">Выбрано: {selectedCount} / {frames.length}</div>
              </div>

              {frames.length ? (
                <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-7 lg:grid-cols-9 gap-3">
                  {frames.map((frame) => (
                    <button
                      key={frame.id}
                      type="button"
                      onClick={() => analyzeFrame(frame)}
                      className={`relative overflow-hidden rounded-xl border-2 bg-slate-950 text-left ${
                        frame.id === selectedFrameId ? "border-blue-500" : frame.selected ? "border-emerald-500" : "border-slate-700"
                      }`}
                    >
                      <img src={frame.dataUrl} alt={`Кадр ${frame.id + 1}`} className="aspect-video w-full object-cover" />
                      <div className="absolute left-1 top-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px]">
                        {frame.id + 1}
                      </div>
                      <div className="absolute right-1 top-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px]">
                        {frame.confidence === null ? "—" : `${frame.confidence}%`}
                      </div>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleFrame(frame.id);
                        }}
                        className={`absolute bottom-1 right-1 rounded-full px-2 py-0.5 text-xs font-bold ${
                          frame.selected ? "bg-emerald-500 text-white" : "bg-black/70 text-slate-200"
                        }`}
                      >
                        {frame.selected ? "✓" : "+"}
                      </button>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-slate-400">
                  Загрузи видео и нажми “Нарезать кадры”.
                </div>
              )}
            </div>
          </div>

          <aside className="space-y-4">
            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <h2 className="mb-3 text-lg font-semibold">Кадр</h2>
              <div className="text-sm text-slate-300">
                {selectedFrame ? `Кадр ${selectedFrame.id + 1}, ${selectedFrame.time.toFixed(2)} сек` : "Кадр не выбран"}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
              <h2 className="mb-3 text-lg font-semibold">Метрики скелета</h2>
              <div className="grid grid-cols-2 gap-2">
                <Metric label="Confidence" value={metrics ? `${metrics.confidence}%` : null} />
                <Metric label="L колено" value={metrics?.leftKnee ? `${metrics.leftKnee}°` : null} />
                <Metric label="R колено" value={metrics?.rightKnee ? `${metrics.rightKnee}°` : null} />
                <Metric label="L голеностоп" value={metrics?.leftAnkle ? `${metrics.leftAnkle}°` : null} />
                <Metric label="R голеностоп" value={metrics?.rightAnkle ? `${metrics.rightAnkle}°` : null} />
                <Metric label="L локоть" value={metrics?.leftElbow ? `${metrics.leftElbow}°` : null} />
                <Metric label="R локоть" value={metrics?.rightElbow ? `${metrics.rightElbow}°` : null} />
              </div>
            </div>

            <div className="rounded-2xl border border-blue-700/60 bg-blue-950/40 p-4 text-sm text-blue-100">
              <h2 className="mb-2 font-semibold">Как выбирать кадры</h2>
              <p>Оставляй кадры, где человек целиком в кадре, видны стопы, нет сильного смаза и тело стоит боком к камере.</p>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
