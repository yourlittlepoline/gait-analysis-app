import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

const LANDMARKS = {
  nose: 0,
  leftEyeInner: 1,
  leftEye: 2,
  leftEyeOuter: 3,
  rightEyeInner: 4,
  rightEye: 5,
  rightEyeOuter: 6,
  leftEar: 7,
  rightEar: 8,
  mouthLeft: 9,
  mouthRight: 10,

  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftPinky: 17,
  rightPinky: 18,
  leftIndex: 19,
  rightIndex: 20,
  leftThumb: 21,
  rightThumb: 22,

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
  // head / face simplified
  ["leftEar", "leftEye"],
  ["leftEye", "nose"],
  ["nose", "rightEye"],
  ["rightEye", "rightEar"],
  ["mouthLeft", "mouthRight"],

  // shoulders / trunk / pelvis
  ["leftShoulder", "rightShoulder"],
  ["leftShoulder", "leftHip"],
  ["rightShoulder", "rightHip"],
  ["leftHip", "rightHip"],

  // left arm
  ["leftShoulder", "leftElbow"],
  ["leftElbow", "leftWrist"],
  ["leftWrist", "leftIndex"],
  ["leftWrist", "leftPinky"],
  ["leftWrist", "leftThumb"],

  // right arm
  ["rightShoulder", "rightElbow"],
  ["rightElbow", "rightWrist"],
  ["rightWrist", "rightIndex"],
  ["rightWrist", "rightPinky"],
  ["rightWrist", "rightThumb"],

  // left leg
  ["leftHip", "leftKnee"],
  ["leftKnee", "leftAnkle"],
  ["leftAnkle", "leftHeel"],
  ["leftHeel", "leftFootIndex"],
  ["leftAnkle", "leftFootIndex"],

  // right leg
  ["rightHip", "rightKnee"],
  ["rightKnee", "rightAnkle"],
  ["rightAnkle", "rightHeel"],
  ["rightHeel", "rightFootIndex"],
  ["rightAnkle", "rightFootIndex"],
];

const COLORS = {
  head: "#facc15",
  trunk: "#38bdf8",
  left: "#22c55e",
  right: "#ef4444",
  center: "#a855f7",
  weak: "rgba(255,255,255,0.25)",
  text: "#ffffff",
};

function getPoint(landmarks, name, width, height) {
  const lm = landmarks[LANDMARKS[name]];
  if (!lm) return null;
  return {
    x: lm.x * width,
    y: lm.y * height,
    z: lm.z ?? 0,
    visibility: lm.visibility ?? 1,
    name,
  };
}

function midpoint(a, b, name = "midpoint") {
  if (!a || !b) return null;
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z ?? 0) + (b.z ?? 0)) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1),
    name,
  };
}

function drawLine(ctx, a, b, color, width = 4) {
  if (!a || !b) return;
  const confidence = Math.min(a.visibility ?? 1, b.visibility ?? 1);
  ctx.save();
  ctx.strokeStyle = confidence < 0.45 ? COLORS.weak : color;
  ctx.lineWidth = confidence < 0.45 ? 2 : width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function drawPoint(ctx, p, color, radius = 5) {
  if (!p) return;
  ctx.save();
  ctx.fillStyle = (p.visibility ?? 1) < 0.45 ? COLORS.weak : color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawLabel(ctx, p, label) {
  if (!p) return;
  ctx.save();
  ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.fillStyle = COLORS.text;
  ctx.fillText(label, p.x + 7, p.y - 7);
  ctx.restore();
}

function vectorAngleDeg(a, b) {
  if (!a || !b) return null;
  const radians = Math.atan2(b.y - a.y, b.x - a.x);
  return (radians * 180) / Math.PI;
}

function jointAngleDeg(a, b, c) {
  if (!a || !b || !c) return null;

  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dot = ab.x * cb.x + ab.y * cb.y;
  const abLen = Math.hypot(ab.x, ab.y);
  const cbLen = Math.hypot(cb.x, cb.y);
  if (!abLen || !cbLen) return null;

  const cosine = Math.max(-1, Math.min(1, dot / (abLen * cbLen)));
  return Math.round((Math.acos(cosine) * 180) / Math.PI);
}

function formatAngle(value) {
  if (value === null || Number.isNaN(value)) return "—";
  return `${Math.round(value)}°`;
}

function getSkeletonColor(a, b) {
  if (a.startsWith("left") || b.startsWith("left")) return COLORS.left;
  if (a.startsWith("right") || b.startsWith("right")) return COLORS.right;
  if (["leftShoulder", "rightShoulder", "leftHip", "rightHip"].includes(a) || ["leftShoulder", "rightShoulder", "leftHip", "rightHip"].includes(b)) {
    return COLORS.trunk;
  }
  return COLORS.head;
}

export default function FullSkeletonGaitAnalyzer() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(null);

  const [status, setStatus] = useState("Загружаю модель скелета…");
  const [videoUrl, setVideoUrl] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showAngles, setShowAngles] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function initPoseLandmarker() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.45,
          minPosePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });

        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setStatus("Модель готова. Загрузи видео.");
        }
      } catch (error) {
        console.error(error);
        setStatus("Ошибка загрузки MediaPipe Pose. Проверь установку @mediapipe/tasks-vision.");
      }
    }

    initPoseLandmarker();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (landmarkerRef.current) landmarkerRef.current.close();
    };
  }, []);

  function handleVideoUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (videoUrl) URL.revokeObjectURL(videoUrl);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setMetrics(null);
    setStatus("Видео загружено. Нажми Play.");
  }

  function buildPoints(landmarks, width, height) {
    const points = {};
    Object.keys(LANDMARKS).forEach((name) => {
      points[name] = getPoint(landmarks, name, width, height);
    });

    points.midShoulder = midpoint(points.leftShoulder, points.rightShoulder, "midShoulder");
    points.midHip = midpoint(points.leftHip, points.rightHip, "midHip");
    points.neck = midpoint(points.nose, points.midShoulder, "neck");

    return points;
  }

  function computeMetrics(points) {
    const pelvisTilt = vectorAngleDeg(points.leftHip, points.rightHip);
    const shoulderTilt = vectorAngleDeg(points.leftShoulder, points.rightShoulder);
    const torsoTilt = vectorAngleDeg(points.midHip, points.midShoulder);

    const leftKnee = jointAngleDeg(points.leftHip, points.leftKnee, points.leftAnkle);
    const rightKnee = jointAngleDeg(points.rightHip, points.rightKnee, points.rightAnkle);

    const leftElbow = jointAngleDeg(points.leftShoulder, points.leftElbow, points.leftWrist);
    const rightElbow = jointAngleDeg(points.rightShoulder, points.rightElbow, points.rightWrist);

    const leftHip = jointAngleDeg(points.leftShoulder, points.leftHip, points.leftKnee);
    const rightHip = jointAngleDeg(points.rightShoulder, points.rightHip, points.rightKnee);

    const leftAnkle = jointAngleDeg(points.leftKnee, points.leftAnkle, points.leftFootIndex);
    const rightAnkle = jointAngleDeg(points.rightKnee, points.rightAnkle, points.rightFootIndex);

    const leftFootAngle = vectorAngleDeg(points.leftHeel, points.leftFootIndex);
    const rightFootAngle = vectorAngleDeg(points.rightHeel, points.rightFootIndex);

    return {
      pelvisTilt,
      shoulderTilt,
      torsoTilt,
      leftHip,
      rightHip,
      leftKnee,
      rightKnee,
      leftAnkle,
      rightAnkle,
      leftElbow,
      rightElbow,
      leftFootAngle,
      rightFootAngle,
    };
  }

  function drawSkeleton(ctx, points) {
    SKELETON_CONNECTIONS.forEach(([aName, bName]) => {
      drawLine(ctx, points[aName], points[bName], getSkeletonColor(aName, bName));
    });

    // central biomechanical axes
    drawLine(ctx, points.midHip, points.midShoulder, COLORS.center, 5);
    drawLine(ctx, points.nose, points.midShoulder, COLORS.center, 3);

    Object.keys(LANDMARKS).forEach((name) => {
      const color = name.startsWith("left")
        ? COLORS.left
        : name.startsWith("right")
          ? COLORS.right
          : COLORS.head;
      drawPoint(ctx, points[name], color, 5);
    });

    drawPoint(ctx, points.midHip, COLORS.center, 7);
    drawPoint(ctx, points.midShoulder, COLORS.center, 7);
    drawPoint(ctx, points.neck, COLORS.center, 6);

    if (showLabels) {
      const importantLabels = [
        ["nose", "голова"],
        ["leftShoulder", "L плечо"],
        ["rightShoulder", "R плечо"],
        ["leftElbow", "L локоть"],
        ["rightElbow", "R локоть"],
        ["leftWrist", "L кисть"],
        ["rightWrist", "R кисть"],
        ["leftHip", "L таз"],
        ["rightHip", "R таз"],
        ["leftKnee", "L колено"],
        ["rightKnee", "R колено"],
        ["leftAnkle", "L голеностоп"],
        ["rightAnkle", "R голеностоп"],
        ["leftHeel", "L пятка"],
        ["rightHeel", "R пятка"],
        ["leftFootIndex", "L носок"],
        ["rightFootIndex", "R носок"],
      ];

      importantLabels.forEach(([name, label]) => drawLabel(ctx, points[name], label));
    }
  }

  function drawAngleText(ctx, points, currentMetrics) {
    if (!showAngles) return;

    const angleLabels = [
      [points.leftKnee, `L knee ${formatAngle(currentMetrics.leftKnee)}`],
      [points.rightKnee, `R knee ${formatAngle(currentMetrics.rightKnee)}`],
      [points.leftAnkle, `L ankle ${formatAngle(currentMetrics.leftAnkle)}`],
      [points.rightAnkle, `R ankle ${formatAngle(currentMetrics.rightAnkle)}`],
      [points.leftElbow, `L elbow ${formatAngle(currentMetrics.leftElbow)}`],
      [points.rightElbow, `R elbow ${formatAngle(currentMetrics.rightElbow)}`],
      [points.midHip, `pelvis ${formatAngle(currentMetrics.pelvisTilt)}`],
      [points.midShoulder, `torso ${formatAngle(currentMetrics.torsoTilt)}`],
    ];

    ctx.save();
    ctx.font = "13px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillStyle = "white";
    ctx.strokeStyle = "rgba(0,0,0,0.7)";
    ctx.lineWidth = 3;

    angleLabels.forEach(([p, text]) => {
      if (!p) return;
      ctx.strokeText(text, p.x + 10, p.y + 18);
      ctx.fillText(text, p.x + 10, p.y + 18);
    });

    ctx.restore();
  }

  function analyzeFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !canvas || !landmarker || video.paused || video.ended) {
      rafRef.current = requestAnimationFrame(analyzeFrame);
      return;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) {
      rafRef.current = requestAnimationFrame(analyzeFrame);
      return;
    }

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    const result = landmarker.detectForVideo(video, performance.now());
    const landmarks = result.landmarks?.[0];

    if (!landmarks) {
      setStatus("Скелет не найден: человек должен быть целиком в кадре, сбоку, с видимыми стопами.");
      rafRef.current = requestAnimationFrame(analyzeFrame);
      return;
    }

    const points = buildPoints(landmarks, width, height);
    const currentMetrics = computeMetrics(points);

    drawSkeleton(ctx, points);
    drawAngleText(ctx, points, currentMetrics);
    setMetrics(currentMetrics);
    setStatus("Скелет читается: голова, корпус, обе руки, обе ноги, пятки и носки.");

    rafRef.current = requestAnimationFrame(analyzeFrame);
  }

  function handlePlay() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    analyzeFrame();
  }

  return (
    <div className="min-h-screen bg-slate-950 text-white p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-4">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold">Full Skeleton Gait Analyzer</h1>
            <p className="text-slate-300 mt-1">
              Разметка всего тела: голова, корпус, обе руки, таз, обе ноги, пятки и носки.
            </p>
          </div>

          <label className="inline-flex cursor-pointer items-center rounded-2xl bg-white text-slate-950 px-4 py-2 font-medium shadow">
            Загрузить видео
            <input type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />
          </label>
        </div>

        <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3 text-sm text-slate-200">
          {status}
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => setShowLabels((v) => !v)}
            className="rounded-xl bg-slate-800 px-3 py-2 text-sm hover:bg-slate-700"
          >
            {showLabels ? "Скрыть подписи" : "Показать подписи"}
          </button>

          <button
            type="button"
            onClick={() => setShowAngles((v) => !v)}
            className="rounded-xl bg-slate-800 px-3 py-2 text-sm hover:bg-slate-700"
          >
            {showAngles ? "Скрыть углы" : "Показать углы"}
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
          <div className="relative overflow-hidden rounded-2xl border border-slate-700 bg-black shadow-xl">
            {videoUrl ? (
              <>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  playsInline
                  onPlay={handlePlay}
                  className="block w-full h-auto"
                />
                <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
              </>
            ) : (
              <div className="aspect-video flex items-center justify-center text-slate-400">
                Загрузи видео с человеком целиком в кадре
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4 space-y-3">
            <h2 className="text-lg font-semibold">Текущие метрики кадра</h2>

            <div className="grid grid-cols-2 gap-2 text-sm">
              <Metric label="Таз" value={formatAngle(metrics?.pelvisTilt ?? null)} />
              <Metric label="Плечи" value={formatAngle(metrics?.shoulderTilt ?? null)} />
              <Metric label="Корпус" value={formatAngle(metrics?.torsoTilt ?? null)} />
              <Metric label="" value="" muted />

              <Metric label="L бедро" value={formatAngle(metrics?.leftHip ?? null)} />
              <Metric label="R бедро" value={formatAngle(metrics?.rightHip ?? null)} />
              <Metric label="L колено" value={formatAngle(metrics?.leftKnee ?? null)} />
              <Metric label="R колено" value={formatAngle(metrics?.rightKnee ?? null)} />
              <Metric label="L голеностоп" value={formatAngle(metrics?.leftAnkle ?? null)} />
              <Metric label="R голеностоп" value={formatAngle(metrics?.rightAnkle ?? null)} />
              <Metric label="L стопа" value={formatAngle(metrics?.leftFootAngle ?? null)} />
              <Metric label="R стопа" value={formatAngle(metrics?.rightFootAngle ?? null)} />

              <Metric label="L локоть" value={formatAngle(metrics?.leftElbow ?? null)} />
              <Metric label="R локоть" value={formatAngle(metrics?.rightElbow ?? null)} />
            </div>

            <div className="rounded-xl bg-slate-950 p-3 text-xs text-slate-300 leading-relaxed">
              Сейчас это слой чтения скелета. Диагностику и патологические флаги лучше навешивать сверху:
              асимметрия, toe drag, перекос таза, завал корпуса, плохая работа рук.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, muted = false }) {
  if (muted) return <div />;

  return (
    <div className="rounded-xl bg-slate-950 p-3">
      <div className="text-slate-400 text-xs">{label}</div>
      <div className="font-semibold text-base">{value}</div>
    </div>
  );
}

