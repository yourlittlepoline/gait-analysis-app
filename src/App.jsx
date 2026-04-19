import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Upload, AlertCircle, CheckCircle2, Video, Activity } from "lucide-react";

const IDEAL_GAIT = {
  initialContact: { hipFlexion: 30, kneeFlexion: 5, ankleDorsi: 0, pelvicDrop: 0 },
  loadingResponse: { hipFlexion: 25, kneeFlexion: 15, ankleDorsi: 5, pelvicDrop: 3 },
  midStance: { hipFlexion: 0, kneeFlexion: 5, ankleDorsi: 5, pelvicDrop: 0 },
  terminalStance: { hipFlexion: -10, kneeFlexion: 0, ankleDorsi: 10, pelvicDrop: 0 },
  swing: { hipFlexion: 20, kneeFlexion: 60, ankleDorsi: 0, pelvicDrop: 0 },
};

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

function getPoint(landmarks, idx, width, height) {
  const p = landmarks?.[idx];
  if (!p) return null;
  return { x: p.x * width, y: p.y * height };
}

function averagePoint(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;
  return {
    x: valid.reduce((s, p) => s + p.x, 0) / valid.length,
    y: valid.reduce((s, p) => s + p.y, 0) / valid.length,
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function estimatePhase(metrics) {
  if (!metrics) return "midStance";
  const { kneeFlexion, ankleDorsi, heelAhead } = metrics;
  if (heelAhead > 16 && kneeFlexion < 20) return "initialContact";
  if (heelAhead > 8 && kneeFlexion >= 10 && kneeFlexion <= 25) return "loadingResponse";
  if (ankleDorsi >= 6 && kneeFlexion < 12) return "terminalStance";
  if (kneeFlexion > 35) return "swing";
  return "midStance";
}

function formatPhaseName(name) {
  return {
    initialContact: "Initial contact",
    loadingResponse: "Loading response",
    midStance: "Mid stance",
    terminalStance: "Terminal stance",
    swing: "Swing",
  }[name] || name;
}

function drawLine(ctx, a, b, stroke, width = 4) {
  if (!a || !b) return;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.lineWidth = width;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function drawCircle(ctx, x, y, r, fill) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

function idealOverlayFromBody(points, phaseName) {
  const leftHip = points.leftHip;
  const leftKnee = points.leftKnee;
  const leftAnkle = points.leftAnkle;
  const pelvisCenter = averagePoint([points.leftHip, points.rightHip]);
  const shoulderCenter = averagePoint([points.leftShoulder, points.rightShoulder]);
  if (!leftHip || !leftKnee || !leftAnkle || !pelvisCenter || !shoulderCenter) return null;

  const thigh = distance(leftHip, leftKnee) || 90;
  const shank = distance(leftKnee, leftAnkle) || 90;
  const trunk = distance(pelvisCenter, shoulderCenter) || 120;
  const foot = Math.max(28, shank * 0.35);

  const ideal = IDEAL_GAIT[phaseName] || IDEAL_GAIT.midStance;
  const hipAngle = (ideal.hipFlexion * Math.PI) / 180;
  const kneeAngle = ((180 - ideal.kneeFlexion) * Math.PI) / 180;
  const ankleAngle = ((90 - ideal.ankleDorsi) * Math.PI) / 180;

  const hip = { ...leftHip };
  const knee = {
    x: hip.x + Math.sin(hipAngle) * thigh,
    y: hip.y + Math.cos(hipAngle) * thigh,
  };
  const shankDirection = hipAngle - Math.PI + kneeAngle;
  const ankle = {
    x: knee.x + Math.sin(shankDirection) * shank,
    y: knee.y + Math.cos(shankDirection) * shank,
  };
  const toe = {
    x: ankle.x + Math.cos(ankleAngle) * foot,
    y: ankle.y - Math.sin(ankleAngle) * foot * 0.35,
  };
  const shoulder = { x: hip.x, y: hip.y - trunk };

  return { shoulder, hip, knee, ankle, toe };
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

function MetricCard({ label, value }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
      <div className="text-xs uppercase tracking-[0.16em] text-slate-500">{label}</div>
      <div className="mt-2 text-lg font-semibold text-slate-100">{value}</div>
    </div>
  );
}

function SectionList({ title, items }) {
  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold uppercase tracking-[0.16em] text-slate-400">{title}</h3>
      <div className="space-y-2">
        {items.map((item, idx) => (
          <div key={`${title}-${idx}`} className="rounded-2xl border border-slate-800 bg-slate-950/60 p-3 text-sm leading-6 text-slate-200">
            {item}
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState({ text }) {
  return (
    <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-950/50 p-8 text-center text-sm text-slate-400">
      {text}
    </div>
  );
}

export default function GaitAnalysisMVP() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const poseLandmarkerRef = useRef(null);

  const [videoUrl, setVideoUrl] = useState("");
  const [videoName, setVideoName] = useState("");
  const [poseReady, setPoseReady] = useState(false);
  const [loadingPose, setLoadingPose] = useState(false);
  const [status, setStatus] = useState("Загрузи видео походки сбоку. Система выберет ключевые этапы и покажет сравнение: зелёный референс и красная фактическая поза.");
  const [error, setError] = useState("");
  const [frames, setFrames] = useState([]);
  const [selectedFrame, setSelectedFrame] = useState(0);

  const currentFrame = frames[selectedFrame] || null;

  useEffect(() => {
    return () => {
      if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  async function ensurePoseLandmarker() {
    if (poseLandmarkerRef.current) return poseLandmarkerRef.current;
    setLoadingPose(true);
    setError("");
    try {
      const model = await loadPoseLandmarker();
      poseLandmarkerRef.current = model;
      setPoseReady(true);
      setStatus("Модель позы загружена. Можно запускать быстрый анализ.");
      return model;
    } catch (e) {
      console.error(e);
      setError("Не удалось загрузить MediaPipe Pose. Проверь интернет и попробуй ещё раз.");
      throw e;
    } finally {
      setLoadingPose(false);
    }
  }

  function analyzeLandmarks(landmarks, width, height) {
    const leftShoulder = getPoint(landmarks, LANDMARKS.leftShoulder, width, height);
    const rightShoulder = getPoint(landmarks, LANDMARKS.rightShoulder, width, height);
    const leftHip = getPoint(landmarks, LANDMARKS.leftHip, width, height);
    const rightHip = getPoint(landmarks, LANDMARKS.rightHip, width, height);
    const leftKnee = getPoint(landmarks, LANDMARKS.leftKnee, width, height);
    const leftAnkle = getPoint(landmarks, LANDMARKS.leftAnkle, width, height);
    const leftHeel = getPoint(landmarks, LANDMARKS.leftHeel, width, height);
    const rightHeel = getPoint(landmarks, LANDMARKS.rightHeel, width, height);
    const leftFootIndex = getPoint(landmarks, LANDMARKS.leftFootIndex, width, height);

    const shoulderCenter = averagePoint([leftShoulder, rightShoulder]);
    const pelvisCenter = averagePoint([leftHip, rightHip]);

    const hipFlexion = signedAngleToVertical(leftHip, leftKnee) ?? 0;
    const rawKnee = angle3(leftHip, leftKnee, leftAnkle) ?? 180;
    const kneeFlexion = Math.max(0, 180 - rawKnee);
    const rawAnkle = angle3(leftKnee, leftAnkle, leftFootIndex) ?? 90;
    const ankleDorsi = 90 - rawAnkle;
    const pelvicDrop = shoulderCenter && pelvisCenter ? signedAngleToVertical(shoulderCenter, pelvisCenter) ?? 0 : 0;
    const heelAhead = leftHeel && rightHeel ? leftHeel.x - rightHeel.x : 0;

    const metrics = { hipFlexion, kneeFlexion, ankleDorsi, pelvicDrop, heelAhead };
    const phase = estimatePhase(metrics);
    return { phase, metrics };
  }

  function deviationComments(metrics, phase) {
    const ideal = IDEAL_GAIT[phase] || IDEAL_GAIT.midStance;
    const notes = [];

    if (Math.abs(metrics.hipFlexion - ideal.hipFlexion) > 12) {
      notes.push(metrics.hipFlexion > ideal.hipFlexion ? "Бедро подано вперёд сильнее референса." : "Бедро работает менее активно, чем ожидается в этой фазе.");
    }
    if (Math.abs(metrics.kneeFlexion - ideal.kneeFlexion) > 12) {
      notes.push(metrics.kneeFlexion > ideal.kneeFlexion ? "Колено согнуто больше нормы для этой фазы." : "Колено сгибается меньше нормы для этой фазы.");
    }
    if (Math.abs(metrics.ankleDorsi - ideal.ankleDorsi) > 8) {
      notes.push(metrics.ankleDorsi > ideal.ankleDorsi ? "Голеностоп уходит в избыточное тыльное сгибание." : "Не хватает тыльного сгибания в голеностопе.");
    }
    if (Math.abs(metrics.pelvicDrop - ideal.pelvicDrop) > 5) {
      notes.push("Есть отклонение по положению таза или компенсаторный наклон корпуса.");
    }
    if (!notes.length) {
      notes.push("Грубых отклонений в этой фазе не видно.");
    }

    return notes;
  }

  function drawComparison(frame) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !frame) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(video, 0, 0, width, height);

    const points = {
      leftShoulder: getPoint(frame.landmarks, LANDMARKS.leftShoulder, width, height),
      rightShoulder: getPoint(frame.landmarks, LANDMARKS.rightShoulder, width, height),
      leftHip: getPoint(frame.landmarks, LANDMARKS.leftHip, width, height),
      rightHip: getPoint(frame.landmarks, LANDMARKS.rightHip, width, height),
      leftKnee: getPoint(frame.landmarks, LANDMARKS.leftKnee, width, height),
      rightKnee: getPoint(frame.landmarks, LANDMARKS.rightKnee, width, height),
      leftAnkle: getPoint(frame.landmarks, LANDMARKS.leftAnkle, width, height),
      rightAnkle: getPoint(frame.landmarks, LANDMARKS.rightAnkle, width, height),
      leftHeel: getPoint(frame.landmarks, LANDMARKS.leftHeel, width, height),
      rightHeel: getPoint(frame.landmarks, LANDMARKS.rightHeel, width, height),
      leftFootIndex: getPoint(frame.landmarks, LANDMARKS.leftFootIndex, width, height),
      rightFootIndex: getPoint(frame.landmarks, LANDMARKS.rightFootIndex, width, height),
    };

    const ideal = idealOverlayFromBody(points, frame.phase);
    if (ideal) {
      drawLine(ctx, ideal.shoulder, ideal.hip, "rgba(34,197,94,0.95)");
      drawLine(ctx, ideal.hip, ideal.knee, "rgba(34,197,94,0.95)");
      drawLine(ctx, ideal.knee, ideal.ankle, "rgba(34,197,94,0.95)");
      drawLine(ctx, ideal.ankle, ideal.toe, "rgba(34,197,94,0.95)");
      [ideal.shoulder, ideal.hip, ideal.knee, ideal.ankle, ideal.toe].forEach((p) => drawCircle(ctx, p.x, p.y, 4, "rgba(34,197,94,0.95)"));
    }

    const actual = [points.leftShoulder, points.leftHip, points.leftKnee, points.leftAnkle, points.leftFootIndex].filter(Boolean);
    for (let i = 0; i < actual.length - 1; i += 1) {
      drawLine(ctx, actual[i], actual[i + 1], "rgba(239,68,68,0.95)");
      drawCircle(ctx, actual[i].x, actual[i].y, 4, "rgba(239,68,68,0.95)");
    }
    const last = actual[actual.length - 1];
    if (last) drawCircle(ctx, last.x, last.y, 4, "rgba(239,68,68,0.95)");

    ctx.fillStyle = "rgba(2,6,23,0.8)";
    ctx.fillRect(16, 16, 420, 130);
    ctx.fillStyle = "white";
    ctx.font = "700 15px sans-serif";
    ctx.fillText(formatPhaseName(frame.phase), 28, 42);
    ctx.fillStyle = "#22c55e";
    ctx.font = "500 14px sans-serif";
    ctx.fillText("Зелёный — как должно быть", 28, 74);
    ctx.fillStyle = "#ef4444";
    ctx.fillText("Красный — как есть", 28, 102);
  }

  async function runFastAnalysis() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !videoUrl) {
      setError("Сначала загрузи видео.");
      return;
    }

    setError("");
    setFrames([]);
    setSelectedFrame(0);
    setStatus("Идёт быстрый анализ ключевых фаз...");

    let model;
    try {
      model = await ensurePoseLandmarker();
    } catch {
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const duration = video.duration || 1;
    const checkpoints = [0.1, 0.28, 0.45, 0.62, 0.8].map((p) => Math.min(duration * p, Math.max(duration - 0.05, 0)));
    const order = ["initialContact", "loadingResponse", "midStance", "terminalStance", "swing"];
    const byPhase = {};

    video.pause();

    for (const t of checkpoints) {
      await new Promise((resolve) => {
        const onSeeked = () => {
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        video.addEventListener("seeked", onSeeked);
        video.currentTime = t;
      });

      const poseResult = model.detectForVideo(video, performance.now());
      const landmarks = poseResult?.landmarks?.[0];
      if (!landmarks) continue;

      const analyzed = analyzeLandmarks(landmarks, width, height);
      if (!byPhase[analyzed.phase]) {
        byPhase[analyzed.phase] = {
          time: t,
          phase: analyzed.phase,
          landmarks,
          metrics: analyzed.metrics,
          comments: deviationComments(analyzed.metrics, analyzed.phase),
        };
      }
    }

    const selected = order.map((phase) => byPhase[phase]).filter(Boolean);
    if (!selected.length) {
      setStatus("Не удалось выделить ключевые фазы. Попробуй более чистое видео сбоку.");
      return;
    }

    setFrames(selected);
    setSelectedFrame(0);
    setStatus(`Готово: найдено фаз ${selected.length}. Показываю только ключевые этапы и отклонения.`);
    setTimeout(() => drawComparison(selected[0]), 50);
  }

  function handleVideoLoaded() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = 960;
    canvas.height = 540;
    setStatus("Видео загружено. Можно запускать быстрый анализ.");
  }

  function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setVideoName(file.name);
    setFrames([]);
    setSelectedFrame(0);
    setError("");
  }

  useEffect(() => {
    if (currentFrame) drawComparison(currentFrame);
  }, [currentFrame]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <div className="mx-auto max-w-7xl p-4 md:p-8">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 grid gap-4 lg:grid-cols-[1.3fr_0.7fr]"
        >
          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
            <div className="flex flex-wrap items-center gap-3">
              <span className="rounded-full bg-slate-100 px-3 py-1 text-sm font-medium text-slate-900">Fast Gait MVP</span>
              <span className="rounded-full border border-slate-700 px-3 py-1 text-sm text-slate-300">быстрое фазовое сравнение вместо долгого покадрового анализа</span>
            </div>
            <h1 className="mt-4 text-3xl font-semibold tracking-tight">Поэтапное сравнение походки: как должно быть vs как есть</h1>
            <p className="mt-3 max-w-3xl text-slate-400">
              Загрузи видео. Система выделит только ключевые этапы походки, наложит зелёный референс поверх красной фактической позы и даст короткие комментарии по отклонениям.
            </p>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
            <h2 className="text-lg font-semibold">Статус системы</h2>
            <p className="mt-1 text-slate-400">Упрощённая версия ради скорости.</p>
            <div className="mt-4 flex items-center justify-between rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
              <span className="text-sm text-slate-300">Модель позы</span>
              {poseReady ? (
                <span className="flex items-center gap-2 text-sm text-emerald-400"><CheckCircle2 className="h-4 w-4" /> готова</span>
              ) : (
                <span className="flex items-center gap-2 text-sm text-amber-400"><AlertCircle className="h-4 w-4" /> не загружена</span>
              )}
            </div>
            <p className="mt-4 text-sm leading-6 text-slate-300">{status}</p>
            {error ? <p className="mt-4 rounded-2xl border border-red-900/50 bg-red-950/40 p-3 text-sm text-red-300">{error}</p> : null}
            <button
              onClick={ensurePoseLandmarker}
              disabled={loadingPose || poseReady}
              className="mt-4 w-full rounded-2xl bg-slate-100 px-4 py-3 font-medium text-slate-900 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loadingPose ? "Загрузка модели…" : poseReady ? "Модель загружена" : "Подготовить модель"}
            </button>
          </div>
        </motion.div>

        <div className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
              <h2 className="flex items-center gap-2 text-xl font-semibold"><Video className="h-5 w-5" /> Видео</h2>
              <p className="mt-2 text-slate-400">Без камеры, без длинного отчёта, только видео сбоку и быстрый фазовый анализ.</p>

              <div className="mt-4 rounded-3xl border border-dashed border-slate-700 bg-slate-950/50 p-6 text-center">
                <Upload className="mx-auto mb-3 h-8 w-8 text-slate-400" />
                <p className="mb-3 text-sm text-slate-300">Загрузи MP4 / MOV / WEBM с проходкой.</p>
                <input type="file" accept="video/*" onChange={handleUpload} className="block w-full rounded-2xl border border-slate-800 bg-slate-900 p-3 text-slate-100" />
                {videoName ? <p className="mt-3 text-sm text-slate-400">Файл: {videoName}</p> : null}
              </div>

              <div className="mt-6 overflow-hidden rounded-3xl border border-slate-800 bg-black">
                <div className="relative aspect-video w-full bg-black">
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    playsInline
                    controls={false}
                    className="hidden"
                    onLoadedMetadata={handleVideoLoaded}
                  />
                  <canvas ref={canvasRef} width={960} height={540} className="h-full w-full" />
                </div>
              </div>

              <button
                onClick={runFastAnalysis}
                disabled={!videoUrl || loadingPose}
                className="mt-4 inline-flex items-center gap-2 rounded-2xl bg-slate-700 px-4 py-3 font-medium text-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Activity className="h-4 w-4" /> Быстрый анализ
              </button>
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
              <h2 className="text-xl font-semibold">Этапы походки</h2>
              <p className="mt-2 text-slate-400">Показываем только найденные ключевые этапы.</p>
              <div className="mt-4 space-y-2">
                {frames.length ? (
                  frames.map((frame, idx) => (
                    <button
                      key={`${frame.phase}-${idx}`}
                      onClick={() => setSelectedFrame(idx)}
                      className={`w-full rounded-2xl px-4 py-3 text-left font-medium ${idx === selectedFrame ? "bg-slate-100 text-slate-900" : "bg-slate-800 text-slate-100"}`}
                    >
                      {formatPhaseName(frame.phase)}
                    </button>
                  ))
                ) : (
                  <EmptyState text="После анализа здесь появятся ключевые этапы походки." />
                )}
              </div>
            </div>

            <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
              <h2 className="text-xl font-semibold">Отклонения в выбранной фазе</h2>
              <p className="mt-2 text-slate-400">Зелёный — как должно быть. Красный — как есть.</p>
              <div className="mt-4">
                {currentFrame ? (
                  <div className="space-y-5">
                    <div className="grid grid-cols-2 gap-3">
                      <MetricCard label="Фаза" value={formatPhaseName(currentFrame.phase)} />
                      <MetricCard label="Hip" value={`${currentFrame.metrics.hipFlexion.toFixed(1)}°`} />
                      <MetricCard label="Knee" value={`${currentFrame.metrics.kneeFlexion.toFixed(1)}°`} />
                      <MetricCard label="Ankle" value={`${currentFrame.metrics.ankleDorsi.toFixed(1)}°`} />
                    </div>
                    <SectionList title="Комментарии" items={currentFrame.comments} />
                  </div>
                ) : (
                  <EmptyState text="Выбери фазу, и здесь появятся комментарии по отклонениям." />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
