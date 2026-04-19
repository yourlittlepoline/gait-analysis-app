import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Upload, AlertCircle, CheckCircle2, Video, Activity } from "lucide-react";

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

const PHASE_REFERENCE = {
  initialContact: { hip: 30, knee: 5, ankle: 0 },
  loadingResponse: { hip: 25, knee: 15, ankle: 5 },
  midStance: { hip: 0, knee: 5, ankle: 5 },
  terminalStance: { hip: -10, knee: 0, ankle: 10 },
  swing: { hip: 20, knee: 60, ankle: 0 },
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
  return {
    x: p.x * width,
    y: p.y * height,
    visibility: p.visibility ?? 0,
  };
}

function averagePoint(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;
  return {
    x: valid.reduce((s, p) => s + p.x, 0) / valid.length,
    y: valid.reduce((s, p) => s + p.y, 0) / valid.length,
  };
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

function estimatePhase(metrics) {
  if (!metrics) return "midStance";
  const { knee, ankle, heelAhead } = metrics;
  if (heelAhead > 16 && knee < 20) return "initialContact";
  if (heelAhead > 8 && knee >= 10 && knee <= 25) return "loadingResponse";
  if (ankle >= 6 && knee < 12) return "terminalStance";
  if (knee > 35) return "swing";
  return "midStance";
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

function drawArcLabel(ctx, b, a, c, label, color) {
  if (!a || !b || !c) return;
  ctx.fillStyle = color;
  ctx.font = "700 14px sans-serif";
  ctx.fillText(label, b.x + 10, b.y - 10);
}

function extractMetrics(landmarks, width, height) {
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

  const hip = signedAngleToVertical(leftHip, leftKnee) ?? 0;
  const rawKnee = angle3(leftHip, leftKnee, leftAnkle) ?? 180;
  const knee = Math.max(0, 180 - rawKnee);
  const rawAnkle = angle3(leftKnee, leftAnkle, leftFootIndex) ?? 90;
  const ankle = 90 - rawAnkle;
  const trunk = shoulderCenter && pelvisCenter ? signedAngleToVertical(shoulderCenter, pelvisCenter) ?? 0 : 0;
  const heelAhead = leftHeel && rightHeel ? leftHeel.x - rightHeel.x : 0;

  const visiblePoints = [leftShoulder, leftHip, leftKnee, leftAnkle, leftFootIndex].filter(Boolean);
  const meanX = visiblePoints.reduce((s, p) => s + p.x, 0) / Math.max(visiblePoints.length, 1);
  const bodyHeight = visiblePoints.length >= 4 ? Math.abs((leftShoulder?.y ?? 0) - (leftAnkle?.y ?? height)) : 0;

  return {
    points: { leftShoulder, rightShoulder, leftHip, rightHip, leftKnee, leftAnkle, leftFootIndex, leftHeel, rightHeel },
    metrics: { hip, knee, ankle, trunk, heelAhead, meanX, bodyHeight },
  };
}

function frameQuality(result, width, height) {
  if (!result) return 0;
  const { points, metrics } = result;
  const required = [points.leftShoulder, points.leftHip, points.leftKnee, points.leftAnkle, points.leftFootIndex];
  const present = required.filter(Boolean).length;
  let score = present * 20;
  if (metrics.bodyHeight > height * 0.35) score += 15;
  if (metrics.meanX > width * 0.15 && metrics.meanX < width * 0.85) score += 15;
  return score;
}

function buildComments(phase, metrics) {
  const ref = PHASE_REFERENCE[phase] || PHASE_REFERENCE.midStance;
  const notes = [];

  if (Math.abs(metrics.hip - ref.hip) > 12) {
    notes.push(metrics.hip > ref.hip ? "Бедро подано вперёд сильнее референса." : "Бедро работает менее активно, чем ожидается для этой фазы.");
  }
  if (Math.abs(metrics.knee - ref.knee) > 12) {
    notes.push(metrics.knee > ref.knee ? "Колено согнуто больше нормы для этой фазы." : "Колено сгибается меньше нормы для этой фазы.");
  }
  if (Math.abs(metrics.ankle - ref.ankle) > 8) {
    notes.push(metrics.ankle > ref.ankle ? "Тыльное сгибание голеностопа выше референса." : "Не хватает тыльного сгибания в голеностопе.");
  }
  if (Math.abs(metrics.trunk) > 8) {
    notes.push(metrics.trunk > 0 ? "Корпус заметно наклонён вперёд/в сторону." : "Есть отклонение положения корпуса от более нейтральной стойки.");
  }
  if (!notes.length) notes.push("Грубых отклонений по этому кадру не видно.");
  return notes;
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
  const [status, setStatus] = useState("Загрузи видео походки сбоку. Система выберет несколько удачных кадров, нарисует реальные углы и покажет простые комментарии по фазам.");
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
      setStatus("Модель позы загружена. Можно запускать быстрый анализ кадров.");
      return model;
    } catch (e) {
      console.error(e);
      setError("Не удалось загрузить модель позы. Проверь интернет и попробуй ещё раз.");
      throw e;
    } finally {
      setLoadingPose(false);
    }
  }

  function drawFrame(frame) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !frame) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(video, 0, 0, width, height);

    const p = frame.points;
    const blue = "rgba(56,189,248,0.95)";
    drawLine(ctx, p.leftShoulder, p.leftHip, blue);
    drawLine(ctx, p.leftHip, p.leftKnee, blue);
    drawLine(ctx, p.leftKnee, p.leftAnkle, blue);
    drawLine(ctx, p.leftAnkle, p.leftFootIndex, blue);

    [p.leftShoulder, p.leftHip, p.leftKnee, p.leftAnkle, p.leftFootIndex].filter(Boolean).forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = blue;
      ctx.fill();
    });

    drawArcLabel(ctx, p.leftHip, p.leftShoulder, p.leftKnee, `hip ${frame.metrics.hip.toFixed(0)}°`, "#f8fafc");
    drawArcLabel(ctx, p.leftKnee, p.leftHip, p.leftAnkle, `knee ${frame.metrics.knee.toFixed(0)}°`, "#f8fafc");
    drawArcLabel(ctx, p.leftAnkle, p.leftKnee, p.leftFootIndex, `ankle ${frame.metrics.ankle.toFixed(0)}°`, "#f8fafc");

    ctx.fillStyle = "rgba(2,6,23,0.82)";
    ctx.fillRect(16, 16, 320, 88);
    ctx.fillStyle = "white";
    ctx.font = "700 15px sans-serif";
    ctx.fillText(formatPhaseName(frame.phase), 28, 44);
    ctx.font = "500 14px sans-serif";
    ctx.fillText(`frame ${selectedFrame + 1} / ${frames.length}`, 28, 72);
  }

  async function runFastAnalysis() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !videoUrl) {
      setError("Сначала загрузи видео.");
      return;
    }
    if (video.readyState < 2) {
      setError("Видео ещё не готово. Подожди пару секунд и попробуй снова.");
      return;
    }

    setError("");
    setFrames([]);
    setSelectedFrame(0);
    setStatus("Ищу несколько удачных кадров и считаю углы...");

    let model;
    try {
      model = await ensurePoseLandmarker();
    } catch {
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const duration = video.duration || 1;
    const checkpoints = [0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82].map((p) =>
      Math.min(duration * p, Math.max(duration - 0.2, 0))
    );

    const candidates = [];
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

      const result = extractMetrics(landmarks, width, height);
      const quality = frameQuality(result, width, height);
      if (quality < 60) continue;

      const phase = estimatePhase(result.metrics);
      candidates.push({
        time: t,
        phase,
        landmarks,
        points: result.points,
        metrics: result.metrics,
        quality,
        comments: buildComments(phase, result.metrics),
      });
    }

    if (!candidates.length) {
      setStatus("Не нашёл ни одного хорошего кадра. Нужен вид сбоку и человек целиком в кадре.");
      setError("Видео не подходит для анализа: человек должен быть целиком в кадре, сбоку, без сильного наклона камеры.");
      return;
    }

    const bestByPhase = {};
    for (const candidate of candidates) {
      if (!bestByPhase[candidate.phase] || candidate.quality > bestByPhase[candidate.phase].quality) {
        bestByPhase[candidate.phase] = candidate;
      }
    }

    const orderedPhases = ["initialContact", "loadingResponse", "midStance", "terminalStance", "swing"];
    const selected = orderedPhases.map((phase) => bestByPhase[phase]).filter(Boolean).slice(0, 4);

    setFrames(selected);
    setSelectedFrame(0);
    setStatus(`Готово: выбрано кадров ${selected.length}. Показываю реальные углы и короткие комментарии.`);

    setTimeout(() => {
      if (selected[0]) drawFrame(selected[0]);
    }, 60);
  }

  function handleVideoLoaded() {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 540;
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
    setStatus("Видео выбрано. Ждём загрузку метаданных...");

    setTimeout(() => {
      if (videoRef.current) videoRef.current.load();
    }, 50);
  }

  useEffect(() => {
    if (currentFrame) drawFrame(currentFrame);
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
              <span className="rounded-full bg-slate-100 px-3 py-1 text-sm font-medium text-slate-900">Angle Gait MVP</span>
              <span className="rounded-full border border-slate-700 px-3 py-1 text-sm text-slate-300">без фейковой идеальной наложки, только реальные углы и фазы</span>
            </div>
            <h1 className="mt-4 text-3xl font-semibold tracking-tight">Разбор походки по кадрам и углам</h1>
            <p className="mt-3 max-w-3xl text-slate-400">
              Загрузи видео. Система выберет несколько пригодных кадров, посчитает hip, knee и ankle angle, присвоит вероятную фазу походки и даст короткие комментарии по отклонениям.
            </p>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
            <h2 className="text-lg font-semibold">Статус системы</h2>
            <p className="mt-1 text-slate-400">Упрощённая версия ради стабильности.</p>
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
              <p className="mt-2 text-slate-400">Нужен боковой вид, человек целиком в кадре, лучше 5–8 секунд и 2–4 шага.</p>

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
                    muted
                    preload="metadata"
                    controls
                    onLoadedMetadata={handleVideoLoaded}
                    className="h-full w-full object-contain"
                  />
                  <canvas ref={canvasRef} className="absolute inset-0 h-full w-full pointer-events-none" />
                </div>
              </div>

              <button
                onClick={runFastAnalysis}
                disabled={!videoUrl || loadingPose}
                className="mt-4 inline-flex items-center gap-2 rounded-2xl bg-slate-700 px-4 py-3 font-medium text-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Activity className="h-4 w-4" /> Анализировать кадры
              </button>
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
              <h2 className="text-xl font-semibold">Выбранные кадры</h2>
              <p className="mt-2 text-slate-400">Берём не всё видео, а только несколько наиболее пригодных кадров.</p>
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
                  <EmptyState text="После анализа здесь появятся выбранные кадры и фазы." />
                )}
              </div>
            </div>

            <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6 shadow-2xl">
              <h2 className="text-xl font-semibold">Углы и комментарии</h2>
              <p className="mt-2 text-slate-400">Показываем реальные измеренные углы, а не псевдо-идеальный скелет.</p>
              <div className="mt-4">
                {currentFrame ? (
                  <div className="space-y-5">
                    <div className="grid grid-cols-2 gap-3">
                      <MetricCard label="Фаза" value={formatPhaseName(currentFrame.phase)} />
                      <MetricCard label="Hip" value={`${currentFrame.metrics.hip.toFixed(1)}°`} />
                      <MetricCard label="Knee" value={`${currentFrame.metrics.knee.toFixed(1)}°`} />
                      <MetricCard label="Ankle" value={`${currentFrame.metrics.ankle.toFixed(1)}°`} />
                    </div>
                    <SectionList title="Комментарии" items={currentFrame.comments} />
                  </div>
                ) : (
                  <EmptyState text="После анализа здесь появятся углы и короткие выводы по выбранному кадру." />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
