import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Upload, Camera, Play, Pause, Download, RefreshCw, AlertCircle, CheckCircle2, Video, Activity, FileText, Settings2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";

/**
 * Gait Analysis MVP
 * - Upload or record gait video
 * - Run browser-side pose analysis with MediaPipe (if available)
 * - Overlay "ideal gait" reference skeleton + live deviations
 * - Generate a downloadable report with hypotheses and next tests
 *
 * This is an MVP scaffold for a clinically-inspired product, not a medical device.
 */

const IDEAL_GAIT = {
  initialContact: {
    hipFlexion: 30,
    kneeFlexion: 5,
    ankleDorsi: 0,
    pelvicDrop: 0,
  },
  loadingResponse: {
    hipFlexion: 25,
    kneeFlexion: 15,
    ankleDorsi: 5,
    pelvicDrop: 3,
  },
  midStance: {
    hipFlexion: 0,
    kneeFlexion: 5,
    ankleDorsi: 5,
    pelvicDrop: 0,
  },
  terminalStance: {
    hipFlexion: -10,
    kneeFlexion: 0,
    ankleDorsi: 10,
    pelvicDrop: 0,
  },
  swing: {
    hipFlexion: 20,
    kneeFlexion: 60,
    ankleDorsi: 0,
    pelvicDrop: 0,
  },
};

const PHASE_SEQUENCE = [
  "initialContact",
  "loadingResponse",
  "midStance",
  "terminalStance",
  "swing",
];

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

const SKELETON_CONNECTIONS = [
  [11, 12],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [27, 29],
  [27, 31],
  [24, 26],
  [26, 28],
  [28, 30],
  [28, 32],
];

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function radToDeg(rad) {
  return (rad * 180) / Math.PI;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
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
    z: p.z ?? 0,
    visibility: p.visibility ?? 1,
  };
}

function averagePoint(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;
  return {
    x: valid.reduce((s, p) => s + p.x, 0) / valid.length,
    y: valid.reduce((s, p) => s + p.y, 0) / valid.length,
    z: valid.reduce((s, p) => s + p.z, 0) / valid.length,
  };
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

function phaseScore(metrics, phaseName) {
  const ideal = IDEAL_GAIT[phaseName];
  if (!metrics || !ideal) return 0;
  const diffs = [
    Math.abs(metrics.hipFlexion - ideal.hipFlexion),
    Math.abs(metrics.kneeFlexion - ideal.kneeFlexion),
    Math.abs(metrics.ankleDorsi - ideal.ankleDorsi),
    Math.abs(metrics.pelvicDrop - ideal.pelvicDrop),
  ];
  const normalized = diffs.map((d) => Math.max(0, 100 - d * 4));
  return Math.round(normalized.reduce((a, b) => a + b, 0) / normalized.length);
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

function drawCircle(ctx, x, y, r, fill) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

function drawLine(ctx, a, b, stroke, width = 2, dash = []) {
  if (!a || !b) return;
  ctx.beginPath();
  ctx.setLineDash(dash);
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.lineWidth = width;
  ctx.strokeStyle = stroke;
  ctx.stroke();
  ctx.setLineDash([]);
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
  const shoulder = {
    x: hip.x,
    y: hip.y - trunk,
  };

  return { hip, knee, ankle, toe, shoulder };
}

async function loadPoseLandmarker() {
  const tasksVision = await import("@mediapipe/tasks-vision");
  const { FilesetResolver, PoseLandmarker } = tasksVision;

  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  const poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  return poseLandmarker;
}

export default function GaitAnalysisMVP() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const chunksRef = useRef([]);

  const [sourceMode, setSourceMode] = useState("upload");
  const [videoUrl, setVideoUrl] = useState("");
  const [videoName, setVideoName] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [poseReady, setPoseReady] = useState(false);
  const [loadingPose, setLoadingPose] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [status, setStatus] = useState("Загрузите или снимите видео сбоку, чтобы начать анализ.");
  const [error, setError] = useState("");
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState(null);
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [clinicalContext, setClinicalContext] = useState(
    "Пример: правосторонний AFO, drop foot, жалобы на instability в mid stance."
  );

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animationRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (videoUrl?.startsWith("blob:")) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  const currentResult = results[selectedFrame] || null;

  const overallMetrics = useMemo(() => {
    if (!results.length) return null;
    const avg = (key) =>
      results.reduce((sum, item) => sum + (item.metrics?.[key] ?? 0), 0) / results.length;
    const dominantPhase = Object.entries(
      results.reduce((acc, r) => {
        acc[r.phase] = (acc[r.phase] || 0) + 1;
        return acc;
      }, {})
    ).sort((a, b) => b[1] - a[1])[0]?.[0];

    return {
      avgHipFlexion: avg("hipFlexion").toFixed(1),
      avgKneeFlexion: avg("kneeFlexion").toFixed(1),
      avgAnkleDorsi: avg("ankleDorsi").toFixed(1),
      avgPelvicDrop: avg("pelvicDrop").toFixed(1),
      avgScore: Math.round(avg("score")),
      dominantPhase,
    };
  }, [results]);

  async function ensurePoseLandmarker() {
    if (poseLandmarkerRef.current) return poseLandmarkerRef.current;
    setLoadingPose(true);
    setError("");
    try {
      const model = await loadPoseLandmarker();
      poseLandmarkerRef.current = model;
      setPoseReady(true);
      setStatus("Модель позы загружена. Можно запускать анализ.");
      return model;
    } catch (e) {
      console.error(e);
      setError("Не удалось загрузить MediaPipe Pose. MVP всё ещё показывает UI, но реальный анализ не запустится, пока не загрузится модель.");
      throw e;
    } finally {
      setLoadingPose(false);
    }
  }

  function drawFrame(result) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(video, 0, 0, width, height);

    if (!result?.landmarks) return;

    const points = {
      leftShoulder: getPoint(result.landmarks, LANDMARKS.leftShoulder, width, height),
      rightShoulder: getPoint(result.landmarks, LANDMARKS.rightShoulder, width, height),
      leftHip: getPoint(result.landmarks, LANDMARKS.leftHip, width, height),
      rightHip: getPoint(result.landmarks, LANDMARKS.rightHip, width, height),
      leftKnee: getPoint(result.landmarks, LANDMARKS.leftKnee, width, height),
      rightKnee: getPoint(result.landmarks, LANDMARKS.rightKnee, width, height),
      leftAnkle: getPoint(result.landmarks, LANDMARKS.leftAnkle, width, height),
      rightAnkle: getPoint(result.landmarks, LANDMARKS.rightAnkle, width, height),
      leftHeel: getPoint(result.landmarks, LANDMARKS.leftHeel, width, height),
      rightHeel: getPoint(result.landmarks, LANDMARKS.rightHeel, width, height),
      leftFootIndex: getPoint(result.landmarks, LANDMARKS.leftFootIndex, width, height),
      rightFootIndex: getPoint(result.landmarks, LANDMARKS.rightFootIndex, width, height),
    };

    const ideal = idealOverlayFromBody(points, result.phase);
    if (ideal) {
      drawLine(ctx, ideal.shoulder, ideal.hip, "rgba(255,255,255,0.85)", 3, [8, 6]);
      drawLine(ctx, ideal.hip, ideal.knee, "rgba(255,255,255,0.85)", 3, [8, 6]);
      drawLine(ctx, ideal.knee, ideal.ankle, "rgba(255,255,255,0.85)", 3, [8, 6]);
      drawLine(ctx, ideal.ankle, ideal.toe, "rgba(255,255,255,0.85)", 3, [8, 6]);
      [ideal.shoulder, ideal.hip, ideal.knee, ideal.ankle, ideal.toe].forEach((p) => drawCircle(ctx, p.x, p.y, 4, "rgba(255,255,255,0.95)"));
    }

    SKELETON_CONNECTIONS.forEach(([a, b]) => {
      const pa = getPoint(result.landmarks, a, width, height);
      const pb = getPoint(result.landmarks, b, width, height);
      drawLine(ctx, pa, pb, "rgba(255,255,255,0.45)", 2);
    });

    Object.values(points).forEach((p) => {
      if (!p) return;
      drawCircle(ctx, p.x, p.y, 4, "rgba(255,255,255,0.95)");
    });

    ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
    ctx.fillRect(16, 16, 320, 132);
    ctx.fillStyle = "white";
    ctx.font = "600 14px sans-serif";
    ctx.fillText(`Phase: ${formatPhaseName(result.phase)}`, 28, 42);
    ctx.fillText(`Match score: ${result.metrics.score}/100`, 28, 66);
    ctx.fillText(`Hip: ${result.metrics.hipFlexion.toFixed(1)}°`, 28, 90);
    ctx.fillText(`Knee: ${result.metrics.kneeFlexion.toFixed(1)}°`, 28, 114);
    ctx.fillText(`Ankle: ${result.metrics.ankleDorsi.toFixed(1)}°`, 170, 90);
    ctx.fillText(`Pelvis: ${result.metrics.pelvicDrop.toFixed(1)}°`, 170, 114);
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

    const metrics = {
      hipFlexion,
      kneeFlexion,
      ankleDorsi,
      pelvicDrop,
      heelAhead,
    };

    const phase = estimatePhase(metrics);
    const score = phaseScore(metrics, phase);

    return {
      phase,
      metrics: {
        ...metrics,
        score,
      },
    };
  }

  async function runAnalysis() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !videoUrl) {
      setError("Сначала загрузите или снимите видео.");
      return;
    }

    setAnalysisProgress(0);
    setResults([]);
    setSummary(null);
    setError("");
    setStatus("Запускаю анализ видео…");

    let model;
    try {
      model = await ensurePoseLandmarker();
    } catch {
      return;
    }

    const width = canvas.width;
    const height = canvas.height;

    const sampleCount = 24;
    const durationMs = (video.duration || 1) * 1000;
    const sampleTimes = Array.from({ length: sampleCount }, (_, i) => (durationMs / sampleCount) * i);
    const allResults = [];

    const wasPaused = video.paused;
    video.pause();

    for (let i = 0; i < sampleTimes.length; i += 1) {
      const t = sampleTimes[i] / 1000;
      await new Promise((resolve) => {
        const onSeeked = () => {
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        video.addEventListener("seeked", onSeeked);
        video.currentTime = Math.min(t, Math.max((video.duration || 0.1) - 0.05, 0));
      });

      const poseResult = model.detectForVideo(video, performance.now());
      const landmarks = poseResult?.landmarks?.[0];
      if (landmarks) {
        const analyzed = analyzeLandmarks(landmarks, width, height);
        allResults.push({
          time: t,
          landmarks,
          phase: analyzed.phase,
          metrics: analyzed.metrics,
        });
      }
      setAnalysisProgress(Math.round(((i + 1) / sampleTimes.length) * 100));
    }

    setResults(allResults);
    setSelectedFrame(0);
    setSummary(buildSummary(allResults, clinicalContext, videoName || "session-video"));

    if (!wasPaused) {
      video.play().catch(() => {});
    }

    setStatus(allResults.length ? "Анализ завершён. Можно просматривать ключевые кадры и выгружать отчёт." : "Не удалось получить позу ни на одном кадре. Попробуйте видео сбоку, при хорошем освещении.");
  }

  function buildSummary(data, context, name) {
    if (!data.length) return null;

    const avg = (key) => data.reduce((sum, item) => sum + (item.metrics?.[key] ?? 0), 0) / data.length;
    const avgScore = Math.round(avg("score"));
    const avgKnee = avg("kneeFlexion");
    const avgAnkle = avg("ankleDorsi");
    const avgPelvis = avg("pelvicDrop");

    const issues = [];
    if (avgScore < 70) issues.push("Общая кинематика заметно отклоняется от референсного паттерна.");
    if (avgKnee < 8) issues.push("Есть тенденция к недостаточному сгибанию колена в analysed cycle.");
    if (avgKnee > 28) issues.push("Есть тенденция к избыточному сгибанию колена относительно референса.");
    if (avgAnkle < -2) issues.push("Виден дефицит тыльного сгибания или forefoot clearance strategy.");
    if (Math.abs(avgPelvis) > 4) issues.push("Есть заметная фронтальная нестабильность таза / компенсаторный trunk strategy.");
    if (!issues.length) issues.push("Грубых отклонений по базовым метрикам MVP не найдено, но нужен клинический контекст и side-by-side review.");

    const recommendations = [];
    if (avgKnee < 8) recommendations.push("Проверить достаточность knee flexion in swing: сила hip flexors, rectus femoris stiffness, timing toe-off.");
    if (avgAnkle < -2) recommendations.push("Проверить dorsiflexion range, контроль tibialis anterior, heel rocker, необходимость/настройку AFO.");
    if (Math.abs(avgPelvis) > 4) recommendations.push("Проверить hip abductors, compensatory trunk lean, length discrepancy, stability strategy.");
    recommendations.push("Снять повторное видео в стандартизированном протоколе: вид сбоку, 3–5 проходов, полный кадр от таза до стоп.");

    const extraTests = [
      "10-Meter Walk Test",
      "Timed Up and Go (TUG)",
      "Single-leg stance",
      "Passive/active ROM ankle and knee",
      "Manual muscle testing: dorsiflexors, plantarflexors, hip abductors",
      "Видео спереди и сзади для оценки frontal plane deviations",
    ];

    return {
      title: `Gait report: ${name}`,
      avgScore,
      issues,
      recommendations,
      extraTests,
      context,
      generatedAt: new Date().toLocaleString(),
    };
  }

  function exportReport() {
    if (!summary || !overallMetrics) return;

    const lines = [
      `# ${summary.title}`,
      "",
      `Generated: ${summary.generatedAt}`,
      `Clinical context: ${summary.context}`,
      "",
      "## Core metrics",
      `- Match score: ${summary.avgScore}/100`,
      `- Average hip flexion: ${overallMetrics.avgHipFlexion}°`,
      `- Average knee flexion: ${overallMetrics.avgKneeFlexion}°`,
      `- Average ankle dorsiflexion: ${overallMetrics.avgAnkleDorsi}°`,
      `- Average pelvic tilt/drop proxy: ${overallMetrics.avgPelvicDrop}°`,
      `- Dominant detected phase: ${formatPhaseName(overallMetrics.dominantPhase)}`,
      "",
      "## Flagged observations",
      ...summary.issues.map((item) => `- ${item}`),
      "",
      "## What the system suggests",
      ...summary.recommendations.map((item) => `- ${item}`),
      "",
      "## Additional tests to run",
      ...summary.extraTests.map((item) => `- ${item}`),
      "",
      "## Disclaimer",
      "- This MVP provides a heuristic visual analysis and is not a medical diagnosis.",
    ];

    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(videoName || "gait-report").replace(/\.[^/.]+$/, "")}-report.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleVideoLoaded() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = 960;
    canvas.height = 540;
    setStatus("Видео готово. Можно запускать анализ.");
    setResults([]);
    setSummary(null);
    setSelectedFrame(0);
  }

  function handleUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    if (videoUrl?.startsWith("blob:")) {
      URL.revokeObjectURL(videoUrl);
    }
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setVideoName(file.name);
    setSourceMode("upload");
    setStatus("Файл загружен. Дождитесь инициализации видео.");
    setError("");
  }

  async function startCamera() {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        video.muted = true;
        await video.play();
      }
      setSourceMode("camera");
      setStatus("Камера готова. Можно записывать проходку.");
    } catch (e) {
      console.error(e);
      setError("Не удалось открыть камеру. Проверьте разрешения браузера.");
    }
  }

  function startRecording() {
    if (!streamRef.current) return;
    chunksRef.current = [];
    const recorder = new MediaRecorder(streamRef.current, { mimeType: "video/webm" });
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      if (videoUrl?.startsWith("blob:")) {
        URL.revokeObjectURL(videoUrl);
      }
      const url = URL.createObjectURL(blob);
      setVideoUrl(url);
      setVideoName(`recording-${Date.now()}.webm`);
      setStatus("Запись завершена. Можно запускать анализ.");
    };

    recorder.start();
    setIsRecording(true);
    setStatus("Идёт запись. Попросите человека пройти боком через кадр.");
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  }

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => setIsPlaying(false);

    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    video.addEventListener("ended", onEnded);

    return () => {
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
      video.removeEventListener("ended", onEnded);
    };
  }, [videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    function render() {
      if (results.length && isPlaying) {
        const closestIdx = results.reduce(
          (best, frame, idx) => {
            const diff = Math.abs(frame.time - video.currentTime);
            return diff < best.diff ? { idx, diff } : best;
          },
          { idx: 0, diff: Infinity }
        ).idx;
        setSelectedFrame(closestIdx);
        drawFrame(results[closestIdx]);
      } else if (!results.length) {
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext("2d");
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          if (video.readyState >= 2) ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }
      }
      animationRef.current = requestAnimationFrame(render);
    }

    animationRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationRef.current);
  }, [results, isPlaying]);

  useEffect(() => {
    if (currentResult) {
      drawFrame(currentResult);
    }
  }, [selectedFrame, results]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <div className="mx-auto max-w-7xl p-4 md:p-8">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 grid gap-4 lg:grid-cols-[1.3fr_0.7fr]"
        >
          <Card className="rounded-3xl border-slate-800 bg-slate-900/80 shadow-2xl">
            <CardHeader>
              <div className="flex flex-wrap items-center gap-3">
                <Badge className="rounded-full bg-slate-100 text-slate-900 hover:bg-slate-100">
                  Gait Lab MVP
                </Badge>
                <Badge variant="outline" className="rounded-full border-slate-700 text-slate-300">
                  upload + camera + overlay + report
                </Badge>
              </div>
              <CardTitle className="mt-3 text-3xl font-semibold tracking-tight">
                Анализ походки с референсной биомеханикой
              </CardTitle>
              <CardDescription className="max-w-3xl text-slate-400">
                Загрузи или сними видео, получи наложение референсной походки, базовый разбор отклонений и отчёт с гипотезами и дополнительными тестами.
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="rounded-3xl border-slate-800 bg-slate-900/80 shadow-2xl">
            <CardHeader>
              <CardTitle className="text-lg">Статус системы</CardTitle>
              <CardDescription className="text-slate-400">Браузерная обработка, без сервера.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                <span className="text-sm text-slate-300">Модель позы</span>
                {poseReady ? (
                  <span className="flex items-center gap-2 text-sm text-emerald-400"><CheckCircle2 className="h-4 w-4" /> готова</span>
                ) : (
                  <span className="flex items-center gap-2 text-sm text-amber-400"><AlertCircle className="h-4 w-4" /> не загружена</span>
                )}
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm text-slate-300">
                  <span>Прогресс анализа</span>
                  <span>{analysisProgress}%</span>
                </div>
                <Progress value={analysisProgress} className="h-2 bg-slate-800" />
              </div>
              <p className="text-sm leading-6 text-slate-300">{status}</p>
              {error ? <p className="rounded-2xl border border-red-900/50 bg-red-950/40 p-3 text-sm text-red-300">{error}</p> : null}
              <Button
                onClick={ensurePoseLandmarker}
                disabled={loadingPose || poseReady}
                className="w-full rounded-2xl"
              >
                {loadingPose ? "Загрузка модели…" : poseReady ? "Модель загружена" : "Подготовить pose-модель"}
              </Button>
            </CardContent>
          </Card>
        </motion.div>

        <div className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
          <div className="space-y-6">
            <Card className="rounded-3xl border-slate-800 bg-slate-900/80 shadow-2xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-xl"><Video className="h-5 w-5" /> Источник видео</CardTitle>
                <CardDescription className="text-slate-400">
                  Лучше всего работает видео сбоку, в полный рост, при хорошем освещении и контрастном фоне.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="upload" className="w-full">
                  <TabsList className="grid w-full grid-cols-2 rounded-2xl bg-slate-950">
                    <TabsTrigger value="upload" className="rounded-2xl">Upload</TabsTrigger>
                    <TabsTrigger value="camera" className="rounded-2xl">Camera</TabsTrigger>
                  </TabsList>

                  <TabsContent value="upload" className="mt-4 space-y-4">
                    <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-950/50 p-6 text-center">
                      <Upload className="mx-auto mb-3 h-8 w-8 text-slate-400" />
                      <p className="mb-3 text-sm text-slate-300">Загрузи MP4 / MOV / WEBM с проходкой.</p>
                      <Input type="file" accept="video/*" onChange={handleUpload} className="rounded-2xl border-slate-800 bg-slate-900" />
                    </div>
                  </TabsContent>

                  <TabsContent value="camera" className="mt-4 space-y-4">
                    <div className="grid gap-3 sm:grid-cols-3">
                      <Button onClick={startCamera} variant="secondary" className="rounded-2xl">
                        <Camera className="mr-2 h-4 w-4" /> Включить камеру
                      </Button>
                      <Button onClick={startRecording} disabled={!streamRef.current || isRecording} className="rounded-2xl">
                        <Play className="mr-2 h-4 w-4" /> Запись
                      </Button>
                      <Button onClick={stopRecording} disabled={!isRecording} variant="outline" className="rounded-2xl border-slate-700">
                        <Pause className="mr-2 h-4 w-4" /> Стоп
                      </Button>
                    </div>
                    <p className="text-sm text-slate-400">
                      Протокол MVP: камера на уровне таза, вид строго сбоку, пройти 3–5 метров в естественном темпе.
                    </p>
                  </TabsContent>
                </Tabs>

                <div className="mt-6 overflow-hidden rounded-3xl border border-slate-800 bg-black">
                  <div className="relative aspect-video w-full bg-black">
                    <video
                      ref={videoRef}
                      src={sourceMode === "upload" ? videoUrl : undefined}
                      playsInline
                      controls={false}
                      className="hidden"
                      onLoadedMetadata={handleVideoLoaded}
                    />
                    <canvas ref={canvasRef} width={960} height={540} className="h-full w-full" />
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap gap-3">
                  <Button onClick={togglePlayback} disabled={!videoUrl && sourceMode !== "camera"} className="rounded-2xl">
                    {isPlaying ? <Pause className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                    {isPlaying ? "Пауза" : "Проиграть"}
                  </Button>
                  <Button onClick={runAnalysis} disabled={!videoUrl || loadingPose} variant="secondary" className="rounded-2xl">
                    <Activity className="mr-2 h-4 w-4" /> Анализировать видео
                  </Button>
                  <Button onClick={exportReport} disabled={!summary} variant="outline" className="rounded-2xl border-slate-700">
                    <Download className="mr-2 h-4 w-4" /> Скачать отчёт
                  </Button>
                  <Button onClick={() => { setResults([]); setSummary(null); setSelectedFrame(0); setAnalysisProgress(0); }} variant="ghost" className="rounded-2xl">
                    <RefreshCw className="mr-2 h-4 w-4" /> Сбросить анализ
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="rounded-3xl border-slate-800 bg-slate-900/80 shadow-2xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-xl"><Settings2 className="h-5 w-5" /> Клинический контекст</CardTitle>
                <CardDescription className="text-slate-400">
                  Этот блок потом можно расширить до анкеты: диагноз, устройство, сторона, жалобы, ROM, боль, цели пациента.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Textarea
                  value={clinicalContext}
                  onChange={(e) => setClinicalContext(e.target.value)}
                  className="min-h-[120px] rounded-2xl border-slate-800 bg-slate-950 text-slate-100"
                />
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <Card className="rounded-3xl border-slate-800 bg-slate-900/80 shadow-2xl">
              <CardHeader>
                <CardTitle className="text-xl">Ключевой кадр</CardTitle>
                <CardDescription className="text-slate-400">
                  Белый пунктир — референсный паттерн для распознанной фазы. Сплошной скелет — фактическая поза.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {currentResult ? (
                  <>
                    <div className="grid grid-cols-2 gap-3">
                      <MetricCard label="Фаза" value={formatPhaseName(currentResult.phase)} />
                      <MetricCard label="Match" value={`${currentResult.metrics.score}/100`} />
                      <MetricCard label="Hip" value={`${currentResult.metrics.hipFlexion.toFixed(1)}°`} />
                      <MetricCard label="Knee" value={`${currentResult.metrics.kneeFlexion.toFixed(1)}°`} />
                      <MetricCard label="Ankle" value={`${currentResult.metrics.ankleDorsi.toFixed(1)}°`} />
                      <MetricCard label="Pelvis" value={`${currentResult.metrics.pelvicDrop.toFixed(1)}°`} />
                    </div>
                    <div>
                      <label className="mb-2 block text-sm text-slate-400">Кадр анализа</label>
                      <input
                        type="range"
                        min={0}
                        max={Math.max(results.length - 1, 0)}
                        value={selectedFrame}
                        onChange={(e) => setSelectedFrame(Number(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  </>
                ) : (
                  <EmptyState text="После анализа здесь появятся метрики выбранного кадра." />
                )}
              </CardContent>
            </Card>

            <Card className="rounded-3xl border-slate-800 bg-slate-900/80 shadow-2xl">
              <CardHeader>
                <CardTitle className="text-xl">Сводка по сессии</CardTitle>
                <CardDescription className="text-slate-400">Первичная эвристическая аналитика.</CardDescription>
              </CardHeader>
              <CardContent>
                {overallMetrics && summary ? (
                  <div className="space-y-5">
                    <div className="grid grid-cols-2 gap-3">
                      <MetricCard label="Avg score" value={`${summary.avgScore}/100`} />
                      <MetricCard label="Dominant phase" value={formatPhaseName(overallMetrics.dominantPhase)} />
                      <MetricCard label="Avg knee" value={`${overallMetrics.avgKneeFlexion}°`} />
                      <MetricCard label="Avg ankle" value={`${overallMetrics.avgAnkleDorsi}°`} />
                    </div>

                    <SectionList title="Что видит система" items={summary.issues} />
                    <SectionList title="Что предлагает система" items={summary.recommendations} />
                    <SectionList title="Какие доп. тесты сделать" items={summary.extraTests} />
                  </div>
                ) : (
                  <EmptyState text="Здесь появится автоматический отчёт после анализа видео." />
                )}
              </CardContent>
            </Card>

            <Card className="rounded-3xl border-slate-800 bg-slate-900/80 shadow-2xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-xl"><FileText className="h-5 w-5" /> Что дальше в продукте</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-3 text-sm leading-6 text-slate-300">
                  <li>1. Нормативная библиотека по возрасту, скорости и патологии, а не один «идеал».</li>
                  <li>2. Детекция gait events: heel strike, toe off, stance/swing split.</li>
                  <li>3. Сравнение до/после ортеза и side-by-side playback.</li>
                  <li>4. Раздел «гипотезы причин» отдельно от раздела «рекомендованные проверки».</li>
                  <li>5. PDF-отчёт с брендингом клиники и экспортом изображений ключевых фаз.</li>
                  <li>6. Серверный пайплайн с хранением кейсов, пациентов и результатов повторных визитов.</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
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
