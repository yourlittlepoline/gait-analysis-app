import React, { useEffect, useRef, useState } from "react";
import { Upload, CheckCircle2 } from "lucide-react";

const LANDMARKS = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  leftAnkle: 27,
  leftHeel: 29,
  rightHeel: 30,
  leftFootIndex: 31,
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

function estimatePhase(metrics) {
  const { knee, ankle, heelAhead } = metrics;
  if (heelAhead > 16 && knee < 20) return "1";
  if (heelAhead > 8 && knee >= 10 && knee <= 25) return "2";
  if (ankle >= 6 && knee < 12) return "3";
  return "4";
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

  return {
    points: { leftShoulder, leftHip, leftKnee, leftAnkle, leftFootIndex },
    metrics: { hip, knee, ankle, trunk, heelAhead },
  };
}

function qualityScore(result, width, height) {
  if (!result) return 0;
  const pts = Object.values(result.points).filter(Boolean);
  if (pts.length < 5) return 0;
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const bodyHeight = Math.max(...ys) - Math.min(...ys);
  const centerX = xs.reduce((a, b) => a + b, 0) / xs.length;
  let score = 60;
  if (bodyHeight > height * 0.35) score += 20;
  if (centerX > width * 0.15 && centerX < width * 0.85) score += 20;
  return score;
}

function comments(metrics) {
  const out = [];
  if (Math.abs(metrics.hip) > 20) out.push("Таз/бедро: отклонение");
  else out.push("Таз: близко к нейтрали");

  if (metrics.knee > 25) out.push("Колено: больше сгибания");
  else if (metrics.knee < 5) out.push("Колено: мало сгибания");
  else out.push("Колено: умеренно");

  if (metrics.ankle < -5) out.push("Голеностоп: мало тыльного сгибания");
  else if (metrics.ankle > 10) out.push("Голеностоп: много тыльного сгибания");
  else out.push("Голеностоп: умеренно");

  return out;
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
      {frame.phase}
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

    const p = frame.points;
    const line = (a, b) => {
      if (!a || !b) return;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.lineWidth = 4;
      ctx.strokeStyle = "#38bdf8";
      ctx.stroke();
    };

    line(p.leftShoulder, p.leftHip);
    line(p.leftHip, p.leftKnee);
    line(p.leftKnee, p.leftAnkle);
    line(p.leftAnkle, p.leftFootIndex);

    ctx.fillStyle = "white";
    ctx.font = "700 14px sans-serif";
    if (p.leftHip) ctx.fillText(`Таз ${frame.metrics.hip.toFixed(0)}°`, p.leftHip.x + 10, p.leftHip.y - 10);
    if (p.leftKnee) ctx.fillText(`Колено ${frame.metrics.knee.toFixed(0)}°`, p.leftKnee.x + 10, p.leftKnee.y - 10);
    if (p.leftAnkle) ctx.fillText(`Голеностоп ${frame.metrics.ankle.toFixed(0)}°`, p.leftAnkle.x + 10, p.leftAnkle.y - 10);
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

    const model = await ensureModel();
    const width = video.videoWidth || 720;
    const height = video.videoHeight || 1280;
    const times = [0.12, 0.28, 0.44, 0.6, 0.76].map((p) => Math.min((video.duration || 1) * p, Math.max((video.duration || 1) - 0.2, 0)));
    const best = [];

    for (const time of times) {
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
      const extracted = extractMetrics(landmarks, width, height);
      const score = qualityScore(extracted, width, height);
      if (score < 70) continue;

      best.push({
        phase: estimatePhase(extracted.metrics),
        metrics: extracted.metrics,
        points: extracted.points,
        comments: comments(extracted.metrics),
        time,
      });
    }

    if (!best.length) {
      setError("Не удалось выделить хорошие кадры");
      setStatus("Попробуйте видео, где человек целиком в кадре");
      return;
    }

    const limited = best.slice(0, 4).map((f, i) => ({ ...f, phase: String(i + 1) }));
    setFrames(limited);
    setSelectedFrame(0);
    setStatus("Готово");

    setTimeout(() => {
      if (limited[0]) drawCurrent(limited[0]);
    }, 60);
  }

  function onLoadedMetadata() {
    setIsReady(true);
    setStatus("Видео загружено");
  }

  function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    setFrames([]);
    setSelectedFrame(0);
    setError("");
    setStatus("Видео загружается...");
    setTimeout(() => videoRef.current?.load(), 50);
  }

  useEffect(() => {
    if (currentFrame) drawCurrent(currentFrame);
  }, [currentFrame]);

  return (
    <div style={{ minHeight: "100vh", background: "#020617", color: "white", padding: 16, fontFamily: "Inter, system-ui, sans-serif" }}>
      <div style={{ maxWidth: 520, margin: "0 auto", display: "grid", gap: 16 }}>
        {!frames.length ? (
          <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 24, padding: 20, display: "grid", gap: 16 }}>
            <div style={{ fontSize: 28, fontWeight: 700 }}>Анализ походки</div>

            <label style={{ border: "1px dashed #334155", borderRadius: 18, padding: 16, display: "grid", gap: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, color: "#cbd5e1" }}>
                <Upload size={18} /> Загрузка видео
              </div>
              <input type="file" accept="video/*" onChange={onUpload} />
            </label>

            <div style={{ display: "flex", alignItems: "center", gap: 8, color: isReady ? "#86efac" : "#cbd5e1" }}>
              <CheckCircle2 size={18} /> {status}
            </div>

            <button
              onClick={analyze}
              disabled={!videoUrl || isLoadingModel}
              style={{
                background: "white",
                color: "#020617",
                border: 0,
                borderRadius: 16,
                padding: "14px 16px",
                fontWeight: 700,
                cursor: "pointer",
                opacity: !videoUrl || isLoadingModel ? 0.5 : 1,
              }}
            >
              Начать анализ
            </button>

            <video
              ref={videoRef}
              src={videoUrl}
              playsInline
              muted
              preload="metadata"
              onLoadedMetadata={onLoadedMetadata}
              style={{ display: "none" }}
            />
          </div>
        ) : (
          <div style={{ display: "grid", gap: 16 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
              {frames.map((frame, idx) => (
                <FrameCard key={idx} frame={frame} isActive={idx === selectedFrame} onClick={() => setSelectedFrame(idx)} />
              ))}
            </div>

            <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 24, overflow: "hidden" }}>
              <div style={{ position: "relative", width: "100%" }}>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  playsInline
                  muted
                  preload="metadata"
                  controls
                  style={{ width: "100%", display: "block" }}
                />
                <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} />
              </div>
            </div>

            <div style={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 24, padding: 16, display: "grid", gap: 12 }}>
              <div style={{ fontWeight: 700 }}>Таз</div>
              <div>{currentFrame?.comments?.[0]}</div>
              <div style={{ fontWeight: 700 }}>Колено</div>
              <div>{currentFrame?.comments?.[1]}</div>
              <div style={{ fontWeight: 700 }}>Голеностоп</div>
              <div>{currentFrame?.comments?.[2]}</div>
            </div>
          </div>
        )}

        {error ? <div style={{ color: "#fca5a5" }}>{error}</div> : null}
      </div>
    </div>
  );
}
