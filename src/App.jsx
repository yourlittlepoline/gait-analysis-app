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
  leftFootIndex: 31,
};

const PHASE_REF = {
  initialContact: { title: "Initial contact", hip: 30, knee: 5, ankle: 0 },
  loadingResponse: { title: "Loading response", hip: 25, knee: 15, ankle: 5 },
  midStance: { title: "Mid stance", hip: 0, knee: 5, ankle: 5 },
  terminalStance: { title: "Terminal stance", hip: -10, knee: 0, ankle: 10 },
  swing: { title: "Swing", hip: 20, knee: 60, ankle: 0 },
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

function extractMetrics(landmarks, width, height) {
  const leftShoulder = getPoint(landmarks, LANDMARKS.leftShoulder, width, height);
  const rightShoulder = getPoint(landmarks, LANDMARKS.rightShoulder, width, height);
  const leftHip = getPoint(landmarks, LANDMARKS.leftHip, width, height);
  const rightHip = getPoint(landmarks, LANDMARKS.rightHip, width, height);
  const leftKnee = getPoint(landmarks, LANDMARKS.leftKnee, width, height);
  const leftAnkle = getPoint(landmarks, LANDMARKS.leftAnkle, width, height);
  const leftHeel = getPoint(landmarks, LANDMARKS.leftHeel, width, height);
  const leftFootIndex = getPoint(landmarks, LANDMARKS.leftFootIndex, width, height);

  const shoulderCenter = averagePoint([leftShoulder, rightShoulder]);
  const pelvisCenter = averagePoint([leftHip, rightHip]);

  const hip = signedAngleToVertical(leftHip, leftKnee) ?? 0;
  const rawKnee = angle3(leftHip, leftKnee, leftAnkle) ?? 180;
  const knee = Math.max(0, 180 - rawKnee);
  const footPoint = leftFootIndex || leftHeel;
  const rawAnkle = angle3(leftKnee, leftAnkle, footPoint) ?? 90;
  const ankle = 90 - rawAnkle;
  const trunk = shoulderCenter && pelvisCenter ? signedAngleToVertical(shoulderCenter, pelvisCenter) ?? 0 : 0;

  return {
    points: { leftShoulder, leftHip, leftKnee, leftAnkle, leftHeel, leftFootIndex, footPoint },
    metrics: { hip, knee, ankle, trunk },
  };
}

function qualityScore(result, width, height) {
  if (!result) return 0;
  const pts = [result.points.leftShoulder, result.points.leftHip, result.points.leftKnee, result.points.leftAnkle, result.points.footPoint].filter(Boolean);
  if (pts.length < 5) return 0;
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const bodyHeight = Math.max(...ys) - Math.min(...ys);
  const centerX = xs.reduce((a, b) => a + b, 0) / xs.length;
  let score = 0;
  if (bodyHeight > height * 0.35) score += 40;
  if (centerX > width * 0.12 && centerX < width * 0.88) score += 30;
  if (pts.length >= 5) score += 30;
  return score;
}

function estimatePhase(metrics) {
  if (metrics.knee > 35) return "swing";
  if (metrics.ankle > 6 && metrics.knee < 12) return "terminalStance";
  if (metrics.knee > 10 && metrics.knee <= 25) return "loadingResponse";
  if (metrics.hip > 15 && metrics.knee < 15) return "initialContact";
  return "midStance";
}

function comments(metrics, phase) {
  const ref = PHASE_REF[phase] || PHASE_REF.midStance;
  return {
    phaseTitle: ref.title,
    hip: `Таз: видео ${metrics.hip.toFixed(0)}°, норма ${ref.hip}°`,
    knee: `Колено: видео ${metrics.knee.toFixed(0)}°, норма ${ref.knee}°`,
    ankle: `Голеностоп: видео ${metrics.ankle.toFixed(0)}°, норма ${ref.ankle}°`,
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

    const extracted = extractMetrics(frame.landmarks, canvas.width, canvas.height);
    const p = extracted.points;

    const line = (a, b) => {
      if (!a || !b) return;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.lineWidth = 5;
      ctx.strokeStyle = "#38bdf8";
      ctx.stroke();
    };

    const dot = (a) => {
      if (!a) return;
      ctx.beginPath();
      ctx.arc(a.x, a.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#38bdf8";
      ctx.fill();
    };

    line(p.leftShoulder, p.leftHip);
    line(p.leftHip, p.leftKnee);
    line(p.leftKnee, p.leftAnkle);
    line(p.leftAnkle, p.footPoint);

    [p.leftShoulder, p.leftHip, p.leftKnee, p.leftAnkle, p.footPoint].forEach(dot);

    ctx.fillStyle = "white";
    ctx.font = "700 14px sans-serif";
    if (p.leftHip) ctx.fillText(`${frame.metrics.hip.toFixed(0)}°`, p.leftHip.x + 8, p.leftHip.y - 8);
    if (p.leftKnee) ctx.fillText(`${frame.metrics.knee.toFixed(0)}°`, p.leftKnee.x + 8, p.leftKnee.y - 8);
    if (p.leftAnkle) ctx.fillText(`${frame.metrics.ankle.toFixed(0)}°`, p.leftAnkle.x + 8, p.leftAnkle.y - 8);

    ctx.fillStyle = "rgba(2, 6, 23, 0.75)";
    ctx.fillRect(12, 12, 180, 44);
    ctx.fillStyle = "white";
    ctx.font = "700 14px sans-serif";
    ctx.fillText(`Фаза: ${frame.phaseTitle}`, 22, 40);
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

    const model = await ensureModel();
    const width = video.videoWidth || 720;
    const height = video.videoHeight || 1280;

    const checkpoints = [0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88].map((p) =>
      Math.min((video.duration || 1) * p, Math.max((video.duration || 1) - 0.2, 0))
    );

    const candidates = [];

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

      const extracted = extractMetrics(landmarks, width, height);
      const score = qualityScore(extracted, width, height);
      if (score < 70) continue;

      const phase = estimatePhase(extracted.metrics);
      const text = comments(extracted.metrics, phase);

      candidates.push({
        step: "",
        metrics: extracted.metrics,
        landmarks,
        phase,
        phaseTitle: text.phaseTitle,
        text,
        time,
        score,
      });
    }

    if (candidates.length < 2) {
      setError("Не удалось выделить хорошие кадры");
      setStatus("Попробуйте видео, где человек целиком в кадре");
      return;
    }

    const sortedByTime = [...candidates].sort((a, b) => a.time - b.time);
    const first = sortedByTime[0];
    let maxKnee = sortedByTime[0];
    let minKnee = sortedByTime[0];
    let maxHip = sortedByTime[0];

    for (const frame of sortedByTime) {
      if (frame.metrics.knee > maxKnee.metrics.knee) maxKnee = frame;
      if (frame.metrics.knee < minKnee.metrics.knee) minKnee = frame;
      if (frame.metrics.hip > maxHip.metrics.hip) maxHip = frame;
    }

    const picked = [];
    const addUnique = (frame) => {
      if (!frame) return;
      const exists = picked.some((item) => Math.abs(item.time - frame.time) < 0.08);
      if (!exists) picked.push(frame);
    };

    addUnique(first);
    addUnique(minKnee);
    addUnique(maxKnee);
    addUnique(maxHip);

    if (picked.length < 4) {
      for (const frame of sortedByTime) {
        addUnique(frame);
        if (picked.length >= 4) break;
      }
    }

    const limited = picked
      .sort((a, b) => a.time - b.time)
      .slice(0, 4)
      .map((f, i) => ({ ...f, step: String(i + 1) }));

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
    if (currentFrame) {
      showFrame(currentFrame);
    }
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
              <div style={{ fontWeight: 700 }}>Фаза</div>
              <div>{currentFrame?.phaseTitle}</div>
              <div style={{ fontWeight: 700 }}>Таз</div>
              <div>{currentFrame?.text?.hip}</div>
              <div style={{ fontWeight: 700 }}>Колено</div>
              <div>{currentFrame?.text?.knee}</div>
              <div style={{ fontWeight: 700 }}>Голеностоп</div>
              <div>{currentFrame?.text?.ankle}</div>
            </div>
          </div>
        )}

        {error ? <div style={{ color: "#fca5a5" }}>{error}</div> : null}
      </div>
    </div>
  );
}

