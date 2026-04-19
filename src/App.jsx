import React, { useEffect, useRef, useState } from "react";
import { Upload, CheckCircle2 } from "lucide-react";

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

const PHASES = {
  loadingResponse: {
    title: "Loading response",
    focus: "приём веса, контакт стопы, колено",
    norm: { hip: 25, knee: 15, ankle: 5, footProgression: 0 },
  },
  midStance: {
    title: "Mid stance",
    focus: "контроль голени и стабильность колена",
    norm: { hip: 0, knee: 5, ankle: 5, footProgression: 0 },
  },
  terminalStance: {
    title: "Terminal stance",
    focus: "tibia progression и push-off",
    norm: { hip: -10, knee: 0, ankle: 10, footProgression: 5 },
  },
  swingClearance: {
    title: "Swing clearance",
    focus: "clearance стопы и сгибание колена",
    norm: { hip: 20, knee: 60, ankle: 0, footProgression: 0 },
  },
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

function signedAngle(a, b) {
  if (!a || !b) return null;
  return radToDeg(Math.atan2(-(b.y - a.y), b.x - a.x));
}

function distance(a, b) {
  if (!a || !b) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function getPoint(landmarks, idx, width, height) {
  const p = landmarks?.[idx];
  if (!p) return null;
  return {
    x: p.x * width,
    y: p.y * height,
    z: p.z ?? 0,
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

function extractLeg(landmarks, side, width, height) {
  const shoulder = getPoint(landmarks, side === "left" ? LANDMARKS.leftShoulder : LANDMARKS.rightShoulder, width, height);
  const hip = getPoint(landmarks, side === "left" ? LANDMARKS.leftHip : LANDMARKS.rightHip, width, height);
  const knee = getPoint(landmarks, side === "left" ? LANDMARKS.leftKnee : LANDMARKS.rightKnee, width, height);
  const ankle = getPoint(landmarks, side === "left" ? LANDMARKS.leftAnkle : LANDMARKS.rightAnkle, width, height);
  const heel = getPoint(landmarks, side === "left" ? LANDMARKS.leftHeel : LANDMARKS.rightHeel, width, height);
  const toe = getPoint(landmarks, side === "left" ? LANDMARKS.leftFootIndex : LANDMARKS.rightFootIndex, width, height);
  const footPoint = heel && toe
    ? { x: (heel.x + toe.x) / 2, y: (heel.y + toe.y) / 2, z: ((heel.z ?? 0) + (toe.z ?? 0)) / 2 }
    : toe || heel;

  const rawKnee = angle3(hip, knee, ankle) ?? 180;
  const kneeFlexion = Math.max(0, 180 - rawKnee);
  const rawAnkle = angle3(knee, ankle, footPoint) ?? 90;
  const ankleAngle = 90 - rawAnkle;
  const hipAngle = signedAngleToVertical(hip, knee) ?? 0;
  const legSize = distance(hip, knee) + distance(knee, ankle) + distance(ankle, footPoint);
  const toeClearance = toe ? ankle.y - toe.y : 0;
  const depthPoints = [hip, knee, ankle, heel, toe].filter(Boolean);
  const meanZ = depthPoints.length ? depthPoints.reduce((s, p) => s + (p.z ?? 0), 0) / depthPoints.length : 0;

  return {
    side,
    points: { shoulder, hip, knee, ankle, heel, toe, footPoint },
    metrics: {
      hip: hipAngle,
      knee: kneeFlexion,
      ankle: ankleAngle,
      toeClearance,
      legSize,
      meanZ,
    },
  };
}

function qualityScore(leg, width, height) {
  if (!leg) return 0;
  const pts = [leg.points.shoulder, leg.points.hip, leg.points.knee, leg.points.ankle, leg.points.footPoint].filter(Boolean);
  if (pts.length < 5) return 0;
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const bodyHeight = Math.max(...ys) - Math.min(...ys);
  const centerX = xs.reduce((a, b) => a + b, 0) / xs.length;
  let score = 0;
  if (bodyHeight > height * 0.35) score += 40;
  if (centerX > width * 0.08 && centerX < width * 0.92) score += 20;
  if (leg.metrics.legSize > height * 0.22) score += 20;
  if (pts.length >= 5) score += 20;
  return score;
}

function chooseNearLeg(landmarks, width, height, previousSide = null) {
  const left = extractLeg(landmarks, "left", width, height);
  const right = extractLeg(landmarks, "right", width, height);
  const leftScore = qualityScore(left, width, height);
  const rightScore = qualityScore(right, width, height);

  if (leftScore < 60 && rightScore < 60) return null;

  let chosen = left.metrics.meanZ <= right.metrics.meanZ ? left : right;

  if (previousSide) {
    const prev = previousSide === "left" ? left : right;
    const alt = previousSide === "left" ? right : left;
    const prevGood = qualityScore(prev, width, height);
    const altGood = qualityScore(alt, width, height);
    if (prevGood >= 60 && Math.abs(prev.metrics.meanZ - alt.metrics.meanZ) < 0.08) {
      chosen = prev;
    } else if (prevGood >= 60 && prev.metrics.meanZ <= alt.metrics.meanZ + 0.03) {
      chosen = prev;
    } else if (altGood >= 60) {
      chosen = alt;
    }
  }

  return chosen;
}

function buildProgressionLine(samples) {
  const valid = samples.filter((s) => s?.points?.hip && s?.points?.footPoint);
  if (valid.length < 2) return null;
  const first = valid[0].points.hip;
  const last = valid[valid.length - 1].points.hip;
  return { start: first, end: last };
}

function progressionAngle(line) {
  if (!line) return 0;
  return signedAngle(line.start, line.end) ?? 0;
}

function footRelativeToProgression(ankle, footPoint, line) {
  if (!ankle || !footPoint) return 0;
  const footAngle = signedAngle(ankle, footPoint) ?? 0;
  return footAngle - progressionAngle(line);
}

function phaseScore(metrics, phaseKey) {
  const norm = PHASES[phaseKey].norm;
  if (phaseKey === "loadingResponse") {
    return Math.abs(metrics.knee - norm.knee) * 1.2 + Math.abs(metrics.footProgression - norm.footProgression) + Math.abs(metrics.ankle - norm.ankle);
  }
  if (phaseKey === "midStance") {
    return Math.abs(metrics.knee - norm.knee) + Math.abs(metrics.ankle - norm.ankle) + Math.abs(metrics.hip - norm.hip) * 0.8;
  }
  if (phaseKey === "terminalStance") {
    return Math.abs(metrics.ankle - norm.ankle) * 1.2 + Math.abs(metrics.hip - norm.hip) + Math.abs(metrics.knee - norm.knee);
  }
  if (phaseKey === "swingClearance") {
    return Math.abs(metrics.knee - norm.knee) * 0.9 + Math.abs(metrics.footProgression - norm.footProgression) + Math.max(0, 10 - metrics.toeClearance) * 2;
  }
  return 999;
}

function footAssessment(metrics, phaseKey) {
  if (phaseKey === "swingClearance") {
    if (metrics.toeClearance < 8) return `Стопа: низкий clearance, риск зацепа · clearance ≈ ${metrics.toeClearance.toFixed(0)} px`;
    if (metrics.footProgression < -10) return `Стопа: стопа свисает вниз в swing · угол ≈ ${metrics.footProgression.toFixed(0)}°`;
    return `Стопа: clearance выглядит приемлемо · clearance ≈ ${metrics.toeClearance.toFixed(0)} px`;
  }
  if (phaseKey === "loadingResponse") {
    if (metrics.footProgression < -12) return `Стопа: выраженная plantarflexed посадка · угол ≈ ${metrics.footProgression.toFixed(0)}°`;
    if (metrics.footProgression > 10) return `Стопа: слишком dorsiflexed контакт · угол ≈ ${metrics.footProgression.toFixed(0)}°`;
    return `Стопа: контакт ближе к ожидаемому · угол ≈ ${metrics.footProgression.toFixed(0)}°`;
  }
  if (phaseKey === "terminalStance") {
    if (metrics.ankle < 4) return `Стопа: мало продвижения над стопой / слабый push-off · угол ≈ ${metrics.footProgression.toFixed(0)}°`;
    return `Стопа: push-off выглядит приемлемо · угол ≈ ${metrics.footProgression.toFixed(0)}°`;
  }
  return `Стопа: оцениваем как опорную стабильность · угол ≈ ${metrics.footProgression.toFixed(0)}°`;
}

function makeText(metrics, phaseKey, side) {
  const ref = PHASES[phaseKey].norm;
  return {
    phaseTitle: PHASES[phaseKey].title,
    focus: PHASES[phaseKey].focus,
    side: side === "left" ? "левая (ближняя)" : "правая (ближняя)",
    hip: `Таз: видео ≈ ${metrics.hip.toFixed(0)}°, норма ${ref.hip}°`,
    knee: `Колено: видео ≈ ${metrics.knee.toFixed(0)}°, норма ${ref.knee}°`,
    ankle: `Голеностоп: видео ≈ ${metrics.ankle.toFixed(0)}°, норма ${ref.ankle}°`,
    foot: footAssessment(metrics, phaseKey),
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
  const [progression, setProgression] = useState(null);

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

    const leg = chooseNearLeg(frame.landmarks, canvas.width, canvas.height, frame.side);
    if (!leg) return;
    const p = leg.points;

    const line = (a, b, color = "#38bdf8", width = 5) => {
      if (!a || !b) return;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.stroke();
    };

    const dot = (a, color = "#38bdf8") => {
      if (!a) return;
      ctx.beginPath();
      ctx.arc(a.x, a.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    };

    if (progression?.start && progression?.end) {
      line(
        { x: 18, y: progression.start.y },
        { x: canvas.width - 18, y: progression.end.y },
        "#ef4444",
        2
      );
    }

    line(p.shoulder, p.hip);
    line(p.hip, p.knee);
    line(p.knee, p.ankle);
    line(p.ankle, p.footPoint);

    [p.shoulder, p.hip, p.knee, p.ankle, p.footPoint].forEach((pt) => dot(pt));

    ctx.fillStyle = "white";
    ctx.font = "700 14px sans-serif";
    if (p.hip) ctx.fillText(`${frame.metrics.hip.toFixed(0)}°`, p.hip.x + 8, p.hip.y - 8);
    if (p.knee) ctx.fillText(`${frame.metrics.knee.toFixed(0)}°`, p.knee.x + 8, p.knee.y - 8);
    if (p.ankle) ctx.fillText(`${frame.metrics.ankle.toFixed(0)}°`, p.ankle.x + 8, p.ankle.y - 8);

    ctx.fillStyle = "rgba(2, 6, 23, 0.8)";
    ctx.fillRect(12, 12, 250, 44);
    ctx.fillStyle = "white";
    ctx.font = "700 14px sans-serif";
    ctx.fillText(`Фаза: ${frame.text.phaseTitle}`, 22, 40);
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
    const checkpoints = Array.from({ length: 18 }, (_, i) => 0.05 + i * 0.05).map((p) =>
      Math.min((video.duration || 1) * p, Math.max((video.duration || 1) - 0.2, 0))
    );

    const tracked = [];
    let lockedSide = null;

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

      const leg = chooseNearLeg(landmarks, width, height, lockedSide);
      if (!leg) continue;
      const score = qualityScore(leg, width, height);
      if (score < 70) continue;

      lockedSide = leg.side;
      tracked.push({
        landmarks,
        side: leg.side,
        points: leg.points,
        metrics: { ...leg.metrics },
        time,
        score,
      });
    }

    if (tracked.length < 4) {
      setError("Не удалось стабильно отследить ближнюю ногу");
      setStatus("Попробуйте видео, где одна сторона тела видна чище и дольше");
      return;
    }

    const prog = buildProgressionLine(tracked);
    setProgression(prog);

    const candidates = tracked.map((item) => {
      const footProgression = footRelativeToProgression(item.points.ankle, item.points.footPoint, prog);
      return {
        ...item,
        metrics: {
          ...item.metrics,
          footProgression,
        },
      };
    });

    const usedTimes = [];
    const takeBestForPhase = (phaseKey) => {
      const sorted = [...candidates]
        .map((c) => ({ ...c, phaseCost: phaseScore(c.metrics, phaseKey) }))
        .sort((a, b) => a.phaseCost - b.phaseCost);

      const chosen = sorted.find((item) => !usedTimes.some((t) => Math.abs(t - item.time) < 0.08)) || sorted[0];
      usedTimes.push(chosen.time);
      return {
        ...chosen,
        text: makeText(chosen.metrics, phaseKey, chosen.side),
      };
    };

    const selected = [
      takeBestForPhase("loadingResponse"),
      takeBestForPhase("midStance"),
      takeBestForPhase("terminalStance"),
      takeBestForPhase("swingClearance"),
    ]
      .sort((a, b) => a.time - b.time)
      .map((f, i) => ({ ...f, step: String(i + 1) }));

    setFrames(selected);
    setSelectedFrame(0);
    setStatus("Готово");

    setTimeout(() => {
      if (selected[0]) drawCurrent(selected[0]);
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
    setProgression(null);
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
              <div>{currentFrame?.text?.phaseTitle}</div>
              <div style={{ color: "#94a3b8" }}>{currentFrame?.text?.focus}</div>
              <div style={{ fontWeight: 700 }}>Нога</div>
              <div>{currentFrame?.text?.side}</div>
              <div style={{ fontWeight: 700 }}>Таз</div>
              <div>{currentFrame?.text?.hip}</div>
              <div style={{ fontWeight: 700 }}>Колено</div>
              <div>{currentFrame?.text?.knee}</div>
              <div style={{ fontWeight: 700 }}>Голеностоп</div>
              <div>{currentFrame?.text?.ankle}</div>
              <div style={{ fontWeight: 700 }}>Стопа</div>
              <div>{currentFrame?.text?.foot}</div>
            </div>
          </div>
        )}

        {error ? <div style={{ color: "#fca5a5" }}>{error}</div> : null}
      </div>
    </div>
  );
}

