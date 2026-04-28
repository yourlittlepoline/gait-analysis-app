import React, { useEffect, useMemo, useRef, useState } from "react";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

const ANALYSIS_MODES = {
  VIDEO: "video",
  PHOTO: "photo",
  BOTH: "both",
};

const VIEWS = {
  FRONT: "front",
  BACK: "back",
  SIDE: "side",
};

const VIEW_LABELS = {
  [VIEWS.FRONT]: "Фото спереди",
  [VIEWS.BACK]: "Фото сзади",
  [VIEWS.SIDE]: "Фото сбоку",
};

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
  ["leftShoulder", "rightShoulder"],
  ["leftShoulder", "leftHip"],
  ["rightShoulder", "rightHip"],
  ["leftHip", "rightHip"],
  ["leftHip", "leftKnee"],
  ["leftKnee", "leftAnkle"],
  ["leftAnkle", "leftHeel"],
  ["leftHeel", "leftFootIndex"],
  ["rightHip", "rightKnee"],
  ["rightKnee", "rightAnkle"],
  ["rightAnkle", "rightHeel"],
  ["rightHeel", "rightFootIndex"],
  ["leftShoulder", "leftElbow"],
  ["leftElbow", "leftWrist"],
  ["rightShoulder", "rightElbow"],
  ["rightElbow", "rightWrist"],
];

const MIN_VISIBILITY = 0.45;
const MAX_VIDEO_FRAME_OPTIONS = 16;

function App() {
  const [screen, setScreen] = useState("mode");
  const [mode, setMode] = useState(null);
  const [poseLandmarker, setPoseLandmarker] = useState(null);
  const [modelStatus, setModelStatus] = useState("Загрузка модели...");
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [videoFrameOptions, setVideoFrameOptions] = useState([]);
  const [selectedFrameIds, setSelectedFrameIds] = useState([]);
  const [photos, setPhotos] = useState({ front: null, back: null, side: null });
  const [photoUrls, setPhotoUrls] = useState({ front: null, back: null, side: null });
  const [analysis, setAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const videoRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
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
          minPoseDetectionConfidence: 0.4,
          minPosePresenceConfidence: 0.4,
          minTrackingConfidence: 0.4,
        });

        if (!cancelled) {
          setPoseLandmarker(landmarker);
          setModelStatus("Модель готова");
        }
      } catch (e) {
        console.error(e);
        if (!cancelled) {
          setModelStatus("Ошибка загрузки модели");
          setError("Не удалось загрузить MediaPipe-модель. Проверь интернет и консоль браузера.");
        }
      }
    }

    loadModel();

    return () => {
      cancelled = true;
    };
  }, []);

  // Не чистим objectURL через useEffect с зависимостями.
  // Иначе React может отозвать фото-URL прямо перед анализом, и фото-анализ зависнет.
  // Чистим URL точечно: при замене файла и при полном сбросе.

  const needsVideo = mode === ANALYSIS_MODES.VIDEO || mode === ANALYSIS_MODES.BOTH;
  const needsPhoto = mode === ANALYSIS_MODES.PHOTO || mode === ANALYSIS_MODES.BOTH;

  const canAnalyze = useMemo(() => {
    if (!poseLandmarker || !mode) return false;

    const videoReady = !needsVideo || selectedFrameIds.length > 0;
    const photoReady = !needsPhoto || Object.values(photos).some(Boolean);

    return videoReady && photoReady;
  }, [poseLandmarker, mode, needsVideo, needsPhoto, selectedFrameIds, photos]);

  function resetAll() {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    Object.values(photoUrls).forEach((url) => url && URL.revokeObjectURL(url));
    videoFrameOptions.forEach((frame) => frame.url && URL.revokeObjectURL(frame.url));

    setScreen("mode");
    setMode(null);
    setVideoFile(null);
    setVideoUrl(null);
    setVideoFrameOptions([]);
    setSelectedFrameIds([]);
    setPhotos({ front: null, back: null, side: null });
    setPhotoUrls({ front: null, back: null, side: null });
    setAnalysis(null);
    setError("");
  }

  function chooseMode(nextMode) {
    setMode(nextMode);
    setAnalysis(null);
    setError("");
    setScreen("upload");
  }

  async function handleVideoUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setVideoFile(file);
    setSelectedFrameIds([]);
    setVideoFrameOptions([]);

    if (videoUrl) URL.revokeObjectURL(videoUrl);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);

    try {
      const frames = await extractVideoFrames(url, MAX_VIDEO_FRAME_OPTIONS);
      setVideoFrameOptions(frames);
    } catch (e) {
      console.error(e);
      setError("Не удалось вытащить кадры из видео. Попробуй другое видео или меньший файл.");
    }
  }

  function handlePhotoUpload(view, event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setPhotos((prev) => ({ ...prev, [view]: file }));

    setPhotoUrls((prev) => {
      if (prev[view]) URL.revokeObjectURL(prev[view]);
      return { ...prev, [view]: URL.createObjectURL(file) };
    });
  }

  function toggleFrame(frameId) {
    setSelectedFrameIds((prev) =>
      prev.includes(frameId) ? prev.filter((id) => id !== frameId) : [...prev, frameId]
    );
  }

  async function runAnalysis() {
    if (!poseLandmarker || !canAnalyze) return;

    setIsAnalyzing(true);
    setError("");
    setScreen("analysis");

    try {
      const videoItems = [];
      const photoItems = [];

      if (needsVideo) {
        const selectedFrames = videoFrameOptions.filter((frame) => selectedFrameIds.includes(frame.id));

        for (const frame of selectedFrames) {
          const result = await analyzeImageSource({
            poseLandmarker,
            sourceUrl: frame.url,
            label: `Видео: кадр ${frame.index + 1}`,
            type: "videoFrame",
            view: "side",
            time: frame.time,
          });
          videoItems.push(result);
        }
      }

      if (needsPhoto) {
        for (const view of Object.values(VIEWS)) {
          const url = photoUrls[view];
          if (!url) continue;

          const result = await analyzeImageSource({
            poseLandmarker,
            sourceUrl: url,
            label: VIEW_LABELS[view],
            type: "photo",
            view,
          });
          photoItems.push(result);
        }
      }

      const summary = buildCombinedSummary(videoItems, photoItems, mode);
      setAnalysis({ mode, videoItems, photoItems, summary });
    } catch (e) {
      console.error(e);
      setError("Анализ сломался. Посмотри консоль браузера: там будет точная ошибка.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-8">
        <header className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Анализ походки и позы</h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-300">
              Приложение не ставит диагноз. Оно ищет визуальные паттерны и дает подсказки, что проверить в ортезе.
            </p>
          </div>
          <div className="rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-300">
            {modelStatus}
          </div>
        </header>

        {error && (
          <div className="mb-6 rounded-2xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-100">
            {error}
          </div>
        )}

        {screen === "mode" && <ModeScreen onChoose={chooseMode} />}

        {screen === "upload" && (
          <UploadScreen
            mode={mode}
            needsVideo={needsVideo}
            needsPhoto={needsPhoto}
            videoRef={videoRef}
            videoUrl={videoUrl}
            videoFile={videoFile}
            videoFrameOptions={videoFrameOptions}
            selectedFrameIds={selectedFrameIds}
            photoUrls={photoUrls}
            onVideoUpload={handleVideoUpload}
            onPhotoUpload={handlePhotoUpload}
            onToggleFrame={toggleFrame}
            onAnalyze={runAnalysis}
            canAnalyze={canAnalyze}
            onBack={() => setScreen("mode")}
          />
        )}

        {screen === "analysis" && (
          <ResultsScreen
            isAnalyzing={isAnalyzing}
            analysis={analysis}
            onBack={() => setScreen("upload")}
            onReset={resetAll}
          />
        )}
      </div>
    </div>
  );
}

function ModeScreen({ onChoose }) {
  return (
    <main className="grid gap-4 md:grid-cols-3">
      <ModeCard
        title="Только видео"
        description="Загружаешь видео, выбираешь стоп-кадры вручную, получаешь анализ каждого кадра и общий вывод по движению."
        button="Анализировать видео"
        onClick={() => onChoose(ANALYSIS_MODES.VIDEO)}
      />
      <ModeCard
        title="Только фото"
        description="Загружаешь фото спереди, сзади и/или сбоку. Приложение анализирует статичную позу и симметрию."
        button="Анализировать фото"
        onClick={() => onChoose(ANALYSIS_MODES.PHOTO)}
      />
      <ModeCard
        title="Видео + фото"
        description="Видео показывает динамику, фото уточняют статику. Итоговый вывод суммирует оба источника."
        button="Совмещенный анализ"
        onClick={() => onChoose(ANALYSIS_MODES.BOTH)}
      />
    </main>
  );
}

function ModeCard({ title, description, button, onClick }) {
  return (
    <section className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-2xl shadow-black/20">
      <h2 className="text-xl font-semibold">{title}</h2>
      <p className="mt-3 min-h-24 text-sm leading-6 text-slate-300">{description}</p>
      <button
        onClick={onClick}
        className="mt-6 w-full rounded-2xl bg-cyan-400 px-4 py-3 font-semibold text-slate-950 transition hover:bg-cyan-300"
      >
        {button}
      </button>
    </section>
  );
}

function UploadScreen({
  mode,
  needsVideo,
  needsPhoto,
  videoRef,
  videoUrl,
  videoFile,
  videoFrameOptions,
  selectedFrameIds,
  photoUrls,
  onVideoUpload,
  onPhotoUpload,
  onToggleFrame,
  onAnalyze,
  canAnalyze,
  onBack,
}) {
  return (
    <main className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold">Загрузка данных</h2>
          <p className="mt-1 text-sm text-slate-300">Режим: {getModeLabel(mode)}</p>
        </div>
        <button onClick={onBack} className="rounded-xl border border-slate-700 px-4 py-2 text-sm hover:bg-slate-800">
          Назад к выбору режима
        </button>
      </div>

      {needsVideo && (
        <section className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
          <h3 className="text-lg font-semibold">1. Видео и выбор кадров</h3>
          <p className="mt-1 text-sm text-slate-300">
            Загрузи видео сбоку. После загрузки выбери несколько стоп-кадров, где хорошо видны стопа, колено, таз и плечи.
          </p>

          <input
            className="mt-4 block w-full rounded-2xl border border-slate-700 bg-slate-950 p-3 text-sm"
            type="file"
            accept="video/*"
            onChange={onVideoUpload}
          />

          {videoUrl && (
            <video ref={videoRef} src={videoUrl} controls className="mt-4 max-h-96 w-full rounded-2xl bg-black object-contain" />
          )}

          {videoFile && videoFrameOptions.length === 0 && (
            <p className="mt-4 text-sm text-slate-400">Достаю кадры из видео...</p>
          )}

          {videoFrameOptions.length > 0 && (
            <div className="mt-5">
              <div className="mb-3 flex items-center justify-between gap-3">
                <p className="text-sm text-slate-300">
                  Выбрано кадров: <span className="font-semibold text-cyan-300">{selectedFrameIds.length}</span>
                </p>
                <p className="text-xs text-slate-500">Совет: выбери 3–6 кадров из разных моментов шага.</p>
              </div>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {videoFrameOptions.map((frame) => {
                  const selected = selectedFrameIds.includes(frame.id);
                  return (
                    <button
                      key={frame.id}
                      onClick={() => onToggleFrame(frame.id)}
                      className={`overflow-hidden rounded-2xl border text-left transition ${
                        selected ? "border-cyan-300 bg-cyan-300/10" : "border-slate-800 bg-slate-950 hover:border-slate-600"
                      }`}
                    >
                      <img src={frame.url} alt={`Кадр ${frame.index + 1}`} className="aspect-video w-full object-cover" />
                      <div className="flex items-center justify-between p-3 text-xs">
                        <span>Кадр {frame.index + 1}</span>
                        <span>{frame.time.toFixed(2)} c</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </section>
      )}

      {needsPhoto && (
        <section className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
          <h3 className="text-lg font-semibold">{needsVideo ? "2" : "1"}. Фото</h3>
          <p className="mt-1 text-sm text-slate-300">
            Можно загрузить все три фото, но анализ запустится даже с одним. Лучше всего: спереди, сзади и сбоку в полный рост.
          </p>

          <div className="mt-4 grid gap-4 md:grid-cols-3">
            {Object.values(VIEWS).map((view) => (
              <PhotoUploader key={view} view={view} url={photoUrls[view]} onUpload={onPhotoUpload} />
            ))}
          </div>
        </section>
      )}

      <section className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
        <h3 className="text-lg font-semibold">{needsVideo && needsPhoto ? "3" : "2"}. Запуск анализа</h3>
        <p className="mt-1 text-sm text-slate-300">
          Если загружено видео и фото, приложение объединит признаки: динамику из видео и статичную симметрию из фото.
        </p>
        <button
          disabled={!canAnalyze}
          onClick={onAnalyze}
          className="mt-4 w-full rounded-2xl bg-emerald-400 px-5 py-4 font-bold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        >
          Анализировать
        </button>
      </section>
    </main>
  );
}

function PhotoUploader({ view, url, onUpload }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950 p-4">
      <p className="font-medium">{VIEW_LABELS[view]}</p>
      <input className="mt-3 block w-full text-sm" type="file" accept="image/*" onChange={(e) => onUpload(view, e)} />
      {url ? (
        <img src={url} alt={VIEW_LABELS[view]} className="mt-3 aspect-[3/4] w-full rounded-xl object-cover" />
      ) : (
        <div className="mt-3 flex aspect-[3/4] items-center justify-center rounded-xl border border-dashed border-slate-700 text-sm text-slate-500">
          Нет фото
        </div>
      )}
    </div>
  );
}

function ResultsScreen({ isAnalyzing, analysis, onBack, onReset }) {
  if (isAnalyzing || !analysis) {
    return (
      <main className="rounded-3xl border border-slate-800 bg-slate-900/70 p-8 text-center">
        <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-slate-700 border-t-cyan-300" />
        <h2 className="text-2xl font-bold">Идет анализ</h2>
        <p className="mt-2 text-sm text-slate-300">Сейчас приложение ищет скелет, считает признаки и собирает подсказки.</p>
      </main>
    );
  }

  const allItems = [...analysis.videoItems, ...analysis.photoItems];

  return (
    <main className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold">Результат анализа</h2>
          <p className="mt-1 text-sm text-slate-300">Режим: {getModeLabel(analysis.mode)}</p>
        </div>
        <div className="flex gap-2">
          <button onClick={onBack} className="rounded-xl border border-slate-700 px-4 py-2 text-sm hover:bg-slate-800">
            Назад
          </button>
          <button onClick={onReset} className="rounded-xl bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-300">
            Новый анализ
          </button>
        </div>
      </div>

      <section className="rounded-3xl border border-cyan-400/30 bg-cyan-400/10 p-5">
        <h3 className="text-xl font-bold text-cyan-100">Общий вывод</h3>
        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          <SummaryBlock title="Оценка" items={analysis.summary.zones} />
          <SummaryBlock title="Что отличается от нормы" items={analysis.summary.deviations} />
          <SummaryBlock title="Что проверить / уточнить" items={analysis.summary.nextSteps} />
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-2">
        {allItems.map((item) => (
          <AnalysisCard key={item.id} item={item} />
        ))}
      </section>
    </main>
  );
}

function SummaryBlock({ title, items }) {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-950/70 p-4">
      <h4 className="font-semibold">{title}</h4>
      <ul className="mt-3 space-y-2 text-sm text-slate-300">
        {items?.length ? items.map((item, index) => <li key={index}>• {item}</li>) : <li>Нет выраженных признаков.</li>}
      </ul>
    </div>
  );
}

function ZoneBadge({ zone }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
      <h4 className="font-semibold">Оценка близости к норме</h4>
      <p className="mt-3 text-lg font-bold text-cyan-200">{zoneLabel(zone)}</p>
    </div>
  );
}

function AnalysisCard({ item }) {
  return (
    <article className="overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/70">
      <div className="border-b border-slate-800 p-4">
        <h3 className="text-lg font-semibold">{item.label}</h3>
        <p className="mt-1 text-xs text-slate-400">
          {item.poseFound ? "Скелет найден" : "Скелет не найден"}
          {typeof item.time === "number" ? ` · ${item.time.toFixed(2)} c` : ""}
        </p>
      </div>

      <div className="grid gap-0 md:grid-cols-2">
        <div className="bg-black p-2">
          {item.annotatedUrl ? (
            <img src={item.annotatedUrl} alt={item.label} className="w-full rounded-2xl object-contain" />
          ) : (
            <img src={item.sourceUrl} alt={item.label} className="w-full rounded-2xl object-contain opacity-60" />
          )}
        </div>
        <div className="space-y-4 p-4">
          <ZoneBadge zone={item.zone} />
          <MetricList title="Измерения" metrics={item.metrics} />
          <SummaryBlock title="Что отличается от нормы" items={item.deviations} />
          <SummaryBlock title="Что проверить дополнительно" items={item.checks} />
          <SummaryBlock title="Что уточнить" items={item.questions} />
          <SummaryBlock title="Команды / фокус" items={item.cues} />
          <SummaryBlock title="Ограничения кадра" items={item.limitations} />
        </div>
      </div>
    </article>
  );
}

function MetricList({ title, metrics }) {
  const entries = Object.entries(metrics || {}).filter(([, value]) => value !== null && value !== undefined && Number.isFinite(value));

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
      <h4 className="font-semibold">{title}</h4>
      {entries.length ? (
        <dl className="mt-3 grid grid-cols-2 gap-2 text-sm">
          {entries.map(([key, value]) => (
            <React.Fragment key={key}>
              <dt className="text-slate-400">{metricLabel(key)}</dt>
              <dd className="text-right font-medium text-slate-100">{value.toFixed(1)}°</dd>
            </React.Fragment>
          ))}
        </dl>
      ) : (
        <p className="mt-3 text-sm text-slate-400">Недостаточно точек.</p>
      )}
    </div>
  );
}

async function extractVideoFrames(videoUrl, count) {
  const video = document.createElement("video");
  video.src = videoUrl;
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = "anonymous";

  await waitForEvent(video, "loadedmetadata");

  const duration = video.duration;
  const safeCount = Math.max(1, Math.min(count, Math.floor(duration * 2) || count));
  const times = Array.from({ length: safeCount }, (_, i) => {
    const padding = duration * 0.05;
    const start = Number.isFinite(padding) ? padding : 0;
    const end = Number.isFinite(duration - padding) ? duration - padding : duration;
    if (safeCount === 1) return duration / 2;
    return start + ((end - start) * i) / (safeCount - 1);
  });

  const frames = [];

  for (let i = 0; i < times.length; i++) {
    const time = times[i];
    video.currentTime = time;
    await waitForEvent(video, "seeked");

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await canvasToBlob(canvas);
    frames.push({
      id: `frame-${i}-${time}`,
      index: i,
      time,
      blob,
      url: URL.createObjectURL(blob),
    });
  }

  return frames;
}

async function analyzeImageSource({ poseLandmarker, sourceUrl, label, type, view, time }) {
  let image;

  try {
    image = await loadImage(sourceUrl);
  } catch (e) {
    return {
      id: `${type}-${view}-${time ?? label}`,
      label,
      type,
      view,
      time,
      sourceUrl,
      annotatedUrl: null,
      poseFound: false,
      metrics: {},
      hints: ["Не удалось загрузить изображение для анализа. Попробуй загрузить фото заново."],
      limitations: ["Файл не прочитан браузером, поэтому MediaPipe не запускался."],
    };
  }

  const result = poseLandmarker.detect(image);
  const landmarks = result.landmarks?.[0] || null;

  if (!landmarks) {
    return {
      id: `${type}-${view}-${time ?? label}`,
      label,
      type,
      view,
      time,
      sourceUrl,
      annotatedUrl: null,
      poseFound: false,
      metrics: {},
      hints: ["MediaPipe не нашел тело. Нужен кадр в полный рост, лучше с контрастным фоном."],
      limitations: ["Нет скелета — углы и симметрию считать нельзя."],
    };
  }

  const metrics = computeMetrics(landmarks, view);
  const structured = buildStructuredFindings(metrics, view, type);
  const limitations = buildLimitations(landmarks, metrics, view);
  const annotatedUrl = drawAnnotatedImage(image, landmarks, metrics);

  return {
    id: `${type}-${view}-${time ?? label}`,
    label,
    type,
    view,
    time,
    sourceUrl,
    annotatedUrl,
    poseFound: true,
    metrics,
    zone: structured.zone,
    deviations: structured.deviations,
    checks: structured.checks,
    questions: structured.questions,
    cues: structured.cues,
    hints: flattenStructuredFindings(structured),
    limitations,
  };
}

function computeMetrics(landmarks, view) {
  const l = getPointGetter(landmarks);

  const leftKneeAngle = angleAt(l("leftHip"), l("leftKnee"), l("leftAnkle"));
  const rightKneeAngle = angleAt(l("rightHip"), l("rightKnee"), l("rightAnkle"));

  const leftAnkleAngle = angleAt(l("leftKnee"), l("leftAnkle"), l("leftFootIndex"));
  const rightAnkleAngle = angleAt(l("rightKnee"), l("rightAnkle"), l("rightFootIndex"));

  const leftFootPitch = lineAngle(l("leftHeel"), l("leftFootIndex"));
  const rightFootPitch = lineAngle(l("rightHeel"), l("rightFootIndex"));

  const shoulderTilt = horizontalTilt(l("leftShoulder"), l("rightShoulder"));
  const pelvisTilt = horizontalTilt(l("leftHip"), l("rightHip"));
  const leftLegAxis = lineAngle(l("leftHip"), l("leftAnkle"));
  const rightLegAxis = lineAngle(l("rightHip"), l("rightAnkle"));
  const leftKneeValgus = frontalKneeOffset(l("leftHip"), l("leftKnee"), l("leftAnkle"));
  const rightKneeValgus = frontalKneeOffset(l("rightHip"), l("rightKnee"), l("rightAnkle"));
  const shoulderWidth = distance(l("leftShoulder"), l("rightShoulder"));
  const trunkLean = trunkLeanAngle(l("leftShoulder"), l("rightShoulder"), l("leftHip"), l("rightHip"));
  const headForward = forwardHeadRatio(l("nose"), l("leftShoulder"), l("rightShoulder"), shoulderWidth);

  const metrics = {
    leftKneeAngle,
    rightKneeAngle,
    leftAnkleAngle,
    rightAnkleAngle,
    leftFootPitch,
    rightFootPitch,
    shoulderTilt,
    pelvisTilt,
    leftLegAxis,
    rightLegAxis,
    leftKneeValgus,
    rightKneeValgus,
    trunkLean,
    headForward,
  };

  if (view === VIEWS.SIDE) {
    metrics.kneeFlexionAsymmetry = absDiff(leftKneeAngle, rightKneeAngle);
    metrics.footPitchAsymmetry = absDiff(leftFootPitch, rightFootPitch);
  }

  if (view === VIEWS.FRONT || view === VIEWS.BACK) {
    metrics.frontalKneeAsymmetry = absDiff(leftKneeValgus, rightKneeValgus);
    metrics.pelvisTiltAbs = Math.abs(pelvisTilt ?? 0);
    metrics.shoulderTiltAbs = Math.abs(shoulderTilt ?? 0);
  }

  if (view === VIEWS.SIDE) {
    metrics.postureQuality = estimatePostureQuality(metrics);
  }

  return metrics;
}

function buildStructuredFindings(metrics, view, type) {
  const deviations = [];
  const checks = [];
  const questions = [];
  const cues = [];
  let score = 0;

  const kneeMin = Math.min(validNumber(metrics.leftKneeAngle), validNumber(metrics.rightKneeAngle));
  const ankleAsymmetry = metrics.footPitchAsymmetry;
  const pelvisTiltAbs = metrics.pelvisTiltAbs;
  const shoulderTiltAbs = metrics.shoulderTiltAbs;
  const frontalKneeAsymmetry = metrics.frontalKneeAsymmetry;

  if (type === "videoFrame") {
    if (Number.isFinite(kneeMin) && kneeMin < 145) {
      score += 2;
      deviations.push("колено заметно сгибается и не удерживается близко к прямому положению");
      checks.push("проверить контроль колена в опоре");
      questions.push("есть ли ощущение, что колено подламывается или проваливается");
      cues.push("удерживать ногу прямее");
    }

    if (Number.isFinite(ankleAsymmetry) && ankleAsymmetry > 18) {
      score += 2;
      deviations.push("стопы двигаются несимметрично");
      checks.push("проверить клиренс носка и контроль стопы при переносе ноги");
      questions.push("цепляется ли носком за пол");
      cues.push("поднимать носок выше");
    }
  }

  if (view === VIEWS.SIDE) {
    if (Number.isFinite(metrics.trunkLean) && Math.abs(metrics.trunkLean) > 8) {
      score += Math.abs(metrics.trunkLean) > 14 ? 2 : 1;
      deviations.push("корпус заметно наклоняется относительно вертикали");
      checks.push("оценить наклон корпуса во время шага и возможную компенсацию балансом");
      questions.push("становится ли сложнее идти ровно при усталости");
      cues.push("держать корпус ровнее");
    }

    if (Number.isFinite(metrics.headForward) && Math.abs(metrics.headForward) > 0.22) {
      score += 1;
      deviations.push("голова выведена вперед относительно плеч");
      checks.push("посмотреть положение головы, шеи и верхней части спины");
      cues.push("смотреть вперед и не заваливать голову");
    }

    if (Number.isFinite(metrics.postureQuality) && metrics.postureQuality < 65) {
      score += 1;
      deviations.push("осанка на боковом кадре отличается от нейтральной");
      checks.push("оценить грудной отдел, положение головы и наклон корпуса");
    }

    if (Number.isFinite(metrics.leftFootPitch) && metrics.leftFootPitch < -10) {
      score += 2;
      deviations.push("левая стопа опускается носком вниз");
      checks.push("проверить клиренс левого носка");
      questions.push("цепляется ли левый носок за пол");
      cues.push("поднимать левый носок выше");
    }

    if (Number.isFinite(metrics.rightFootPitch) && metrics.rightFootPitch < -10) {
      score += 2;
      deviations.push("правая стопа опускается носком вниз");
      checks.push("проверить клиренс правого носка");
      questions.push("цепляется ли правый носок за пол");
      cues.push("поднимать правый носок выше");
    }
  }

  if (view === VIEWS.FRONT || view === VIEWS.BACK) {
    if (Number.isFinite(frontalKneeAsymmetry) && frontalKneeAsymmetry > 0.08) {
      score += frontalKneeAsymmetry > 0.14 ? 2 : 1;
      deviations.push("колени стоят или двигаются несимметрично во фронтальной плоскости");
      checks.push("проверить положение коленей относительно стоп и таза");
      questions.push("есть ли боль или дискомфорт в колене, стопе или бедре");
      cues.push("удерживать колени ровнее");
    }

    if (Number.isFinite(pelvisTiltAbs) && pelvisTiltAbs > 4) {
      score += pelvisTiltAbs > 8 ? 2 : 1;
      deviations.push("таз расположен не параллельно полу");
      checks.push("оценить перекос таза, симметрию опоры и распределение веса");
      questions.push("есть ли ощущение, что на одну ногу опираться легче");
      cues.push("распределить вес на обе ноги");
    }

    if (Number.isFinite(shoulderTiltAbs) && shoulderTiltAbs > 5) {
      score += shoulderTiltAbs > 10 ? 2 : 1;
      deviations.push("плечи расположены не параллельно полу");
      checks.push("оценить компенсацию корпусом и положение плечевого пояса");
      cues.push("держать плечи ровнее");
    }
  }

  if (!deviations.length) {
    deviations.push("выраженных отличий от нормы по выбранному кадру не видно");
    checks.push("проверить качество кадра и при необходимости выбрать другой момент шага");
  }

  return {
    zone: scoreToZone(score),
    deviations: unique(deviations),
    checks: unique(checks),
    questions: unique(questions),
    cues: unique(cues),
  };
}

function flattenStructuredFindings(structured) {
  return [
    ...structured.deviations,
    ...structured.checks,
    ...structured.questions,
    ...structured.cues,
  ];
}

function scoreToZone(score) {
  if (score <= 0) return "normal";
  if (score <= 2) return "mild";
  if (score <= 5) return "moderate";
  return "severe";
}

function zoneLabel(zone) {
  if (zone === "normal") return "Близко к норме";
  if (zone === "mild") return "Есть отклонения";
  if (zone === "moderate") return "Заметные отклонения";
  if (zone === "severe") return "Сильно отличается от нормы";
  return "Нет оценки";
}

function buildLimitations(landmarks, metrics, view) {
  const limitations = [];
  const lowVisibility = importantLandmarks(view).filter((name) => {
    const point = landmarks[LANDMARKS[name]];
    return !point || (point.visibility ?? 1) < MIN_VISIBILITY;
  });

  if (lowVisibility.length) {
    limitations.push(`Плохо видны точки: ${lowVisibility.map(metricLabel).join(", ")}. Анализ может ошибаться.`);
  }

  if (view === VIEWS.SIDE) {
    limitations.push("На боковом кадре фронтальный вальгус/варус считается плохо. Для этого нужны фото спереди/сзади.");
  }

  if (view === VIEWS.FRONT || view === VIEWS.BACK) {
    limitations.push("На фото спереди/сзади сгибание колена и голеностопа по сагиттали видно ограниченно. Для этого лучше видео сбоку.");
  }

  if (!Object.values(metrics).some((value) => Number.isFinite(value))) {
    limitations.push("Точек недостаточно для расчетов.");
  }

  return limitations;
}

function buildCombinedSummary(videoItems, photoItems, mode) {
  const allItems = [...videoItems, ...photoItems];
  const analyzedItems = allItems.filter((item) => item.poseFound);

  const zones = [];
  const deviations = [];
  const nextSteps = [];

  const videoPoseFound = videoItems.filter((item) => item.poseFound).length;
  const photoPoseFound = photoItems.filter((item) => item.poseFound).length;

  if (mode === ANALYSIS_MODES.BOTH) {
    zones.push(`Проанализировано: видео-кадров — ${videoPoseFound}, фото — ${photoPoseFound}.`);
  } else if (mode === ANALYSIS_MODES.VIDEO) {
    zones.push(`Проанализировано видео-кадров: ${videoPoseFound}.`);
  } else {
    zones.push(`Проанализировано фото: ${photoPoseFound}.`);
  }

  const worstZone = getWorstZone(analyzedItems.map((item) => item.zone));
  zones.push(zoneLabel(worstZone));

  analyzedItems.forEach((item) => {
    item.deviations?.forEach((text) => deviations.push(`${item.label}: ${text}`));
    item.checks?.forEach((text) => nextSteps.push(`Проверить: ${text}`));
    item.questions?.forEach((text) => nextSteps.push(`Уточнить: ${text}`));
  });

  if (!analyzedItems.length) {
    deviations.push("Не удалось получить надежный анализ: скелет не найден или файл не прочитан.");
    nextSteps.push("Загрузить кадр в полный рост с хорошим светом и видимыми стопами, коленями, тазом и плечами.");
  }

  if (!deviations.length) {
    deviations.push("Выраженных отличий от нормы по выбранным кадрам не видно.");
  }

  if (!nextSteps.length) {
    nextSteps.push("При сомнениях выбрать другой кадр шага или добавить фото спереди, сзади и сбоку.");
  }

  return {
    zones: unique(zones),
    deviations: unique(deviations).slice(0, 8),
    nextSteps: unique(nextSteps).slice(0, 10),
  };
}

function getWorstZone(zones) {
  const order = ["normal", "mild", "moderate", "severe"];
  if (!zones.length) return "unknown";

  return zones.reduce((worst, zone) => {
    return order.indexOf(zone) > order.indexOf(worst) ? zone : worst;
  }, "normal");
}

function drawAnnotatedImage(image, landmarks, metrics) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  const ctx = canvas.getContext("2d");

  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(3, canvas.width * 0.004);
  ctx.font = `${Math.max(14, canvas.width * 0.025)}px Arial`;

  for (const [aName, bName] of SKELETON_CONNECTIONS) {
    const a = landmarkToPixel(landmarks[LANDMARKS[aName]], canvas.width, canvas.height);
    const b = landmarkToPixel(landmarks[LANDMARKS[bName]], canvas.width, canvas.height);
    if (!a || !b) continue;

    ctx.strokeStyle = "rgba(34, 211, 238, 0.9)";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (const name of Object.keys(LANDMARKS)) {
    const point = landmarkToPixel(landmarks[LANDMARKS[name]], canvas.width, canvas.height);
    if (!point) continue;

    ctx.fillStyle = "rgba(16, 185, 129, 0.95)";
    ctx.beginPath();
    ctx.arc(point.x, point.y, Math.max(4, canvas.width * 0.006), 0, Math.PI * 2);
    ctx.fill();
  }

  const textLines = [
    formatMetric("L knee", metrics.leftKneeAngle),
    formatMetric("R knee", metrics.rightKneeAngle),
    formatMetric("L foot", metrics.leftFootPitch),
    formatMetric("R foot", metrics.rightFootPitch),
    formatMetric("Pelvis", metrics.pelvisTilt),
  ].filter(Boolean);

  ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
  ctx.fillRect(12, 12, canvas.width * 0.34, 28 + textLines.length * 28);
  ctx.fillStyle = "white";
  textLines.forEach((line, index) => ctx.fillText(line, 24, 46 + index * 28));

  return canvas.toDataURL("image/jpeg", 0.9);
}

function getPointGetter(landmarks) {
  return function getPoint(name) {
    const point = landmarks[LANDMARKS[name]];
    if (!point || (point.visibility ?? 1) < 0.25) return null;
    return point;
  };
}

function landmarkToPixel(point, width, height) {
  if (!point || (point.visibility ?? 1) < 0.25) return null;
  return { x: point.x * width, y: point.y * height };
}

function angleAt(a, b, c) {
  if (!a || !b || !c) return null;

  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magA = Math.hypot(ab.x, ab.y);
  const magC = Math.hypot(cb.x, cb.y);

  if (!magA || !magC) return null;

  const cosine = clamp(dot / (magA * magC), -1, 1);
  return radiansToDegrees(Math.acos(cosine));
}

function lineAngle(a, b) {
  if (!a || !b) return null;
  return radiansToDegrees(Math.atan2(b.y - a.y, b.x - a.x));
}

function horizontalTilt(a, b) {
  if (!a || !b) return null;
  return radiansToDegrees(Math.atan2(b.y - a.y, b.x - a.x));
}

function frontalKneeOffset(hip, knee, ankle) {
  if (!hip || !knee || !ankle) return null;

  const denominator = Math.hypot(ankle.x - hip.x, ankle.y - hip.y);
  if (!denominator) return null;

  const numerator =
    (ankle.x - hip.x) * (hip.y - knee.y) - (hip.x - knee.x) * (ankle.y - hip.y);

  return numerator / denominator;
}

function distance(a, b) {
  if (!a || !b) return null;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpoint(a, b) {
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function trunkLeanAngle(leftShoulder, rightShoulder, leftHip, rightHip) {
  const shoulderMid = midpoint(leftShoulder, rightShoulder);
  const hipMid = midpoint(leftHip, rightHip);
  if (!shoulderMid || !hipMid) return null;

  // 0° примерно вертикально. Плюс/минус — наклон корпуса относительно вертикали кадра.
  return radiansToDegrees(Math.atan2(shoulderMid.x - hipMid.x, hipMid.y - shoulderMid.y));
}

function forwardHeadRatio(nose, leftShoulder, rightShoulder, shoulderWidth) {
  const shoulderMid = midpoint(leftShoulder, rightShoulder);
  if (!nose || !shoulderMid || !shoulderWidth) return null;

  // Нормализуем смещение головы на ширину плеч, чтобы не зависеть от масштаба фото.
  return (nose.x - shoulderMid.x) / shoulderWidth;
}

function estimatePostureQuality(metrics) {
  let score = 100;

  if (Number.isFinite(metrics.trunkLean)) score -= Math.min(30, Math.abs(metrics.trunkLean) * 2.2);
  if (Number.isFinite(metrics.headForward)) score -= Math.min(30, Math.abs(metrics.headForward) * 90);
  if (Number.isFinite(metrics.shoulderTiltAbs)) score -= Math.min(15, metrics.shoulderTiltAbs * 1.5);
  if (Number.isFinite(metrics.pelvisTiltAbs)) score -= Math.min(20, metrics.pelvisTiltAbs * 2);

  return clamp(score, 0, 100);
}

function absDiff(a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.abs(a - b);
}

function validNumber(value) {
  return Number.isFinite(value) ? value : Infinity;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function radiansToDegrees(value) {
  return (value * 180) / Math.PI;
}

function unique(items) {
  return [...new Set(items.filter(Boolean))];
}

function waitForEvent(target, eventName) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`Timeout waiting for ${eventName}`)), 12000);
    target.addEventListener(
      eventName,
      () => {
        clearTimeout(timeout);
        resolve();
      },
      { once: true }
    );
  });
}

function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Canvas toBlob failed"));
    }, "image/jpeg", 0.9);
  });
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const timeout = setTimeout(() => reject(new Error("Image load timeout")), 12000);

    image.onload = () => {
      clearTimeout(timeout);
      resolve(image);
    };

    image.onerror = (e) => {
      clearTimeout(timeout);
      reject(e);
    };

    image.src = url;
  });
}

function importantLandmarks(view) {
  if (view === VIEWS.SIDE) {
    return ["leftHip", "leftKnee", "leftAnkle", "leftHeel", "leftFootIndex", "rightHip", "rightKnee", "rightAnkle", "rightHeel", "rightFootIndex"];
  }

  return ["leftShoulder", "rightShoulder", "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"];
}

function formatMetric(label, value) {
  if (!Number.isFinite(value)) return null;
  return `${label}: ${value.toFixed(0)}°`;
}

function metricLabel(key) {
  const labels = {
    leftKneeAngle: "Левое колено",
    rightKneeAngle: "Правое колено",
    leftAnkleAngle: "Левый голеностоп",
    rightAnkleAngle: "Правый голеностоп",
    leftFootPitch: "Левая стопа",
    rightFootPitch: "Правая стопа",
    shoulderTilt: "Наклон плеч",
    pelvisTilt: "Наклон таза",
    leftLegAxis: "Ось левой ноги",
    rightLegAxis: "Ось правой ноги",
    leftKneeValgus: "Левое колено фронтально",
    rightKneeValgus: "Правое колено фронтально",
    kneeFlexionAsymmetry: "Асимметрия коленей",
    footPitchAsymmetry: "Асимметрия стоп",
    frontalKneeAsymmetry: "Фронтальная асимметрия",
    pelvisTiltAbs: "Перекос таза",
    shoulderTiltAbs: "Перекос плеч",
    trunkLean: "Наклон корпуса",
    headForward: "Голова вперед",
    postureQuality: "Качество осанки",
    leftShoulder: "левое плечо",
    rightShoulder: "правое плечо",
    leftHip: "левый таз",
    rightHip: "правый таз",
    leftKnee: "левое колено",
    rightKnee: "правое колено",
    leftAnkle: "левая лодыжка",
    rightAnkle: "правая лодыжка",
    leftHeel: "левая пятка",
    rightHeel: "правая пятка",
    leftFootIndex: "левый носок",
    rightFootIndex: "правый носок",
  };

  return labels[key] || key;
}

function getModeLabel(mode) {
  if (mode === ANALYSIS_MODES.VIDEO) return "только видео";
  if (mode === ANALYSIS_MODES.PHOTO) return "только фото";
  if (mode === ANALYSIS_MODES.BOTH) return "видео + фото";
  return "не выбран";
}

export default App;
