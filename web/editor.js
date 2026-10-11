"use strict";
const $ = (id) => document.getElementById(id);
const config = window.FRAMEREAD;
let session = null,
  markers = [],
  activeFrame = null,
  currentFrame = 0,
  busy = false;
let zoom = 0,
  pixelsPerSecond = 1,
  history = [],
  future = [],
  saveTimer = null;
let pollTimer = null,
  showingStill = false,
  dirty = false,
  seekGeneration = 0,
  listFrame = null;
let previewTimer = null,
  previewAbort = null,
  previewPending = false,
  playAttempt = 0,
  playPreparing = false,
  playerRAF = null;
let drawingArea = false,
  drawStart = null,
  roiDraft = null;
let previewCropped = false;
let detailPending = false;
let detailRenderFrame = null, detailAnalysisTimer = null, saveInFlight = false, savePromise = null;
const KIND = {
  opening: "Opening frame",
  pre_change: "Completed outgoing slide",
  slide_change: "Slide change",
  burst_end: "Writing / erasing finished",
  manual: "Manual mark",
};
let uiEvents = [];
function recordUI(action, details = {}) {
  uiEvents.push({
    action,
    frame_idx: currentFrame,
    timestamp_s: session ? currentFrame / session.fps : null,
    video_name: session?.video_name,
    ...details,
  });
  if (uiEvents.length > 100) uiEvents.shift();
}
setInterval(() => {
  if (!uiEvents.length) return;
  const events = uiEvents.splice(0, 100);
  api("/api/ui-events", { events }).catch(() => {});
}, 2000);
window.addEventListener("error", (e) =>
  recordUI("javascript_error", { message: e.message }),
);
window.addEventListener("unhandledrejection", (e) =>
  recordUI("promise_error", { message: String(e.reason) }),
);
const frameURL = (frame, width = 240) =>
  `${config.server}/api/frame?frame=${frame}&width=${width}&video=${encodeURIComponent(session?.folder || "")}&token=${config.token}`;
function fmt(s) {
  const ms = Math.max(0, Math.round(s * 1000));
  return `${String(Math.floor(ms / 3600000)).padStart(2, "0")}:${String(Math.floor(ms / 60000) % 60).padStart(2, "0")}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
}
async function api(path, data) {
  const response = await fetch(config.server + path, {
    method: data === undefined ? "GET" : "POST",
    headers: {
      "X-TransVision-Token": config.token,
      ...(data === undefined ? {} : { "Content-Type": "application/json" }),
    },
    ...(data === undefined ? {} : { body: JSON.stringify(data) }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Request failed.");
  return result;
}
function error(message, kind = "action") {
  $("error").textContent = message;
  $("error").hidden = !message;
  $("error").dataset.kind = kind;
}
function clearPlaybackWarning() {
  if ($("error").dataset.kind === "playback_loading") error("");
}
function guard(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (e) {
      error(e.message);
    }
  };
}
function selected() {
  return markers.filter((m) => m.included);
}
function snapshot() {
  return JSON.stringify(markers);
}
function mutate(fn) {
  if (busy || session?.preflight) return;
  history.push(snapshot());
  if (history.length > 100) history.shift();
  future = [];
  fn();
  markers.sort((a, b) => a.frame_idx - b.frame_idx);
  dirty = true;
  renderMarks();
  scheduleSave();
}
function markAtPlayhead(ai = false) {
  if (!session || previewPending) return;
  mutate(() => {
    let mark = markers.find((m) => m.frame_idx === currentFrame);
    if (!mark) {
      mark = {
        id: `manual-${currentFrame}`,
        frame_idx: currentFrame,
        timestamp_s: currentFrame / session.fps,
        kind: "manual",
        included: true,
        ai: false,
      };
      markers.push(mark);
    }
    mark.included = true;
    mark.user_override = true;
    if (ai) mark.ai = !mark.ai;
    activeFrame = currentFrame;
  });
}
function toggleAI(frame) {
  mutate(() => {
    const m = markers.find((m) => m.frame_idx === frame);
    m.ai = !m.ai;
    activeFrame = frame;
  });
}
function remove(frame) {
  mutate(() => {
    markers = markers.filter((m) => m.frame_idx !== frame);
    activeFrame = null;
  });
}
function include(frame) {
  mutate(() => {
    const m = markers.find((m) => m.frame_idx === frame);
    m.included = !m.included;
    m.user_override = true;
  });
}
function scheduleSave() {
  clearTimeout(saveTimer);
  $("saveStatus").textContent = "Unsaved changes";
  saveTimer = setTimeout(() => save().catch((e) => error(e.message)), 700);
}
async function save() {
  clearTimeout(saveTimer);
  if (!session || busy || session.preflight) return;
  if (saveInFlight) {
    await savePromise;
    if (dirty) return save();
    return;
  }
  const payload = reviewPayload();
  const version = JSON.stringify(payload);
  saveInFlight = true;
  try {
    savePromise = api("/api/save", payload);
    await savePromise;
    if (JSON.stringify(reviewPayload()) === version) {
      dirty = false;
      $("saveStatus").textContent = "Review saved";
    }
  } finally {
    saveInFlight = false;
    savePromise = null;
  }
}
function seek(frame, scroll = true) {
  if (!session) return;
  currentFrame = Math.max(
    session.preflight ? 0 : session.trim_start_frame || 0,
    Math.min(
      session.preflight
        ? session.frame_count - 1
        : (session.trim_end_frame || session.frame_count) - 1,
      Math.round(frame),
    ),
  );
  const t = currentFrame / session.fps;
  recordUI("seek");
  const previewStarted = performance.now();
  playAttempt++;
  playPreparing = false;
  $("play").disabled = false;
  $("player").pause();
  $("play").textContent = "Play";
  // A server-decoded still is used for paused inspection and frame-accurate manual marks.
  showingStill = true;
  $("still").hidden = false;
  $("player").hidden = true;
  const generation = ++seekGeneration;
  clearTimeout(previewTimer);
  previewAbort?.abort();
  previewPending = true;
  $("previewStatus").textContent = "Loading source frame…";
  $("still").style.opacity = ".45";
  previewTimer = setTimeout(async () => {
    previewAbort = new AbortController();
    try {
      const response = await fetch(frameURL(currentFrame, 1280), {
        signal: previewAbort.signal,
      });
      if (!response.ok)
        throw new Error(
          "Preview unavailable. Reload the editor if its terminal was restarted.",
        );
      const url = URL.createObjectURL(await response.blob());
      const image = new Image();
      image.onload = () => {
        if (generation === seekGeneration) {
          const previous = $("still").src;
          $("still").src = url;
          $("still").style.opacity = "1";
          previewPending = false;
          $("addMark").disabled = busy;
          $("flagAI").disabled = busy;
          $("previewStatus").textContent = "Source preview";
          clearPlaybackWarning();
          recordUI("preview_ready", {
            seconds: (performance.now() - previewStarted) / 1000,
          });
          if (previous.startsWith("blob:")) URL.revokeObjectURL(previous);
        } else URL.revokeObjectURL(url);
      };
      image.src = url;
    } catch (e) {
      if (e.name !== "AbortError" && generation === seekGeneration) {
        previewPending = false;
        $("previewStatus").textContent = "Preview unavailable";
        error(e.message);
      }
    }
  }, 70);
  activeFrame = markers.some((m) => m.frame_idx === currentFrame)
    ? currentFrame
    : null;
  updatePosition();
  renderActiveMark();
  if (scroll) scrollToTime(t);
}
function updatePosition() {
  if (!session) return;
  const t = currentFrame / session.fps;
  $("timecode").textContent = fmt(t);
  $("frameInfo").textContent =
    `Frame ${currentFrame.toLocaleString()} / ${(session.frame_count - 1).toLocaleString()} · ${session.fps.toFixed(2)} fps`;
  $("scrubber").value = currentFrame;
  $("playhead").style.left = `${t * pixelsPerSecond}px`;
}
function scrollToTime(t) {
  const wrap = $("timelineWrap"),
    x = t * pixelsPerSecond;
  if (x < wrap.scrollLeft + 30 || x > wrap.scrollLeft + wrap.clientWidth - 30)
    wrap.scrollLeft = Math.max(0, x - wrap.clientWidth / 2);
}
function renderMarks() {
  if (!session) return;
  const list = $("markList");
  const keepScroll = list.scrollTop;
  list.replaceChildren();
  const filter = $("filter").value;
  for (const m of markers) {
    if (
      (filter === "included" && !m.included) ||
      (filter === "excluded" && m.included) ||
      (filter === "ai" && !m.ai) ||
      (filter === "changes" && m.duplicate_of !== undefined && !m.included) ||
      (filter === "duplicates" && m.duplicate_of === undefined)
    )
      continue;
    const row = document.createElement("div");
    row.className = `mark-row${m.included ? "" : " excluded"}${activeFrame === m.frame_idx ? " active" : ""}`;
    row.dataset.frame = m.frame_idx;
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = m.included;
    check.disabled = busy;
    check.setAttribute("aria-label", `Include frame ${fmt(m.timestamp_s)}`);
    check.onclick = (e) => {
      e.stopPropagation();
      include(m.frame_idx);
    };
    const thumb = document.createElement("img");
    thumb.src = frameURL(m.frame_idx);
    thumb.loading = "lazy";
    thumb.alt = `Frame at ${fmt(m.timestamp_s)}`;
    const details = document.createElement("div");
    details.className = "details";
    const label = document.createElement("strong");
    label.textContent = fmt(m.timestamp_s);
    const kind = document.createElement("small");
    kind.textContent =
      m.duplicate_of !== undefined
        ? `Similar to ${fmt(m.duplicate_of / session.fps)}`
        : KIND[m.kind] || "Manual mark";
    details.append(label, kind);
    row.append(check, thumb, details);
    if (m.ai) {
      const badge = document.createElement("span");
      badge.className = "ai-badge";
      badge.textContent = "AI";
      row.append(badge);
    }
    row.tabIndex = 0;
    row.onclick = () => seek(m.frame_idx);
    row.onkeydown = (e) => {
      if (e.key === "Enter") seek(m.frame_idx);
    };
    list.append(row);
  }
  list.scrollTop = keepScroll;
  if (!list.children.length) {
    const text = document.createElement("p");
    text.textContent =
      "No marks here. Seek to a helpful frame and press M to add it.";
    text.style.padding = "12px";
    list.append(text);
  }
  const ai = selected().filter((m) => m.ai).length;
  $("counts").textContent = `${selected().length} / ${markers.length}`;
  $("selectionSummary").textContent =
    `${selected().length} frames selected · ${ai} for AI`;
  detailLabels();
  const duplicateCount = markers.filter(
    (m) => m.duplicate_of !== undefined && !m.included,
  ).length;
  $("optimizationSummary").textContent = duplicateCount
    ? `${duplicateCount} similar marks excluded. Choose All marks to restore them.`
    : "Compare the content area to reduce repeated checkpoints.";
  $("undo").disabled = busy || !history.length;
  $("redo").disabled = busy || !future.length;
  $("run").disabled = busy || detailPending || !selected().length;
  $("runPDF").disabled = busy || detailPending || !selected().length;
  renderActiveMark();
}
function renderActiveMark() {
  if (!session) return;
  for (const row of $("markList").querySelectorAll(".mark-row"))
    row.classList.toggle("active", Number(row.dataset.frame) === activeFrame);
  const panel = $("activeMark");
  panel.replaceChildren();
  const active = markers.find((m) => m.frame_idx === activeFrame);
  if (active) {
    const heading = document.createElement("strong");
    heading.textContent = fmt(active.timestamp_s) + (active.ai ? " | AI" : "");
    const actions = document.createElement("div");
    actions.className = "actions";
    for (const [text, action] of [
      [
        active.included ? "Exclude" : "Include",
        () => include(active.frame_idx),
      ],
      [
        active.ai ? "Unflag AI" : "Flag for AI",
        () => toggleAI(active.frame_idx),
      ],
      ["Copy time", () => copyText(fmt(active.timestamp_s))],
      ["Delete mark", () => remove(active.frame_idx)],
    ]) {
      const b = document.createElement("button");
      b.textContent = text;
      b.disabled = busy;
      b.onclick = guard(action);
      actions.append(b);
    }
    panel.append(heading, actions);
  } else
    panel.textContent =
      "M adds the current frame. A adds or toggles its AI flag. You can exclude or delete any automatic mark.";
  renderTimelineMarks();
}
function renderTimelineMarks() {
  if (!session) return;
  const lane = $("markerLane");
  lane.replaceChildren();
  const wrap = $("timelineWrap");
  const left = wrap.scrollLeft - 130,
    right = wrap.scrollLeft + wrap.clientWidth + 130;
  for (const [index, m] of markers.entries()) {
    if (!m.included && !$("showExcluded").checked) continue;
    const x = m.timestamp_s * pixelsPerSecond;
    if (x < left || x > right) continue;
    const b = document.createElement("button");
    b.className = `timeline-mark${m.ai ? " ai" : ""}${m.included ? "" : " excluded"}${m.frame_idx === activeFrame ? " active" : ""}`;
    const nextX =
      index + 1 < markers.length
        ? markers[index + 1].timestamp_s * pixelsPerSecond
        : session.duration * pixelsPerSecond + 100;
    b.style.left = `${Math.max(0, x - 8)}px`;
    b.textContent =
      nextX - x >= 125
        ? (m.ai ? "AI · " : "") + fmt(m.timestamp_s)
        : m.ai
          ? "AI"
          : "●";
    b.title = `${fmt(m.timestamp_s)} — ${KIND[m.kind]}${m.ai ? " | AI" : ""}`;
    b.onclick = () => seek(m.frame_idx);
    lane.append(b);
  }
}
function renderTimeline() {
  if (!session) return;
  const wrap = $("timelineWrap");
  const fit = wrap.clientWidth / session.duration;
  pixelsPerSecond = fit * Math.pow(2, zoom / 10);
  $("timelineTrack").style.width =
    `${Math.max(wrap.clientWidth, session.duration * pixelsPerSecond)}px`;
  $("zoomSlider").value = zoom;
  $("zoomLabel").textContent = `${Math.pow(2, zoom / 10).toFixed(1)}×`;
  renderVisibleTimeline();
  updatePosition();
}
function renderVisibleTimeline() {
  if (!session) return;
  const wrap = $("timelineWrap"),
    left = wrap.scrollLeft,
    right = left + wrap.clientWidth;
  const ruler = $("ruler");
  ruler.replaceChildren();
  const intervals = [
    0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600,
  ];
  const step = intervals.find((s) => s * pixelsPerSecond >= 90) || 3600;
  for (
    let t = Math.floor(left / pixelsPerSecond / step) * step;
    t <= Math.min(session.duration, right / pixelsPerSecond + step);
    t += step
  ) {
    const tick = document.createElement("div");
    tick.className = "tick";
    tick.style.left = `${t * pixelsPerSecond}px`;
    tick.textContent = fmt(t);
    ruler.append(tick);
  }
  // Render only visible thumbnail tiles: zooming a three-hour video does not create thousands of image requests.
  const strip = $("filmstrip"),
    needed = new Set();
  for (
    let i = Math.max(0, Math.floor(left / 133) - 1);
    i <= Math.ceil(right / 133) + 1 &&
    i * 133 < session.duration * pixelsPerSecond;
    i++
  ) {
    const frame = Math.min(
      session.frame_count - 1,
      Math.round(((i * 133 + 66.5) / pixelsPerSecond) * session.fps),
    );
    const key = `${i}-${frame}`;
    needed.add(key);
    if (!strip.querySelector(`[data-key="${key}"]`)) {
      const img = document.createElement("img");
      img.className = "film-thumb";
      img.dataset.key = key;
      img.style.left = `${i * 133}px`;
      img.src = frameURL(frame, 240);
      img.alt = "Filmstrip preview";
      strip.append(img);
    }
  }
  for (const img of [...strip.children])
    if (!needed.has(img.dataset.key)) img.remove();
  renderTimelineMarks();
}
function setZoom(value) {
  zoom = Math.max(0, Math.min(100, value));
  renderTimeline();
  scrollToTime(currentFrame / session.fps);
}
function loadSession(s) {
  clearTimeout(saveTimer);
  clearTimeout(detailAnalysisTimer);
  cancelAnimationFrame(detailRenderFrame);
  const previousFrame =
    session?.folder === s.folder ? currentFrame : s.trim_start_frame || 0;
  if (session?.folder !== s.folder) uiEvents = [];
  session = s;
  markers = structuredClone(s.markers);
  history = [];
  future = [];
  activeFrame = null;
  dirty = false;
  detailPending = false;
  $("saveStatus").textContent = s.preflight
    ? "Video setup · not scanned"
    : "Review saved";
  $("setup").hidden = true;
  $("editor").hidden = false;
  $("videoName").textContent = s.video_name;
  $("scanOptions").hidden = !s.preflight;
  $("scanInterval").value = String(s.sample_seconds ?? 0.5);
  $("scanHideCamera").checked = s.hide_camera !== false;
  $("detailSensitivity").value = TransVisionSelection.sensitivity(s);
  $("detailHideCamera").checked = s.hide_camera !== false;
  detailLabels();
  document.body.classList.toggle("preflight-mode", !!s.preflight);
  $("preflight").hidden = !s.preflight;
  $("applySettings").textContent = s.preflight
    ? "Use this crop & settings"
    : "Apply & check similar marks";
  if (s.preflight) {
    $("trimStart").value = fmt(s.trim_start_s || 0);
    $("trimEnd").value = fmt(s.trim_end_s ?? s.duration);
    trimSummary();
  }
  $("scrubber").min = s.preflight ? 0 : s.trim_start_frame || 0;
  $("scrubber").max = s.preflight
    ? s.frame_count - 1
    : (s.trim_end_frame || s.frame_count) - 1;
  $("durationLabel").textContent = "/ " + fmt(s.duration);
  $("outputHint").textContent = "Only selected frames are exported.";
  $("outputHint").title = `Review saved in ${s.folder}`;
  $("player").src = `${config.server}/video?token=${config.token}`;
  $("pdfTimestamps").checked = s.pdf_timestamps !== false;
  showPDF(s.pdf_result);
  fillSettings();
  zoom = 0;
  renderTimeline();
  resizePreview();
  requestAnimationFrame(renderTimeline);
  seek(previousFrame);
  if (s.result) showResult(s.result);
  else $("results").hidden = true;
  if (!s.preflight && s.analysis_pending) detailChanged();
}
function showPDF(result) {
  $("pdfResults").hidden = !result;
  if (!result) return;
  $("pdfSummary").textContent =
    `${result.pages} cropped pages${result.parts?.length > 1 ? ` · ${result.parts.length} PDFs, at most 100 pages each` : ""} · ${result.timestamps ? "timestamps on each page" : "no page timestamps"} · ${result.elapsed.toFixed(1)} seconds`;
  $("downloadPDF").textContent = result.parts?.length > 1 ? "Download part 1" : "Download PDF";
  $("downloadPDF").download = result.filename;
  $("downloadPDF").href =
    `${config.server}/pdf/${result.filename}?token=${config.token}`;
  $("pdfPartLinks").replaceChildren();
  for (const [index, part] of (result.parts || []).entries()) {
    if (!index) continue;
    const link = document.createElement("a");
    const name = part.file.split(/[\\/]/).at(-1);
    link.className = "button";
    link.textContent = `Download part ${index + 1}`;
    link.download = name;
    link.href = `${config.server}/pdf/${encodeURIComponent(name)}?token=${config.token}`;
    $("pdfPartLinks").append(link);
  }
  $("downloadPageMap").href =
    `${config.server}/pdf/pages.json?token=${config.token}`;
}
function showResult(result) {
  $("results").hidden = false;
  $("resultSummary").textContent =
    `${result.frames} images · ${result.transcript_blocks ?? result.frames} text blocks · ${result.duplicates_skipped ?? 0} repeats skipped · ${result.cached_from ? "reused existing OCR" : result.elapsed.toFixed(1) + " seconds"}`;
  $("resultPath").textContent =
    `${result.folder}\n${result.ocr_check_frames} frames need inspection: empty/low-confidence OCR or ambiguous symbols. OCR confidence does not guarantee accuracy.`;
  $("aiPrompt").value = result.prompt;
  for (const [id, file] of [
    ["downloadText", "transcription.txt"],
    ["downloadBundle", "AI_UPLOAD.zip"],
  ])
    $(id).href = `${config.server}/output/${file}?token=${config.token}`;
}
async function poll() {
  try {
    const state = await api(
      `/api/status?folder=${encodeURIComponent(session?.folder || "")}&revision=${session?.revision ?? -1}`,
    );
    $("testStatus").textContent = state.test_folder
      ? "Recording locally"
      : "Starts when you choose a video";
    $("testPath").textContent = state.test_folder || "";
    busy = ["preparing", "detecting", "processing", "optimizing", "pdf_export"].includes(
      state.phase,
    );
    $("phaseLabel").textContent = {
      idle: "Ready",
      preparing: "Loading video",
      setup: "Video setup",
      detecting:
        state.stage === "selection_analysis"
          ? "Reviewing selection"
          : "Detecting",
      review: "Review",
      processing: state.stage === "extracting" ? "Extracting" : "OCR running",
      pdf_export: "Creating PDF",
      optimizing: "Comparing",
      done: "Finished",
    }[state.phase];
    $("statusText").textContent = state.message;
    $("stopJob").disabled = !busy || !!state.stopping;
    $("stopJob").textContent = state.stopping ? "Stopping…" : "Stop";
    $("stopPreview").disabled = busy;
    $("progress").value = state.progress;
    if (state.ocr_engines)
      for (const option of $("ocrEngine").options)
        option.disabled = !state.ocr_engines.includes(option.value);
    if (state.error) error(state.error);
    if (state.phase === "idle" && !session) $("setup").hidden = false;
    if (
      state.session &&
      !["detecting", "optimizing"].includes(state.phase) &&
      (!dirty || !session || session.folder !== state.session.folder) &&
      (!session ||
        session.folder !== state.session.folder ||
        session.revision !== state.session.revision)
    )
      loadSession(state.session);
    if (state.phase === "done" && state.session?.result) {
      session.result = state.session.result;
      showResult(session.result);
    }
    for (const id of [
      "detect",
      "browse",
      "newVideo",
      "addMark",
      "flagAI",
      "includeAll",
      "excludeAll",
      "removeExcluded",
      "finalFrames",
      "saveReview",
      "exportSelection",
      "settingsButton",
      "optimizeMarks",
      "applySettings",
    ])
      $(id).disabled = busy;
    for (const id of [
      "startDetection",
      "trimStart",
      "trimEnd",
      "setTrimStart",
      "setTrimEnd",
      "resetTrim",
    ])
      $(id).disabled = busy;
    for (const id of [
      "detailSensitivity",
      "detailHideCamera",
      "scanInterval",
      "scanHideCamera",
    ])
      $(id).disabled = busy;
    for (const button of document.querySelectorAll("[data-sensitivity]"))
      button.disabled = busy;
    $("addMark").disabled = busy || previewPending;
    $("flagAI").disabled = busy || previewPending;
    $("run").disabled = busy || detailPending || !selected().length;
    $("runPDF").disabled = busy || detailPending || !selected().length;
    $("pdfTimestamps").disabled = busy;
    $("undo").disabled = busy || !history.length;
    $("redo").disabled = busy || !future.length;
    if (state.phase === "processing")
      $("saveStatus").textContent = "Processing saved selection";
    if (state.phase === "done" && !dirty)
      $("saveStatus").textContent = "Export complete · Review saved";
    $("openFrames").hidden =
      state.phase === "processing"
        ? !state.frames_ready
        : !(state.frames_ready || state.session?.result);
    $("openFrames").dataset.ready = String(!!state.frames_ready);
    if (busy) {
      const eta = state.eta_seconds;
      const since = Math.max(
        0,
        Date.now() / 1000 - (state.updated_at || Date.now() / 1000),
      );
      const remaining = Math.ceil((eta || 0) - since);
      $("etaLabel").textContent =
        eta == null
          ? "Estimating…"
          : remaining <= 0
            ? "Still working · updating estimate"
            : remaining < 60
              ? `~${remaining} sec remaining`
              : `~${Math.floor(remaining / 60)} min ${remaining % 60} sec remaining`;
    } else $("etaLabel").textContent = "";
    if (!busy && listFrame !== busy) {
      renderMarks();
    }
    listFrame = busy;
  } catch (e) {
    error(
      `Cannot reach TransVision. Keep its terminal open, or resume session.json using the launcher. ${e.message}`,
    );
  }
  pollTimer = setTimeout(poll, busy ? 900 : 1800);
}
$("browse").onclick = guard(async () => {
  $("browse").disabled = true;
  try {
    const r = await api("/api/browse", {});
    if (r.path) $("videoPath").value = r.path;
  } finally {
    $("browse").disabled = false;
  }
});
$("detect").onclick = guard(async () => {
  error("");
  $("detect").disabled = true;
  try {
    await api("/api/prepare", { video: $("videoPath").value });
    busy = true; $("stopJob").disabled = false;
    clearTimeout(pollTimer);
    poll();
  } finally {
    $("detect").disabled = false;
  }
});
function parseTimestamp(value) {
  const m = value.trim().match(/^(\d+):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$/);
  if (!m || +m[2] > 59 || +m[3] > 59)
    throw new Error("Enter HH:MM:SS.mmm, for example 00:10:00.000.");
  return +m[1] * 3600 + +m[2] * 60 + +m[3] + Number("0." + (m[4] || "0"));
}
function trimSummary() {
  try {
    const start = parseTimestamp($("trimStart").value),
      end = parseTimestamp($("trimEnd").value);
    $("trimSummary").textContent =
      `Process ${fmt(Math.max(0, end - start))} · Original-video timestamps ${fmt(start)} to ${fmt(end)}`;
  } catch {
    $("trimSummary").textContent = "Enter start and end as HH:MM:SS.mmm";
  }
}
for (const id of ["trimStart", "trimEnd"]) $(id).oninput = trimSummary;
$("setTrimStart").onclick = () => {
  $("trimStart").value = fmt(currentFrame / session.fps);
  trimSummary();
};
$("setTrimEnd").onclick = () => {
  $("trimEnd").value = fmt((currentFrame + 1) / session.fps);
  trimSummary();
};
$("resetTrim").onclick = () => {
  $("trimStart").value = fmt(0);
  $("trimEnd").value = fmt(session.duration);
  trimSummary();
};
$("startDetection").onclick = guard(async () => {
  error("");
  $("startDetection").disabled = true;
  try {
    await api("/api/setup", {
      ...reviewPayload(),
      start_s: parseTimestamp($("trimStart").value),
      end_s: parseTimestamp($("trimEnd").value),
    });
    await api("/api/start", {});
    busy = true;
    clearTimeout(pollTimer);
    poll();
  } finally {
    $("startDetection").disabled = busy;
  }
});
async function unloadVideo() {
  clearTimeout(saveTimer); clearTimeout(detailAnalysisTimer);
  if (dirty) await save();
  await api("/api/reset", {});
  clearTimeout(previewTimer); previewAbort?.abort(); playAttempt++;
  $("player").pause(); $("player").removeAttribute("src"); $("player").load();
  session = null; markers = []; dirty = false; previewPending = false;
  $("setup").hidden = false; $("editor").hidden = true;
  $("videoName").textContent = "No video selected";
  $("results").hidden = true; $("pdfResults").hidden = true;
  error(""); clearTimeout(pollTimer); poll();
}
$("newVideo").onclick = guard(unloadVideo);
$("stopPreview").onclick = guard(unloadVideo);
$("stopJob").onclick = guard(async () => {
  $("player").pause(); $("stopJob").disabled = true;
  await api("/api/stop", {}); clearTimeout(pollTimer); poll();
});
$("addMark").onclick = () => markAtPlayhead();
$("flagAI").onclick = () => markAtPlayhead(true);
$("undo").onclick = () => {
  if (busy || !history.length) return;
  future.push(snapshot());
  markers = JSON.parse(history.pop());
  dirty = true;
  renderMarks();
  scheduleSave();
};
$("redo").onclick = () => {
  if (busy || !future.length) return;
  history.push(snapshot());
  markers = JSON.parse(future.pop());
  dirty = true;
  renderMarks();
  scheduleSave();
};
$("includeAll").onclick = () =>
  mutate(() =>
    markers.forEach((m) => {
      m.included = true;
      m.user_override = false;
    }),
  );
$("excludeAll").onclick = () => {
  $("filter").value = "all";
  $("showExcluded").checked = true;
  mutate(() =>
    markers.forEach((m) => {
      m.included = false;
      m.user_override = true;
    }),
  );
};
$("filter").onchange = renderMarks;
$("prevMark").onclick = () => {
  const prev = selected()
    .filter((m) => m.frame_idx < currentFrame)
    .at(-1);
  if (prev) seek(prev.frame_idx);
};
$("nextMark").onclick = () => {
  const next = selected().find((m) => m.frame_idx > currentFrame);
  if (next) seek(next.frame_idx);
};
$("showExcluded").onchange = renderTimelineMarks;
$("removeExcluded").onclick = () =>
  mutate(() => {
    markers = markers.filter((m) => m.included);
    recordUI("remove_excluded");
  });
$("finalFrames").onclick = guard(async () => {
  clearTimeout(saveTimer);
  await api("/api/final-frames", reviewPayload());
  dirty = false;
  busy = true;
  clearTimeout(pollTimer);
  poll();
});
$("openTest").onclick = guard(() => api("/api/open-test", {}));
$("saveTestNote").onclick = guard(async () => {
  await api("/api/test-note", {
    note: $("testNote").value,
    frame_idx: currentFrame,
    timestamp_s: session ? currentFrame / session.fps : null,
  });
  $("testNote").value = "";
  $("testNoteStatus").textContent =
    "Observation recorded at " + $("timecode").textContent;
});
$("backFrame").onclick = () => seek(currentFrame - 1);
$("nextFrame").onclick = () => seek(currentFrame + 1);
$("scrubber").oninput = () => seek(Number($("scrubber").value));
$("jump").onclick = guard(() => {
  const match = $("jumpTime")
    .value.trim()
    .match(/^(\d+):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$/);
  if (!match || Number(match[2]) > 59 || Number(match[3]) > 59)
    throw new Error("Enter HH:MM:SS.mmm, for example 00:02:33.300.");
  seek(
    (Number(match[1]) * 3600 +
      Number(match[2]) * 60 +
      Number(match[3]) +
      Number("0." + (match[4] || "0"))) *
      session.fps,
  );
});
$("play").onclick = guard(async () => {
  recordUI("play_pause");
  const video = $("player");
  if (!video.paused) {
    video.pause();
    seek(currentFrame, false);
    return;
  }
  const attempt = ++playAttempt,
    target = currentFrame / session.fps;
  playPreparing = true;
  $("play").disabled = true;
  try {
    error("");
    if (video.readyState < 1) await mediaEvent(video, "loadedmetadata", 12000);
    if (attempt !== playAttempt) return;
    // Ignore old timeupdate events until the requested seek has actually completed.
    if (
      Math.abs(video.currentTime - target) > 0.5 / session.fps ||
      video.seeking
    ) {
      const ready = mediaEvent(video, "seeked", 12000);
      video.currentTime = target;
      await ready;
    }
    if (attempt !== playAttempt) return;
    showingStill = false;
    $("still").hidden = true;
    video.hidden = false;
    clearTimeout(previewTimer);
    previewAbort?.abort();
    seekGeneration++;
    previewPending = false;
    await video.play();
    if (attempt === playAttempt) {
      $("play").textContent = "Pause";
      $("previewStatus").textContent = "Playing source video";
      animatePlayhead();
    }
  } catch (e) {
    if (attempt !== playAttempt || e.name === "AbortError") return;
    seek(currentFrame, false);
    error(
      e.name === "NotSupportedError"
        ? "This browser cannot decode this video. Frame review still works."
        : `Playback could not start: ${e.message}. Try Play again after the preview loads.`,
      e.name === "NotSupportedError" ? "playback_codec" : "playback_loading",
    );
  } finally {
    if (attempt === playAttempt) {
      playPreparing = false;
      $("play").disabled = false;
    }
  }
});
for (const event of ["canplay", "playing", "loadeddata"])
  $("player").addEventListener(event, clearPlaybackWarning);
$("player").addEventListener("timeupdate", syncPlaybackPosition);
$("player").addEventListener("ended", () => {
  $("play").textContent = "Play";
  seek(currentFrame, false);
});
$("previewZoom").onchange = resizePreview;
$("zoomSlider").oninput = () => setZoom(Number($("zoomSlider").value));
$("zoomOut").onclick = () => setZoom(zoom - 10);
$("zoomIn").onclick = () => setZoom(zoom + 10);
$("fitTimeline").onclick = () => setZoom(0);
$("scrollBack").onclick = () =>
  ($("timelineWrap").scrollLeft -= $("timelineWrap").clientWidth * 0.8);
$("scrollForward").onclick = () =>
  ($("timelineWrap").scrollLeft += $("timelineWrap").clientWidth * 0.8);
let scrollQueued = false;
$("timelineWrap").onscroll = () => {
  if (!scrollQueued) {
    scrollQueued = true;
    requestAnimationFrame(() => {
      scrollQueued = false;
      renderVisibleTimeline();
    });
  }
};
$("timelineWrap").addEventListener(
  "wheel",
  (e) => {
    if (!session) return;
    e.preventDefault();
    const wrap = $("timelineWrap");
    if (e.ctrlKey) {
      const local = e.clientX - wrap.getBoundingClientRect().left,
        anchor = (wrap.scrollLeft + local) / pixelsPerSecond;
      zoom = Math.max(0, Math.min(100, zoom + (e.deltaY < 0 ? 5 : -5)));
      renderTimeline();
      wrap.scrollLeft = Math.max(0, anchor * pixelsPerSecond - local);
      recordUI("timeline_zoom", { zoom });
    } else wrap.scrollLeft += e.deltaX || e.deltaY;
  },
  { passive: false },
);
for (const id of ["ruler", "filmstrip"]) {
  let dragging = false;
  const move = (e) => {
    const rect = $("timelineTrack").getBoundingClientRect();
    seek(((e.clientX - rect.left) / pixelsPerSecond) * session.fps, false);
  };
  $(id).onpointerdown = (e) => {
    dragging = true;
    $(id).setPointerCapture(e.pointerId);
    move(e);
  };
  $(id).onpointermove = (e) => {
    if (dragging) move(e);
  };
  $(id).onpointerup = () => (dragging = false);
  $(id).onpointercancel = () => (dragging = false);
}
$("saveReview").onclick = guard(save);
$("exportSelection").onclick = () => {
  const payload = selected().map((m) => ({
    frame_idx: m.frame_idx,
    timestamp_s: m.frame_idx / session.fps,
    timestamp_label: fmt(m.frame_idx / session.fps),
    ai: m.ai,
  }));
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "selected_timestamps.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("run").onclick = guard(async () => {
  error("");
  clearTimeout(saveTimer);
  await api("/api/finalize", reviewPayload());
  dirty = false;
  busy = true;
  renderMarks();
  $("run").disabled = true;
  clearTimeout(pollTimer);
  poll();
});
$("runPDF").onclick = guard(async () => {
  error("");
  clearTimeout(saveTimer);
  await api("/api/pdf", reviewPayload());
  dirty = false;
  busy = true;
  clearTimeout(pollTimer);
  poll();
});
$("pdfTimestamps").onchange = () => {
  if (session) {
    session.pdf_timestamps = $("pdfTimestamps").checked;
    dirty = true;
    scheduleSave();
  }
};
$("openFolder").onclick = guard(() => api("/api/open-output", {}));
$("openFrames").onclick = $("openFolder").onclick;
$("copyPrompt").onclick = guard(async () => {
  try {
    await navigator.clipboard.writeText($("aiPrompt").value);
    $("copyStatus").textContent = "Copied";
  } catch (e) {
    $("aiPrompt").focus();
    $("aiPrompt").select();
    $("copyStatus").textContent = "Press Ctrl+C to copy the selected prompt";
  }
});
document.addEventListener("keydown", (e) => {
  if (
    !session ||
    e.target.closest(
      'input,textarea,select,summary,[contenteditable="true"]',
    ) ||
    e.metaKey ||
    e.altKey ||
    $("settingsDialog").open
  )
    return;
  if (e.ctrlKey) {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      $(e.key === "ArrowLeft" ? "prevMark" : "nextMark").click();
    } else if (e.key.toLowerCase() === "z") {
      e.preventDefault();
      $(e.shiftKey ? "redo" : "undo").click();
    } else if (e.key.toLowerCase() === "y") {
      e.preventDefault();
      $("redo").click();
    }
    return;
  }
  if (["ArrowLeft", "ArrowRight", " ", "m", "M", "a", "A"].includes(e.key))
    e.preventDefault();
  if (e.key === "ArrowLeft")
    seek(currentFrame - (e.shiftKey ? Math.round(session.fps) : 1));
  if (e.key === "ArrowRight")
    seek(currentFrame + (e.shiftKey ? Math.round(session.fps) : 1));
  if (e.key === " ") $("play").click();
  if (e.key.toLowerCase() === "m") markAtPlayhead();
  if (e.key.toLowerCase() === "a") markAtPlayhead(true);
  if (["Delete", "Backspace"].includes(e.key) && activeFrame !== null) {
    e.preventDefault();
    remove(activeFrame);
  }
});
function reviewPayload() {
  return {
    markers,
    content_roi: session?.content_roi,
    ocr_engine: session?.ocr_engine,
    ocr_profile: session?.ocr_profile ?? "prose",
    dedupe_ocr: session?.dedupe_ocr ?? true,
    export_cropped: session?.export_cropped ?? false,
    ocr_second_pass: session?.ocr_second_pass ?? false,
    pdf_timestamps: session?.pdf_timestamps ?? true,
    sensitivity: Number($("detailSensitivity").value),
    hide_camera: session?.preflight
      ? $("scanHideCamera").checked
      : $("detailHideCamera").checked,
    sample_seconds: Number($("scanInterval").value),
  };
}
function detailLabels() {
  $("detailValue").textContent = $("detailSensitivity").value;
  $("detailSensitivity").disabled = busy;
  $("detailCount").textContent = detailPending
    ? "Analyzing available frames…"
    : `${selected().length.toLocaleString()} / ${markers.length.toLocaleString()} frames`;
  for (const button of document.querySelectorAll("[data-sensitivity]"))
    button.setAttribute("aria-pressed", String(Number(button.dataset.sensitivity) === Number($("detailSensitivity").value)));
}
function detailChanged() {
  if (!session || session.preflight || busy) return;
  clearTimeout(saveTimer);
  clearTimeout(detailAnalysisTimer);
  session.sensitivity = Number($("detailSensitivity").value);
  session.hide_camera = $("detailHideCamera").checked;
  try {
    if (session.analysis_pending || JSON.stringify(session.selection_analysis_roi) !== JSON.stringify(session.content_roi))
      throw new Error("Analysis required");
    markers = TransVisionSelection.select(markers, session.sensitivity, session.hide_camera);
    detailPending = false;
    dirty = true;
    detailLabels();
    scheduleSave();
    cancelAnimationFrame(detailRenderFrame);
    detailRenderFrame = requestAnimationFrame(() => renderMarks());
    return;
  } catch {
    // Old checkpoints or a changed crop need analysis once; slider edits use its cache.
  }
  detailPending = true;
  detailLabels();
  $("run").disabled = true;
  $("runPDF").disabled = true;
  detailAnalysisTimer = setTimeout(() => analyzeDetail().catch((e) => {
    busy = false;
    error(e.message);
    $("detailCount").textContent = "Could not analyze frames. Adjust sensitivity to retry.";
  }), 150);
}
$("detailSensitivity").oninput = detailChanged;
$("detailSensitivity").onpointerup = () => {
  const value = Number($("detailSensitivity").value);
  const stop = [0, 50, 100].find((n) => Math.abs(value - n) <= 3);
  if (stop !== undefined && stop !== value) {
    $("detailSensitivity").value = stop;
    detailChanged();
  }
};
for (const button of document.querySelectorAll("[data-sensitivity]"))
  button.onclick = () => { $("detailSensitivity").value = button.dataset.sensitivity; detailChanged(); };
$("detailHideCamera").onchange = detailChanged;
async function analyzeDetail() {
  clearTimeout(saveTimer);
  error("");
  // Serialize with autosave so an older save cannot replace the analyzed result.
  if (saveInFlight) {
    detailAnalysisTimer = setTimeout(() => analyzeDetail().catch((e) => error(e.message)), 100);
    return;
  }
  busy = true;
  detailLabels();
  try {
    await api("/api/detail", reviewPayload());
    dirty = false;
    detailPending = false;
    clearTimeout(pollTimer);
    poll();
  } catch (e) {
    busy = false;
    throw e;
  }
}
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    $("saveStatus").textContent =
      `Copied ${text.includes("\n") ? "marked timestamps" : text}`;
  } catch {
    const box = document.createElement("textarea");
    box.value = text;
    document.body.append(box);
    box.select();
    const copied = document.execCommand("copy");
    box.remove();
    if (!copied)
      throw new Error(
        "Clipboard unavailable. Use the timestamp text with Ctrl+C.",
      );
    $("saveStatus").textContent = "Timestamp copied";
  }
}
$("copyTime").onclick = guard(() => copyText(fmt(currentFrame / session.fps)));
$("copyMarkedTimes").onclick = guard(() =>
  copyText(
    selected()
      .map((m) => fmt(m.timestamp_s) + (m.ai ? " | AI" : ""))
      .join("\n"),
  ),
);
function mediaEvent(video, event, timeout) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener(event, done);
      video.removeEventListener("error", fail);
    };
    const done = () => {
      cleanup();
      resolve();
    };
    const fail = () => {
      cleanup();
      reject(
        new Error("Video unavailable or session expired. Reload the editor."),
      );
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error("Video is still loading"));
    }, timeout);
    video.addEventListener(event, done, { once: true });
    video.addEventListener("error", fail, { once: true });
  });
}
function syncPlaybackPosition() {
  if (showingStill || !session) return;
  const frame = Math.floor($("player").currentTime * session.fps),
    end = session.preflight
      ? session.frame_count
      : session.trim_end_frame || session.frame_count;
  if (frame >= end) {
    seek(end - 1, false);
    return;
  }
  currentFrame = Math.max(
    session.preflight ? 0 : session.trim_start_frame || 0,
    Math.min(end - 1, frame),
  );
  updatePosition();
  scrollToTime(currentFrame / session.fps);
}
function animatePlayhead() {
  cancelAnimationFrame(playerRAF);
  const update = () => {
    if (!showingStill && !$("player").paused && session) {
      syncPlaybackPosition();
      if (!showingStill) playerRAF = requestAnimationFrame(update);
    }
  };
  playerRAF = requestAnimationFrame(update);
}
function resizePreview() {
  if (!session) return;
  const preview = $("preview"),
    z = Number($("previewZoom").value),
    roi = previewCropped
      ? session.content_roi
      : [0, 0, session.width, session.height];
  const [x, y, w, h] = roi,
    scale = Math.min(preview.clientWidth / w, preview.clientHeight / h) * z;
  $("previewContent").style.width = `${Math.floor(w * scale)}px`;
  $("previewContent").style.height = `${Math.floor(h * scale)}px`;
  for (const id of ["still", "player"])
    Object.assign($(id).style, {
      left: `${-x * scale}px`,
      top: `${-y * scale}px`,
      width: `${session.width * scale}px`,
      height: `${session.height * scale}px`,
    });
  drawOutline();
}
function drawOutline(roi = roiDraft || session?.content_roi) {
  if (!session || !roi) return;
  const [x, y, w, h] = roi;
  const overlay = $("contentOverlay");
  overlay.hidden = !$("showArea").checked || previewCropped;
  Object.assign(overlay.style, {
    left: `${(x / session.width) * 100}%`,
    top: `${(y / session.height) * 100}%`,
    width: `${(w / session.width) * 100}%`,
    height: `${(h / session.height) * 100}%`,
  });
}
function fillSettings() {
  if (!session) return;
  const roi = session.content_roi || [0, 0, session.width, session.height];
  ["areaX", "areaY", "areaW", "areaH"].forEach(
    (id, i) => ($(id).value = roi[i]),
  );
  $("ocrEngine").value = session.ocr_engine || "tesseract";
  $("ocrProfile").value = session.ocr_profile || "auto";
  $("dedupeOCR").checked = session.dedupe_ocr !== false;
  $("secondPass").checked = !!session.ocr_second_pass;
  $("exportCropped").checked = !!session.export_cropped;
  roiDraft = null;
  drawOutline();
}
function readArea() {
  return ["areaX", "areaY", "areaW", "areaH"].map((id) =>
    Math.round(Number($(id).value)),
  );
}
$("settingsButton").onclick = () => {
  fillSettings();
  $("settingsDialog").showModal();
};
$("closeSettings").onclick = () => {
  $("settingsDialog").close();
  roiDraft = null;
  drawOutline();
};
$("autoArea").onclick = guard(async () => {
  const frame = currentFrame,
    folder = session.folder;
  $("player").pause();
  $("autoArea").disabled = true;
  try {
    const r = await api("/api/suggest-crop", { frame_idx: frame, folder });
    if (session.folder !== folder) return;
    session.suggested_roi = r.content_roi;
    ["areaX", "areaY", "areaW", "areaH"].forEach(
      (id, i) => ($(id).value = r.content_roi[i]),
    );
    roiDraft = r.content_roi;
    $("cropReference").textContent =
      "Suggested from " + fmt(r.timestamp_s) + ". Apply to confirm this area.";
    drawOutline();
  } finally {
    $("autoArea").disabled = false;
  }
});
$("fullArea").onclick = () => {
  roiDraft = [0, 0, session.width, session.height];
  ["areaX", "areaY", "areaW", "areaH"].forEach(
    (id, i) => ($(id).value = roiDraft[i]),
  );
  drawOutline();
};
$("showArea").onchange = () => drawOutline();
$("drawArea").onclick = () => {
  $("settingsDialog").close();
  previewCropped = false;
  resizePreview();
  drawingArea = true;
  $("areaSelector").hidden = false;
  $("previewStatus").textContent = "Drag a rectangle around the content";
};
const sourcePoint = (e) => {
  const r = $("previewContent").getBoundingClientRect();
  return [
    Math.max(
      0,
      Math.min(
        session.width,
        Math.round(((e.clientX - r.left) / r.width) * session.width),
      ),
    ),
    Math.max(
      0,
      Math.min(
        session.height,
        Math.round(((e.clientY - r.top) / r.height) * session.height),
      ),
    ),
  ];
};
$("areaSelector").onpointerdown = (e) => {
  drawStart = sourcePoint(e);
  $("areaSelector").setPointerCapture(e.pointerId);
};
$("areaSelector").onpointermove = (e) => {
  if (!drawStart) return;
  const p = sourcePoint(e);
  roiDraft = [
    Math.min(p[0], drawStart[0]),
    Math.min(p[1], drawStart[1]),
    Math.abs(p[0] - drawStart[0]),
    Math.abs(p[1] - drawStart[1]),
  ];
  drawOutline();
};
$("areaSelector").onpointerup = () => {
  drawingArea = false;
  drawStart = null;
  $("areaSelector").hidden = true;
  if (roiDraft)
    ["areaX", "areaY", "areaW", "areaH"].forEach(
      (id, i) => ($(id).value = roiDraft[i]),
    );
  $("settingsDialog").showModal();
  $("previewStatus").textContent = "Source preview";
};
async function optimize(payload) {
  clearTimeout(saveTimer);
  error("");
  await api("/api/optimize", payload);
  dirty = false;
  busy = true;
  $("saveStatus").textContent = "Review saved · Comparing";
  clearTimeout(pollTimer);
  poll();
}
$("applySettings").onclick = guard(async () => {
  const payload = {
    ...reviewPayload(),
    content_roi: readArea(),
    ocr_engine: $("ocrEngine").value,
    ocr_profile: $("ocrProfile").value,
    dedupe_ocr: $("dedupeOCR").checked,
    export_cropped: $("exportCropped").checked,
    ocr_second_pass: $("secondPass").checked,
  };
  if (session.preflight) {
    const updated = await api("/api/setup", {
      ...payload,
      start_s: parseTimestamp($("trimStart").value),
      end_s: parseTimestamp($("trimEnd").value),
    });
    Object.assign(session, payload, updated);
    roiDraft = null;
    resizePreview();
    $("saveStatus").textContent = "Crop confirmed for detection";
  } else await optimize(payload);
  $("settingsDialog").close();
});
$("optimizeMarks").onclick = guard(() => optimize(reviewPayload()));
new ResizeObserver(resizePreview).observe($("preview"));
$("focusPreview").onclick = () => {
  document.body.classList.toggle("focus-preview");
  $("focusPreview").textContent = document.body.classList.contains(
    "focus-preview",
  )
    ? "Show filmstrip"
    : "Focus preview";
  resizePreview();
};
$("cropPreview").onclick = () => {
  previewCropped = !previewCropped;
  $("cropPreview").textContent = previewCropped
    ? "Show full frame"
    : "Presentation preview";
  resizePreview();
};
window.addEventListener("resize", () => {
  renderTimeline();
  resizePreview();
});
window.addEventListener("beforeunload", (e) => {
  if (dirty) {
    e.preventDefault();
    e.returnValue = "";
  }
});
poll();

const themeMedia = matchMedia("(prefers-color-scheme: dark)");
let themePreference = localStorage.getItem("transvision-theme") || "system";
function applyTheme() {
  document.documentElement.dataset.theme =
    themePreference === "system"
      ? themeMedia.matches
        ? "dark"
        : "light"
      : themePreference;
  $("themeToggle").textContent = "Theme: " + themePreference;
}
$("themeToggle").onclick = () => {
  themePreference = { system: "light", light: "dark", dark: "system" }[
    themePreference
  ];
  localStorage.setItem("transvision-theme", themePreference);
  applyTheme();
};
themeMedia.addEventListener("change", applyTheme);
applyTheme();

const windowClient = crypto.randomUUID();
function windowPresence(closed = false) {
  return fetch(config.server + "/api/window", { method: "POST", keepalive: true,
    headers: { "Content-Type": "application/json", "X-TransVision-Token": config.token },
    body: JSON.stringify({ client: windowClient, closed }) }).catch(() => {});
}
windowPresence(); setInterval(() => windowPresence(), 15000);
window.addEventListener("pagehide", () => windowPresence(true));
window.addEventListener("pageshow", () => windowPresence());
