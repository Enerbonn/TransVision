"use strict";
let batchPollTimer = null;
function batchError(message) {
  $("batchError").textContent = message;
  $("batchError").hidden = !message;
}
async function refreshBatch() {
  clearTimeout(batchPollTimer);
  try {
    const state = await api("/api/batch");
    const preference = await api("/api/export-settings");
    $("batchDestination").textContent = state.running
      ? `This run: ${state.output_root}`
      : `Next run: ${preference.folder}${preference.mode === "ask" || !preference.configured ? " · choose before starting" : ""}`;
    $("batchExportSettings").disabled = state.running;
    $("batchRows").replaceChildren();
    const counts = {};
    for (const job of state.jobs) {
      counts[job.status] = (counts[job.status] || 0) + 1;
      const row = document.createElement("tr");
      for (const value of [job.name, (job.settings || state.settings).task.toUpperCase(), job.status.replaceAll("_", " "),
        `${Math.round(job.progress || 0)}% · ${job.message}${job.selected_frames != null ? ` · ${job.selected_frames} frames` : ""}`,
        new Date(job.added_at).toLocaleString(), job.completed_at ? new Date(job.completed_at).toLocaleString() : "—"]) {
        const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
      }
      const output = document.createElement("td");
      job.outputs.forEach((file, index) => {
        const link = document.createElement("a");
        link.textContent = file.split(/[\\/]/).at(-1);
        link.href = `${config.server}/api/batch/file?id=${encodeURIComponent(job.id)}&index=${index}&token=${config.token}`;
        link.target = "_blank"; link.rel = "noopener";
        output.append(link, document.createElement("br"));
      });
      if (job.session_path) {
        const review = document.createElement("button"); review.textContent = "Open review";
        review.disabled = state.running || busy;
        review.onclick = () => batchAction(async () => {
          clearTimeout(saveTimer);
          if (dirty) await save();
          await api("/api/batch/review", { id: job.id });
          const status = await api("/api/status");
          loadSession(status.session); renderMarks(); $("batchDialog").close();
        });
        output.append(review);
      }
      row.append(output); $("batchRows").append(row);
    }
    $("batchSummary").textContent = `${state.running ? "Running" : "Stopped"}${state.pause_requested && state.running ? " · will pause after this file" : ""} · ${state.jobs.length} files · ${counts.completed || 0} completed · ${counts.needs_review || 0} awaiting review · ${counts.failed || 0} failed`;
    $("enqueueBatch").disabled = state.running;
    $("startBatch").disabled = state.running || busy || !state.jobs.some((j) => ["queued", "scanning", "exporting"].includes(j.status));
    $("pauseBatch").disabled = !state.running;
    $("stopBatch").disabled = !state.running || state.stop_requested;
    $("stopBatch").textContent = state.running && state.stop_requested ? "Stopping…" : "Stop now";
    $("retryBatch").disabled = state.running || !counts.failed;
  } catch (e) { batchError(e.message); }
  if ($("batchDialog").open) batchPollTimer = setTimeout(refreshBatch, 2000);
}
async function batchAction(fn) {
  batchError("");
  try { await fn(); await refreshBatch(); } catch (e) { batchError(e.message); }
}
$("batchButton").onclick = () => { $("batchDialog").showModal(); refreshBatch(); };
$("closeBatch").onclick = () => { $("batchDialog").close(); clearTimeout(batchPollTimer); };
$("enqueueBatch").onclick = () => batchAction(() => api("/api/batch/enqueue", {
  videos: $("batchPaths").value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean),
  settings: { task: $("batchTask").value, sensitivity: Number($("batchSensitivity").value),
    motion_mode: $("batchMotionMode").value,
    crop_reference_seconds: Number($("batchReference").value), automatic_export: $("batchAuto").checked,
    max_pdf_pages: Number($("batchMaxPages").value),
    pdf_timestamps: $("batchTimestamps").checked, slides_only: $("batchSlides").checked },
}));
$("startBatch").onclick = () => batchAction(async () => {
  const destination = await chooseExportDestination();
  if (destination) await api("/api/batch/start", { export_root: destination });
});
$("batchExportSettings").onclick = () => batchAction(() => editExportPreference());
$("pauseBatch").onclick = () => batchAction(() => api("/api/batch/pause", {}));
$("retryBatch").onclick = () => batchAction(() => api("/api/batch/retry", {}));
$("openBatchFolder").onclick = () => batchAction(() => api("/api/batch/open-folder", {}));

$("stopBatch").onclick = () => batchAction(() => api("/api/batch/stop", {}));
