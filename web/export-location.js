"use strict";
let exportDialogPending = null;

function showExportPreference(state) {
  $("exportDestination").textContent = state.mode === "ask" || !state.configured
    ? "Export folder: choose before exporting"
    : `Export folder: ${state.folder}`;
  $("exportDestination").title = state.folder;
}

async function editExportPreference(forExport = false) {
  if (exportDialogPending) return null;
  const state = await api("/api/export-settings");
  if (exportDialogPending) return null;
  $("exportPath").value = state.folder;
  $("exportMode").value = state.mode;
  $("confirmExport").textContent = forExport ? "Use folder" : "Save preference";
  $("exportError").hidden = true;
  return new Promise((resolve) => {
    exportDialogPending = resolve;
    $("exportDialog").showModal();
  });
}

async function chooseExportDestination() {
  const state = await api("/api/export-settings");
  return !state.configured || state.mode === "ask"
    ? editExportPreference(true)
    : state.folder;
}

$("exportForm").onsubmit = async (event) => {
  event.preventDefault();
  $("confirmExport").disabled = true;
  try {
    const state = await api("/api/export-settings", {
      folder: $("exportPath").value, mode: $("exportMode").value,
    });
    showExportPreference(state);
    const resolve = exportDialogPending;
    exportDialogPending = null;
    $("exportDialog").close();
    resolve?.(state.folder);
  } catch (e) {
    $("exportError").textContent = e.message;
    $("exportError").hidden = false;
  } finally { $("confirmExport").disabled = false; }
};
$("exportDialog").addEventListener("close", () => {
  exportDialogPending?.(null);
  exportDialogPending = null;
});
$("cancelExport").onclick = () => $("exportDialog").close();
for (const id of ["exportPath", "exportMode"]) {
  $(id).addEventListener("input", () => { $("exportError").hidden = true; });
}
$("browseExport").onclick = async () => {
  $("browseExport").disabled = true;
  try {
    const result = await api("/api/export-folder/browse", {});
    if (result.path) $("exportPath").value = result.path;
  } catch (e) {
    $("exportError").textContent = e.message;
    $("exportError").hidden = false;
  } finally { $("browseExport").disabled = false; }
};
$("exportSettings").onclick = guard(() => editExportPreference());
api("/api/export-settings").then(showExportPreference).catch(() => {});
