/* Cached frame selection: keep this curve in sync with src/selection.py. */
"use strict";
const TransVisionSelection = (() => {
  function threshold(value) {
    const middle = 0.012 * 0.12 ** 0.5;
    return value <= 50
      ? 0.12 * (middle / 0.12) ** ((value - 1) / 49)
      : middle * (0.0003 / middle) ** ((value - 50) / 50);
  }
  function sensitivity(session) {
    if (Number.isInteger(session.sensitivity)) return session.sensitivity;
    if (!session.detail_level || session.detail_level === "completed") return 0;
    const old = session.detail_sensitivity ?? 50;
    const cutoff = session.detail_level === "major"
      ? 0.012 * 0.12 ** (old / 100) : 0.003 * 0.1 ** (old / 100);
    const middle = threshold(50);
    const value = cutoff >= middle
      ? 1 + 49 * Math.log(cutoff / 0.12) / Math.log(middle / 0.12)
      : 50 + 50 * Math.log(cutoff / middle) / Math.log(0.0003 / middle);
    return Math.max(1, Math.min(100, Math.round(value)));
  }
  function select(markers, value, hideCamera = true, motionMode = "summary") {
    if (!Number.isInteger(value) || value < 0 || value > 100)
      throw new Error("Sensitivity must be a whole number between 0 and 100.");
    const result = markers.map((m) => ({ ...m }));
    const analysis = (m) => ({...m.visual_analysis, ...m.visual_analysis?.motion?.[motionMode],
      ...(motionMode === "demo" && m.visual_analysis?.motion?.summary?.force_keep ? {force_keep:true,suppressed:false} : {})});
    const groups = new Map();
    for (const mark of result) {
      delete mark.detail_excluded;
      if (mark.user_override || mark.kind === "manual" || mark.ai) continue;
      if (!mark.visual_analysis) throw new Error("Analyze this review first to preview sensitivity.");
      const a = analysis(mark);
      mark.included = false;
      if (a.suppressed) { mark.detail_excluded = a.reason || "repeated content"; continue; }
      if (hideCamera && a.camera) {
        mark.detail_excluded = "camera";
        continue;
      }
      if (!groups.has(a.page_group)) groups.set(a.page_group, []);
      groups.get(a.page_group).push(mark);
    }
    const cutoff = threshold(value);
    for (const group of groups.values()) {
      group.sort((a, b) => a.frame_idx - b.frame_idx);
      const eligible = group.filter((m) => !analysis(m).noise);
      const final = (eligible.length ? eligible : group).at(-1);
      const keep = new Set([final.frame_idx]);
      for (const mark of group) if (analysis(mark).force_keep && value >= (analysis(mark).preserve_min_sensitivity ?? 0)) keep.add(mark.frame_idx);
      if (value) for (const mark of eligible)
        if (value === 100 || (analysis(mark).changed_fraction ?? 0) >= cutoff)
          keep.add(mark.frame_idx);
      for (const mark of group) {
        mark.included = keep.has(mark.frame_idx);
        if (!mark.included) mark.detail_excluded = analysis(mark).noise
          ? analysis(mark).reason || "pointer/no change" : "earlier writing step";
      }
    }
    return result;
  }
  return { threshold, sensitivity, select };
})();
if (typeof module !== "undefined") module.exports = TransVisionSelection;
