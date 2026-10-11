# TransVision manual

[Back to README](../README.md)

## Start and stop

On Windows, run **Setup TransVision.cmd** once, then **Start TransVision.cmd**. Keep the app open while a job runs. Closing the last app window/tab stops its local workers, shuts down its server and lets the launcher terminal close. Normal refreshes and multiple tabs are supported. A crashed or disconnected tab may take about 90 seconds to be detected when idle; a running job is allowed to finish if no explicit close notification arrives. Background tab throttling does not cancel unattended processing. A recognition or decoding call already in progress must finish before shutdown. Use `python -m src.app --keep-alive` to retain the server after closing tabs; `--no-open` also leaves it running until Ctrl+C.

- **Stop preview** unloads the selected video and returns to the chooser; a saved review is preserved.
- **Stop** cancels loading, detection, comparison, PDF export or OCR at the next safe checkpoint. It is not an immediate kill of a native decoder or recognition call.
- **Stop now** in Batch queue cancels the current file and leaves it queued for Start / resume. Completed files are skipped. A saved scan is reused if available; an interrupted scan may need rescanning.
- **Pause after file** lets the active batch file finish, then stops before the next file.

Completed exports are retained. Interrupted exports may leave partial files and an INCOMPLETE.txt notice; do not treat those as finished. Retry creates a fresh export folder and never overwrites a completed one. Closing during unfinished processing cancels it; leave the app open for unattended batches.

## Choose PDF, OCR, or both

| Mode | Output | When to use it |
|---|---|---|
| PDF only | Cropped frame images with optional timestamps; no OCR | Preserve diagrams, handwriting, equations and layouts faithfully |
| OCR only | Selected PNG frames and a plain-text visual transcript | Search or copy clear printed text |
| PDF + OCR | Both outputs | Keep faithful images alongside a searchable draft |

**OCR works best for plain text.** Complex text, equations, handwriting, diagrams and unusual layouts may be incomplete or incorrect. Even a high confidence score does not prove accuracy. Use the original images/PDF as evidence, or the optional AI correction workflow below. PDF export does not require OCR models.

## Correct complex content with an LLM

1. In the editor, mark complex frames **Mark for AI** (shortcut A). Make sure those frames are included, then export OCR.
2. Open the export folder. Inside **AI-marked/** are only copies of included, flagged images. Press Ctrl+A there and drag those images into the LLM you use.
3. Attach **transcription.txt** from the export root and copy the generated correction prompt from the app or **AI_ENHANCEMENT_PROMPT.txt**.
4. Ask the LLM to return the complete corrected transcription. The prompt limits edits to matching AI blocks, preserves timestamps and unflagged blocks, and asks for unclear content to be identified. Review the result against the images.

No images are uploaded automatically and TransVision does not call a paid LLM API. Your chosen LLM may have attachment/context limits; send smaller chronological groups if necessary. **AI_UPLOAD.zip** is an optional bundle of the transcript, prompt and flagged images, for services that accept ZIP files. If no frames were marked, AI-marked is absent and the prompt explains that there are no flagged images.

## Movement and repeated animation

**Summarize movement** is the default. It compares content after accounting for
shifts and zooming, groups repeating motion, and keeps views that reveal new areas.
New writing is tracked separately from motion; when detected new ink disappears,
a previous frame can be kept with its original timestamp. Temporary dark overlays
are suppressed only when the clear underlying view returns shortly afterward.

**Capture demonstration changes** keeps more distinct application/layout states
and animation phases. Repeated recognized animation phases are still suppressed.
Use it when movement itself matters, such as a software demonstration. Disable
**Light slides only** in batches if the useful content is dark or is not a slide.

Movement is separate from the 0–100 sensitivity slider and is available for both
single-video review and batch jobs. Changing it uses cached analysis, so frame
counts update without rescanning. Older saved reviews are analyzed once when
opened. Hover over a frame to see its selection reason. Manual and AI-marked
choices remain protected; automatically excluded candidates are recoverable.

This is a visual comparison, not a text-recognition or meaning-understanding
system. It works in PDF-only mode without OCR. Uncertain views may remain,
especially during fast transitions, complex moving scenes, or unreliable
alignment. Brief changes between scan samples can still be missed. Review
important exports, and use shorter scan intervals or manual marks when needed.

## Export names and folders

Click **Export folder** to choose where PDF and OCR exports are saved. On your first
export (or first batch start), TransVision asks you to confirm a folder. The initial
suggestion is the application's `output` folder.

- **Remember this folder:** use it for later exports and after restarting the app.
- **Ask each time:** choose before each single-video PDF/OCR export or batch run.
- **Change anytime:** open Export folder again, or use Change export folder in the batch queue.

A batch uses one destination for its entire run so it can work unattended. Its
current destination is shown above the queue. Every completed video is saved
immediately; you do not have to wait for the whole queue to finish. Canceling the
folder dialog does not start processing. An unwritable folder gives an error
before processing starts.

Changing the preference affects future exports; it does not move earlier files.
When you resume a stopped batch in a different folder, unfinished jobs use that
folder, while completed jobs retain their original output links. Completed scans
can still be reused from their original review folders.

Single exports create a video-named subfolder in the chosen folder. Batch PDF parts
are also placed directly in the batch destination; OCR and AI-marked images are
grouped in video-named subfolders. Review sessions and scan caches may remain in
the application's original `output` folder. Preferences are stored locally in
`.local/export-preferences.json` and are not included in GitHub downloads.

Frames use original-video time: `HH-MM-SS-mmm.png`. Included AI flags add `__AI.png`. The same images also appear in AI-marked for convenient upload. Unflagged images stay in the export root.

The OCR export contains transcription.txt, a compatibility copy named visual_transcript.txt, selected_frames.json, duplicates.json, AI_ENHANCEMENT_PROMPT.txt and AI_UPLOAD.zip. Text represents what was visible, **not spoken audio**. Pair a separate audio transcript with the timestamps.

PDF exports also include copies of AI-flagged frames in AI-marked; creating a text transcript and correction prompt requires an OCR export.

PDFs keep the source video name. Over-limit exports become `recording_01.pdf`, `recording_02.pdf`, etc.; each part has at most 100 pages. PDF page captions preserve source time. The intermediate master PDF and pages.json remain in the export folder for traceability.

## Detailed controls and setup

## Get started

You need **Python 3.12 or newer**, a local video file, and a modern browser. Windows is the tested platform; the Python server may work elsewhere, but the launchers and folder-opening actions are Windows-specific. This is a terminal-launched local application, not a packaged `.exe`.

1. Download or clone this repository.
2. Double-click **Setup TransVision.cmd**. Internet is needed to install dependencies and for the first OCR-model downloads.
3. Double-click **Start TransVision.cmd**. Keep its terminal open while working.

Or use a terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r setup/requirements.txt
.\.venv\Scripts\python.exe -m src.app --app-window
```

The CPU OCR installation is the default. **Tesseract** is additionally needed for the Programming profile and optional English recovery. On Windows, install it separately and make `tesseract.exe` available on PATH or in the normal Program Files installation folder. PDF/image export works without running OCR.

## The workflow

### Prepare the recording

Browse for a video or paste its path. The preview opens **before detection**.

- Seek to a representative slide or screen.
- Open **Crop & OCR → Suggest from current frame**. The suggestion is recomputed from the frame you are viewing, not the opening frame.
- Adjust the rectangle, draw your own area, or use the full frame. Confirm it before scanning.
- Optionally trim a stale intro or outro. Trim endpoints snap to source frames; the end is exclusive. **End after current frame** includes that frame.
- Choose a scan interval. The default is 0.5 seconds, with midpoint probes and local refinement around larger changes.

The crop is used for change comparison, camera filtering, OCR, and cropped exports. It is one area for the recording; check it when the layout changes. Editing it after detection updates the selection analysis and exports, but does not retroactively rescan every possible source-frame event.

### Review what matters

Use the filmstrip, seek bar, timestamp input, or frame controls. The paused preview is decoded from the original video. You can zoom, focus the preview, inspect the presentation crop, add or delete marks, include/exclude frames, and flag complicated frames for optional AI help.

Use the **Sensitivity** slider. The selected frame count updates immediately as you move it:

| Value | Useful for |
|---|---|
| 0 · Completed frames | The last useful state of each page or visual segment |
| 50 · Major writing steps | Balanced coverage of important intermediate stages |
| 100 · Fine details | All meaningful detected writing changes, while filtering pointer/no-change noise |

Values such as 37 and 67 are selectable. Pointer release snaps within three points of 0, 50, or 100; keyboard arrows move one point at a time. Click the labeled stops to jump directly. The count is selected frames out of available detected frames, not the total number of frames in the source video. Cached analysis makes changes immediate without another scan. Manual marks, AI flags, and explicit include/exclude choices are preserved. Fine detail cannot recover events missed during scanning. Older saved reviews translate their former two settings to the unified slider.

Excluded marks remain recoverable. **Exclude all** grays them out; **Remove excluded** deletes those marks and can be undone. Camera suppression is optional, and automatic filtering can be wrong.

### Export

- **Export OCR:** selected source-resolution PNGs plus a chronological `transcription.txt` draft.
- **Export PDF:** one cropped source image per selected frame. OCR is not required.
- Both normal PDF exports and batch PDF exports are split into video-named numbered parts when they exceed 100 pages. The normal editor shows download links for each part; source timestamps stay intact.
- **PDF timestamps:** optional selectable timestamps above each image, without covering the slide.
- **Copy time / Copy marked times:** copy one timestamp or the included selection.
- **Open frames folder:** available as soon as extraction finishes, while OCR continues.
- **AI flag:** optional `__AI` filenames, a copyable enhancement prompt, and a ZIP for manual upload to the AI you choose. TransVision makes no LLM API calls.

Exports go into the chosen export folder. Existing exports are kept; another run gets a new numbered folder. Review edits autosave in `session.json` in the review folder.

```powershell
# Open a video at the preparation stage
.\.venv\Scripts\python.exe -m src.app --video "D:\Recordings\lesson.mp4"

# Resume a saved review, including an interrupted post-scan analysis checkpoint
.\.venv\Scripts\python.exe -m src.app --session ".\output\lesson\session.json"
```

Pending checkpoints finish analyzing their saved candidates automatically when opened. Changing the crop can require one analysis pass when you next adjust sensitivity; this does not rerun detection.

### Batch processing

Open **Batch queue** from the header. Paste one video path per line, choose
**PDF only**, **OCR only**, or **PDF + OCR**, and configure sensitivity,
timestamps, and the crop reference time (default: five seconds).
PDF batches default to **100 pages per file**. Longer exports are split into
`video-name_01.pdf`, `video-name_02.pdf`, and so on, preserving the original
timestamps and page content. Shorter exports retain `video-name.pdf`.
The page limit can be lowered in batch settings.

- **Automatically export without review** runs each recording through to its selected exports. Turn it off to save scanned reviews without exporting, then use **Open review** from the queue.
- Jobs run sequentially, with per-file status, progress, completion dates, errors, and output links. Adding another batch preserves earlier jobs and their settings.
- **Pause after file** safely finishes that recording. **Start / resume** continues pending jobs. **Retry failed files** queues failures without rerunning successful exports.
- Crop suggestions start at the reference time and check later frames if it shows no slide. A crop that cannot be inferred reliably fails clearly instead of silently exporting a full-screen meeting.
- **Light slides only** additionally excludes dark/no-slide content and faces inside the presentation crop. Turn it off for dark slides or slides containing portrait photos; automatic filters can be wrong.
- The local queue persists in `.local/batch-queue.json`; output reviews remain resumable. Completed scans can be reused after interrupted exports. A scan interrupted before its saved checkpoint restarts.
- Keep the application's terminal open while a UI batch runs. The batch worker requests Windows to stay awake while running; closing the lid or shutting down still interrupts work.

For a standalone worker that continues without an open editor:

```powershell
.\.venv\Scripts\python.exe -m src.cli --queue ".\.local\batch-queue.json"
# Or create a new set of jobs from a JSON list of local video paths:
.\.venv\Scripts\python.exe -m src.cli --queue ".\.local\my-queue.json" --inputs ".\videos.json" --task pdf --sensitivity 50
```

PDF-only jobs never run OCR. All batch processing is local and makes no paid AI/API calls. Existing exports are retained; batch PDFs also receive a conveniently named copy in the output root.

## Why filenames are timestamps

An ordinary frame is named `00-12-33-250.png`. An AI-flagged frame is `00-12-33-250__AI.png`. Both represent **00:12:33.250 in the original video**.

The filename is durable context: even after moving an image into another folder or uploading it elsewhere, you know where it belongs. Windows-safe hyphens replace colons. The millisecond field distinguishes nearby frames; only AI flags add a suffix. Trimming never resets the time to zero.

The TXT uses blocks such as `[00:12:33.250] [VISUAL]`. PDF page captions and `pages.json` preserve the same mapping. Pair them with an audio transcript at the same source time. A visual transcript is not an audio transcript, and TransVision does not generate or automatically merge spoken transcription.

## OCR choices and honest limits

- **Fast English:** a mobile detector/recognizer and two bounded CPU workers. Best starting point for ordinary slide text.
- **Programming:** Tesseract with spacing/indentation recovery. Inspect punctuation and indentation.
- **Auto / Math:** slower symbol recovery. Equations, matrices, handwriting, and diagrams can still be damaged.
- **Second pass:** optional additional recovery on uncertain lines; slower.

**The PNG/PDF is the evidence. OCR is a draft.** A high confidence score does not prove that a minus sign, matrix row, or handwritten variable is correct. AI flags are an optional escape hatch, not a requirement. If your AI can read images well, the timestamped PDF and a separate audio transcript may be more useful than the OCR text.

GPU execution is requested only if the installed ONNX Runtime exposes a supported CUDA or DirectML provider. The default dependency installation is CPU-based. GPU speed has not been benchmarked; installing GPU runtimes is not necessary to use this beta.

Timestamps use source frame indices and the reported frame rate. For variable-frame-rate recordings, convert a copy to a constant frame rate before relying on alignment with audio.

Very brief slides can fall between sampling probes. Camera/photo filtering, pointer filtering, page grouping, slide animations, and unusual layouts can require manual correction. Browser playback depends on codec support; paused frame inspection can still work when playback cannot.

## Keyboard controls

| Shortcut | Action |
|---|---|
| Space | Play/pause |
| Left / Right | Previous/next frame |
| Shift + Left / Right | Step one second |
| Ctrl + Left / Right | Previous/next included mark |
| M / A | Mark frame / toggle AI |
| Delete / Backspace | Delete active mark |
| Ctrl + Z | Undo |
| Ctrl + Y / Ctrl + Shift + Z | Redo |
| Wheel / Ctrl + wheel over filmstrip | Scroll / zoom |

The editor includes a collapsible shortcut guide. Text fields keep their normal typing shortcuts.

## Privacy and diagnostics

The editor serves only on `127.0.0.1`, with an ephemeral local token and host checks. Video, frames, and OCR are processed locally. Installing packages or downloading OCR models needs internet; processing does not upload your recordings. Any optional AI upload is something you do yourself.

Local diagnostics record timings, resource samples, capture/selection decisions, errors, and optional notes under `test_runs`. These logs can contain local paths and recognized text metadata. Inspect/redact them before attaching them to a public issue. Videos, output folders, diagnostics, environments, and old test artifacts are excluded from Git.

Progress has separate scanning and selection-analysis stages. An estimate that runs out shows that work is continuing, rather than staying at a fictitious one second. A checkpoint is saved before post-scan analysis.

## Troubleshooting and feedback

Keep the terminal error message and describe the processing stage, video format/duration, crop and sensitivity when reporting a problem. Share a short non-sensitive example rather than a private recording. Stop and retry an interrupted export in a fresh folder; completed exports are preserved.

The app serves only on loopback and uses a temporary local token. Do not expose its port through a proxy or share live session URLs. Setup upgrades pip before installing dependencies.

## Application folders

The root contains README, the Windows setup/start shortcuts and required license notices. Application code is in src/, browser assets in web/, the camera-filter model and its license in models/, and installation dependencies in setup/requirements.txt. This manual is in docs/. Generated exports go into your chosen destination (initially output/); original recordings remain untouched.
