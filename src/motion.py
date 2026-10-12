"""Visual content continuity, local repetition and coverage; no text recognition.

Registration affects comparison only. Exports always decode original source pixels.
Every automatic decision remains reversible through manual/AI frame selection.
"""

from collections import deque, OrderedDict
import cv2
import numpy as np

from src.cancellation import check_cancel
from src.ink import InkTracker

VERSION = 6


class GrayCache:
    """Lossless compact proxies with a bounded decoded lookback window."""
    def __init__(self, capacity=192):
        self.data = []
        self.cache = OrderedDict()
        self.capacity = capacity

    def append(self, image):
        ok, encoded = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError('Cannot cache visual analysis frame.')
        index = len(self.data)
        self.data.append(encoded)
        self.cache[index] = image
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index < 0:
            index += len(self.data)
        if not 0 <= index < len(self.data):
            raise IndexError(index)
        if index not in self.cache:
            self.cache[index] = cv2.imdecode(self.data[index], cv2.IMREAD_GRAYSCALE)
        self.cache.move_to_end(index)
        image = self.cache[index]
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return image


def validate_motion_mode(value):
    if value not in ("summary", "demo"):
        raise ValueError("Choose Summarize movement or Capture demonstration changes.")
    return value


def effective_analysis(mark, mode="summary"):
    analysis = mark["visual_analysis"]
    result = {**analysis, **analysis.get("motion", {}).get(mode, {})}
    if mode == "demo" and analysis.get("motion", {}).get("summary", {}).get("force_keep"):
        result.update(force_keep=True, suppressed=False)
    return result


def circle_candidates(image):
    """Potential colored pointer halos; movement is checked separately."""
    small = cv2.resize(image, (480, 270))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    color = ((hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 150)).astype(np.uint8)
    color = cv2.morphologyEx(color, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = np.zeros(color.shape, np.uint8)
    contours, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        x, y, w, h = cv2.boundingRect(hull)
        filled = float(color[y:y+h,x:x+w].sum()) / max(1, area)
        if 30 < area < 1300 and filled > .55 and .7 < w / max(1, h) < 1.4 and perimeter and 4 * np.pi * area / perimeter ** 2 > .82:
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            cv2.circle(mask, (round(cx), round(cy)), round(radius + 3), 1, -1)
    return mask.astype(bool)


class View:
    def __init__(self, gray, orb, circles=None):
        self.gray = gray
        self.core = np.full(gray.shape, 255, np.uint8)
        h, w = gray.shape
        self.feature_mask = np.zeros(gray.shape, np.uint8)
        self.feature_mask[round(h * .08):round(h * .90), round(w * .015):round(w * .985)] = 255
        self.edges = cv2.Canny(gray, 35, 100) & self.core
        self.orb = orb
        self.points = None
        self.descriptors = None
        self.circles = circles

    def features(self):
        if self.points is None:
            self.points, self.descriptors = self.orb.detectAndCompute(self.gray, self.feature_mask)


def compare(left, right, dynamic=None):
    """Account for translation/zoom; explicitly measure newly exposed areas."""
    h, w = right.gray.shape
    matrix = np.float32([[1, 0, 0], [0, 1, 0]])
    registered = False
    matches = []
    raw_delta = cv2.absdiff(left.gray, right.gray)
    alignment_delta = raw_delta.copy()
    if dynamic is not None:
        alignment_delta[dynamic] = 0
    need_alignment = float(alignment_delta.mean()) > .8 or float((alignment_delta > 25).mean()) > .002
    if need_alignment:
        left.features()
        right.features()
    if need_alignment and left.points is not None and right.points is not None and left.descriptors is not None and right.descriptors is not None:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left.descriptors, right.descriptors, k=2)
        matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .72 * b.distance]
        if dynamic is not None:
            matches = [m for m in matches
                       if not dynamic[min(h - 1, round(left.points[m.queryIdx].pt[1])), min(w - 1, round(left.points[m.queryIdx].pt[0]))]
                       and not dynamic[min(h - 1, round(right.points[m.trainIdx].pt[1])), min(w - 1, round(right.points[m.trainIdx].pt[0]))]]
    if len(matches) >= 10:
        a = np.float32([left.points[m.queryIdx].pt for m in matches])
        b = np.float32([right.points[m.trainIdx].pt for m in matches])
        fitted, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                    ransacReprojThreshold=2.5, maxIters=400)
        if fitted is not None and inliers is not None:
            count = int(inliers.sum())
            scale = float(np.linalg.norm(fitted[:, 0]))
            spread = np.ptp(a[inliers.ravel().astype(bool)], axis=0)
            angle = abs(float(np.arctan2(fitted[1, 0], fitted[0, 0])))
            if count >= 9 and count / len(matches) >= .35 and min(spread) > 22 and .35 < scale < 2.8 and angle < .06:
                matrix = fitted.astype(np.float32)
                registered = True
    old = cv2.warpAffine(left.gray, matrix, (w, h), borderValue=255)
    edges = cv2.warpAffine(left.edges, matrix, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.warpAffine(left.core, matrix, (w, h), flags=cv2.INTER_NEAREST) & right.core
    active = right.core.copy()
    if left.circles is not None and right.circles is not None and left.circles.any() and right.circles.any():
        ay, ax = np.nonzero(left.circles); by, bx = np.nonzero(right.circles)
        moved = np.hypot(ax.mean() - bx.mean(), ay.mean() - by.mean()) > 2
        pointer_area = left.circles | right.circles
        # A moving disk is ignored only when the surrounding view stays still.
        # Color changes in place, broad highlights and changing wires remain active.
        quiet_area = ~pointer_area
        if dynamic is not None:
            quiet_area &= ~dynamic
        if moved and float(pointer_area.mean()) < .08 and float(((raw_delta > 20) & quiet_area).mean()) < .008:
            active[pointer_area] = 0
    if registered:
        # Ignore stationary chrome only for registered comparison. New content
        # appearing at an image edge remains active, including slide headings.
        outer = np.ones(right.gray.shape, bool)
        outer[round(h * .08):round(h * .90), round(w * .015):round(w * .985)] = False
        active[outer & (raw_delta < 5)] = 0
    if dynamic is not None:
        active[dynamic] = 0
    common = (valid > 0) & (active > 0)
    kernel = np.ones((3, 3), np.uint8)
    added = (right.edges > 0) & (cv2.dilate(edges, kernel) == 0) & (active > 0)
    removed = (edges > 0) & (cv2.dilate(right.edges, kernel) == 0) & common
    new_area = added & ~common
    delta = (cv2.absdiff(old, right.gray) > 25) & common
    added_ink = (right.gray < 225) & (old >= 225) & ((old.astype(np.int16)-right.gray)>15) & common
    removed_ink = (old < 225) & (right.gray >= 225) & ((right.gray.astype(np.int16)-old)>15) & common
    # Individual compression speckles must not become pending writing.
    def coherent(mask):
        if not mask.any():
            return mask
        _, labels, areas, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        valid_labels = np.flatnonzero(areas[:,cv2.CC_STAT_AREA] >= 6)
        valid_labels = valid_labels[valid_labels != 0]
        return np.isin(labels, valid_labels)
    added_ink = coherent(added_ink)
    removed_ink = coherent(removed_ink)
    old_count = max(1, int(((edges > 0) & common).sum()))
    new_count = max(1, int(((right.edges > 0) & (active > 0)).sum()))
    shift = float(np.linalg.norm(matrix[:, 2]))
    scale = float(np.linalg.norm(matrix[:, 0]))
    return {
        "registered": registered,
        "moving": registered and (shift > .6 or abs(scale - 1) > .004),
        "overlap": float(common.sum() / max(1, (active > 0).sum())),
        "added": float((added & common).sum() / new_count),
        "removed": float(removed.sum() / old_count),
        "uncovered": float(new_area.sum() / new_count),
        "change": float(delta.sum() / max(1, common.sum())),
        "new_pixels": int(added.sum()),
        "ink_added": int(added_ink.sum()),
        "ink_removed": int(removed_ink.sum()),
        "_added_ink": added_ink,
        "_removed_ink": removed_ink,
        "_matrix": matrix,
    }


def repeating_region(history):
    if len(history) < 24:
        return None
    stack = np.stack([cv2.resize(v.gray, (120, 68)) for v in list(history)[-24:]])
    flips = (np.abs(np.diff(stack.astype(np.int16), axis=0)) > 24).sum(axis=0)
    mask = (flips >= 4).astype(np.uint8)
    active_pixels = int(mask.sum())
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    if active_pixels / max(1, int(mask.sum())) < .30:
        return None
    area = float(mask.mean())
    # A cursor's small repeated trail is not an animation panel. Masking a broad
    # writing area from cursor visits can otherwise hide cumulative new ink.
    if not .12 < area < .90:
        return None
    ys, xs = np.nonzero(mask)
    density = float(mask.sum() / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)))
    if density < .45:
        return None
    # Recurrent appearances are required; ordinary new writing or one-way
    # scrolling must not become a permanently ignored motion region.
    region = mask > 0
    returns = [float(cv2.absdiff(stack[-1], old)[region].mean()) for old in stack[:-5]]
    if not returns or min(returns) > 12:
        return None
    a, b = stack[-2], stack[-1]
    if float(((cv2.absdiff(a, b) > 24) & (mask == 0)).mean()) > .015:
        return None
    return cv2.resize(mask, (480, 270), interpolation=cv2.INTER_NEAREST).astype(bool)


def transient_overlays(markers, images, fps):
    """Suppress an occluding rectangle only when the underlying view returns."""
    flags = set()
    for i, gray in enumerate(images):
        check_cancel()
        dark = (gray < 65).astype(np.uint8)
        dark[:22] = 0
        dark[-27:] = 0
        count, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        boxes = []
        for x, y, w, h, area in stats[1:]:
            if .06 * gray.size < w * h < .65 * gray.size and area / (w * h) > .4:
                boxes.append((x, y, w, h))
        if not boxes:
            continue
        for x, y, w, h in boxes:
            outside = np.ones(gray.shape, bool)
            outside[max(0,y-3):y+h+3,max(0,x-3):x+w+3] = False
            before = None
            for j in range(i-1,max(-1,i-80),-1):
                if (markers[i]['frame_idx']-markers[j]['frame_idx']) / fps > 10:break
                delta = cv2.absdiff(images[j],gray)
                if float((delta[outside]>25).mean()) < .006 and float((delta[~outside]>25).mean()) > .25:
                    before=images[j];break
            if before is None:
                continue
            for j in range(i+1,min(len(images),i+120)):
                if (markers[j]['frame_idx']-markers[i]['frame_idx']) / fps > 10:break
                if float(cv2.absdiff(before,images[j]).mean()) < 2.5:
                    flags.add(i);break
    return flags


class ScanMotionGuard:
    """Avoid expensive one-frame boundary searches for continuous movement.

    Coarse candidates are still retained; demo and later review can use them.
    """
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=900, fastThreshold=12)
        self.history = deque(maxlen=24)

    def check(self, gray, raw):
        current = View(gray, self.orb)
        self.history.append(current)
        if len(self.history) < 2 or not raw.get("page_change"):
            return raw
        previous = self.history[-2]
        result = compare(previous, current)
        continuous = result["moving"] and result["added"] < .19 and result["removed"] < .19 and result["change"] < .075
        dynamic = repeating_region(self.history)
        if dynamic is not None:
            outside = compare(previous, current, dynamic)
            continuous |= outside["change"] < .008 and outside["added"] < .08 and outside["removed"] < .08
        return {**raw, "page_change": False} if continuous else raw


def annotate_motion(markers, images, fps, report=None, circles=None, colors=None):
    """Add summary/demo analyses without replacing raw comparison evidence."""
    orb = cv2.ORB_create(nfeatures=900, fastThreshold=12)
    history = deque(maxlen=24)
    loop_history = deque(maxlen=32)
    group = -1
    reference = None
    reference_index = 0
    loop_group = None
    loop_mask = None
    loop_baseline = None
    loop_bounds = None
    phases = []
    ink = InkTracker(images[0].shape) if images else None
    previous_index = None
    overlays = transient_overlays(markers, images, fps)
    for i, (mark, gray) in enumerate(zip(markers, images)):
        check_cancel()
        pointer = circles[i] if circles is not None else None
        if pointer is not None and pointer.ndim == 1:
            pointer = np.unpackbits(pointer, count=gray.size).reshape(gray.shape).astype(bool)
        current = View(gray, orb, pointer)
        raw = mark["visual_analysis"]
        if i in overlays:
            raw["motion"] = {"version": VERSION,
                             "summary": {"page_group": group, "suppressed": True, "reason": "temporary overlay; clear view returns"},
                             "demo": {"page_group": raw["page_group"], "noise": raw["noise"],
                                      "changed_fraction": raw["changed_fraction"], "suppressed": False,
                                      "force_keep": False, "reason": "application view"}}
            continue
        history.append(current)
        if not loop_history or (mark["frame_idx"] - markers[loop_history[-1][0]]["frame_idx"]) / fps >= .18:
            loop_history.append((i, current))
        dynamic = repeating_region([view for _, view in loop_history])
        summary = {"page_group": group, "noise": raw["noise"],
                   "changed_fraction": raw["changed_fraction"], "suppressed": False,
                   "force_keep": False, "reason": "content change"}
        demo = {"page_group": raw["page_group"], "noise": raw["noise"],
                "changed_fraction": raw["changed_fraction"], "suppressed": False,
                "force_keep": False, "reason": "demonstration change"}
        if reference is None or raw.get("camera"):
            group += 1
            reference, reference_index = current, i
            loop_mask = None
            loop_baseline = None
            loop_bounds = None
            loop_group = None
            ink.reset()
            summary.update(page_group=group, reason="new scene")
        else:
            adjacent = compare(history[-2], current)
            persistent_new = False
            if loop_mask is not None:
                outside_pixels = (current.core > 0) & ~loop_mask
                for v in (history[-2], current):
                    if v.circles is not None:
                        outside_pixels &= ~v.circles
                outside_change = float(((cv2.absdiff(history[-2].gray, current.gray) > 25) & outside_pixels).sum() / max(1, outside_pixels.sum()))
                latest_small = cv2.resize(current.gray, (120, 68))
                compatible = any(float(cv2.absdiff(latest_small, cv2.resize(v.gray,(120,68))).mean()) < 15
                                 for j,v in loop_history if j < i and 1 < (mark['frame_idx']-markers[j]['frame_idx'])/fps < 8)
                if loop_baseline is not None and len(loop_history) >= 8:
                    recent = np.stack([v.gray for _,v in list(loop_history)[-8:]])
                    stable = (recent.max(axis=0).astype(np.int16)-recent.min(axis=0)) < 12
                    median = np.median(recent,axis=0).astype(np.int16)
                    low,high = loop_bounds
                    novel = (median < low.astype(np.int16)-25) | (median > high.astype(np.int16)+25)
                    persistent = stable & loop_mask & novel
                    for _,v in list(loop_history)[-8:]:
                        if v.circles is not None:
                            persistent[v.circles] = False
                    _,_,components,_ = cv2.connectedComponentsWithStats(persistent.astype(np.uint8))
                    persistent_new = int(max(components[1:,cv2.CC_STAT_AREA],default=0)) >= 12
                if outside_change > .015 or not compatible or persistent_new:
                    loop_mask = None
                    loop_baseline = None
                    loop_bounds = None
                    loop_group = None
            if dynamic is not None and not persistent_new:
                outside = compare(history[-2], current, dynamic)
                trusted_pan = adjacent["moving"] and adjacent["removed"] < .1 and adjacent["added"] < .1 and adjacent["change"] < .02
                if not trusted_pan and outside["change"] < .008 and outside["added"] < .08 and outside["removed"] < .08:
                    loop_mask = dynamic
                    if loop_baseline is None:
                        learned = np.stack([v.gray for _,v in list(loop_history)[-24:]])
                        loop_baseline = np.median(learned,axis=0).astype(np.uint8)
                        loop_bounds = (learned.min(axis=0),learned.max(axis=0))
                    if loop_group is None:
                        start = i
                        for j,oldest in loop_history:
                            if compare(oldest, current, dynamic)["change"] < .008:
                                start = j
                                break
                        loop_group = markers[start]["visual_analysis"].get("motion", {}).get("summary", {}).get("page_group", group)
                        for earlier in markers[start:i]:
                            a = earlier["visual_analysis"].get("motion")
                            if a:
                                a["summary"].update(page_group=loop_group, reason="repeating motion")
                        group = loop_group
            comparison = compare(history[-2], current, loop_mask)
            coverage = compare(reference, current, loop_mask)
            summary["comparison"] = {k: round(v, 5) if isinstance(v, float) else v for k, v in comparison.items() if not k.startswith("_")}
            similar = comparison["removed"] < .19 and comparison["added"] < .19 and comparison["change"] < .075
            growing = comparison["removed"] < .09 and comparison["added"] < .65 and comparison["change"] < .14
            reveal = coverage["moving"] and coverage["uncovered"] > .20 and coverage["new_pixels"] > 180
            if (similar or growing) and not reveal:
                color = colors[i] if colors is not None else None
                if color is not None and color.ndim == 1:
                    color = np.unpackbits(color, count=gray.size).reshape(gray.shape).astype(bool)
                for index, minimum in ink.step(history[-2].gray, gray, comparison, color,
                                                mark['frame_idx']/fps, i):
                    earlier = markers[index]["visual_analysis"].get("motion", {}).get("summary")
                    if earlier is not None:
                        old_minimum = earlier.get('preserve_min_sensitivity', 100)
                        minimum = min(old_minimum, minimum)
                        earlier.update(force_keep=True, preserve_min_sensitivity=minimum,
                                       noise=False, reason="temporary emphasis before fading" if minimum else "before new writing disappears")
                redundant = (comparison["added"] < .015 and comparison["removed"] < .015 and comparison["change"] < .002)
                new_ink = comparison["ink_added"] >= 6 and comparison["ink_removed"] <= 3
                summary.update(page_group=group, noise=raw["noise"] or redundant,
                               changed_fraction=round(comparison["change"], 5),
                               reason="repeating motion" if loop_mask is not None else "same content moved" if comparison["moving"] else "writing added" if comparison["added"] > .01 else "same content")
                if new_ink:
                    summary.update(noise=False, reason="writing added")
            else:
                group += 1
                summary.update(page_group=group, reason="new area revealed" if reveal else "before content disappears / new scene")
                reference, reference_index = current, i
                loop_mask = None
                loop_baseline = None
                loop_bounds = None
                loop_group = None
                ink.reset()
        raw["motion"] = {"version": VERSION, "summary": summary, "demo": demo}
        if reference_index != i and persistent_new:
            summary.update(force_keep=True, noise=False, reason="new persistent content during animation")
        previous_index = i
        if loop_mask is not None:
            signature = cv2.dct(cv2.resize(current.gray.astype(np.float32), (32, 32)))[:8,:8].ravel()[1:]
            signature = signature > np.median(signature)
            if any(np.count_nonzero(signature != old) <= 5 for old in phases):
                demo.update(suppressed=True, reason="repeated animation phase")
            elif len(phases) < 64:
                phases.append(signature)
        else:
            phases.clear()
        if report and (i % 50 == 0 or i == len(markers) - 1):
            report(f"Checking movement and content coverage: {i + 1}/{len(markers)}",
                   progress=round(100 * (i + 1) / len(markers)), completed=i + 1, total=len(markers))
    return markers
