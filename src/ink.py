"""Track new ink through partial fading, independently of view registration."""

import cv2
import numpy as np


def colored_ink(image):
    small = cv2.resize(image, (480, 270))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return (hsv[:, :, 1] > 65) & (gray < 225)


class InkTracker:
    """One before-loss capture per coherent addition, not per fading pixel.

    Color alone never decides. Colored additions near established ink are
    possible emphasis; new marks in empty space remain protected. One label
    image follows registration, avoiding a full image per tracked component.
    """
    def __init__(self, shape):
        self.shape = shape
        self.labels = np.zeros(shape, np.int32)
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.labels.fill(0)
        self.tracks.clear()
        self.next_id = 1

    def step(self, old, current, metric, color, now, current_index):
        matrix = metric['_matrix']
        events = []
        if not np.array_equal(matrix, np.float32([[1,0,0],[0,1,0]])):
            self.labels = cv2.warpAffine(self.labels.astype(np.float32), matrix,
                                         (self.shape[1], self.shape[0]),
                                         flags=cv2.INTER_NEAREST).astype(np.int32)
        sizes = np.bincount(self.labels.ravel(), minlength=self.next_id)
        live = np.bincount(self.labels[current < 225], minlength=self.next_id)
        lost = np.bincount(self.labels[metric['_removed_ink']], minlength=self.next_id)
        retired = []
        for key, track in self.tracks.items():
            size = sizes[key]
            if size < 6:
                retired.append(key)
                continue
            if live[key] < size*.45 and lost[key]:
                stable = track['observations'] >= 2 and now-track['born'] >= .75
                minimum = 50 if track['emphasis'] else 0
                if minimum == 0 or stable:
                    events.append((track['best_index'], minimum))
                retired.append(key)
            elif live[key] >= size*.8:
                track['observations'] += 1
                track['best_index'] = current_index
        if retired:
            lookup = np.arange(self.next_id, dtype=np.int32)
            lookup[retired] = 0
            self.labels = lookup[self.labels]
            for key in retired:
                del self.tracks[key]
        added = metric['_added_ink'].copy()
        occupied = cv2.dilate((self.labels > 0).astype(np.uint8), np.ones((3,3),np.uint8)) > 0
        added[occupied] = False
        if metric['moving'] or metric['ink_removed'] > max(3, metric['ink_added']*.15):
            return events
        n, components, areas, _ = cv2.connectedComponentsWithStats(added.astype(np.uint8))
        old = cv2.warpAffine(old, matrix, (self.shape[1], self.shape[0]), borderValue=255)
        near = cv2.dilate((old < 225).astype(np.uint8), np.ones((17,17),np.uint8)) > 0
        near_counts = np.bincount(components[near], minlength=n)
        # Grayscale analysis is blurred: include antialiased color fringes.
        color_region = cv2.dilate(color.astype(np.uint8), np.ones((3,3),np.uint8)) > 0 if color is not None else np.zeros(self.shape,bool)
        color_counts = np.bincount(components[color_region], minlength=n)
        lookup = np.zeros(n, np.int32)
        for label in range(1, n):
            size = areas[label, cv2.CC_STAT_AREA]
            if size < 6:
                continue
            emphasis = color_counts[label] > size*.65 and near_counts[label] > size*.65
            key = self.next_id
            self.next_id += 1
            lookup[label] = key
            self.tracks[key] = dict(born=now, observations=1, best_index=current_index, emphasis=bool(emphasis))
        mask = components > 0
        self.labels[mask] = lookup[components[mask]]
        return events
