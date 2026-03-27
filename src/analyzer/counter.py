# Thêm vào cuối hàm update() trong LineCounter
# để xem triggered values thực tế

def update(self, tracks: list[Track]) -> CounterStats:
    if not tracks:
        return self.get_stats(active_tracks=0)

    sv_detections = sv.Detections(
        xyxy       = np.array([t.bbox_xyxy  for t in tracks]),
        confidence = np.array([t.confidence for t in tracks]),
        class_id   = np.array([t.class_id   for t in tracks]),
        tracker_id = np.array([t.track_id   for t in tracks]),
    )

    for t in tracks:
        self._id_to_class[t.track_id] = t.class_name

    crossed_in, crossed_out = self.line.trigger(sv_detections)

    for i, track in enumerate(tracks):
        tid = track.track_id
        cls = track.class_name

        if crossed_in[i] and tid not in self._counted_ids:
            self._in_count += 1
            self._per_class_in[cls] += 1
            self._counted_ids.add(tid)
        elif crossed_out[i] and tid not in self._counted_ids:
            self._out_count += 1
            self._per_class_out[cls] += 1
            self._counted_ids.add(tid)

    return self.get_stats(active_tracks=len(tracks))