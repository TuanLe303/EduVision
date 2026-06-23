# Seat Assignment Monitor

This task assumes a fixed camera. When a teacher starts a class, the monitor
collects several recognized canonical tracks and freezes a median bbox, bottom
center anchor, and seat ROI for each `student_id`.

Assignments are identity-based rather than track-ID-based. A new track for an
existing identity is rebound to its existing seat. A newly recognized identity
after initial calibration is marked `late`, calibrated automatically, and given
a `provisional` assignment immediately. Teacher confirmation is optional and
does not block monitoring.

The monitor retains an unobserved `track_id` binding during the session so brief
occlusions do not discard identity. When face recognition confirms the same
student on a new track, the old track binding is replaced while the student's
seat assignment remains unchanged. This assumes the canonical tracker does not
reuse a terminated ID for another person without a new recognition result.

Missing detections become `temporarily_occluded` or `missing`; they are not
proof of `away_from_seat`. Leaving is confirmed only after an observed student
remains outside the learned seat region, the seat stays empty for the configured
duration, and face recognition has confirmed that student at the outside
location. Recognition may occur once during the continuous outside interval;
returning to the seat or becoming missing clears that evidence.

Every seat result exposes `spatial_score` and `temporal_score`. Its `confidence`
is their product for the state being returned: away states use away evidence,
while seated/returned states use the complementary seated evidence.
