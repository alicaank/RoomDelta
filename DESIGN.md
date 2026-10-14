# Design notes

These notes explain the main decisions behind RoomDelta, what I tried that didn't work, and what I would do next.

## The problem I was actually solving

Finding differences between two videos of a room is easy. Almost all of the differences mean nothing: the camera took another path, something in front hides something behind it, or one video simply never looked at a corner. A system that reports differences ends up reporting dozens of "changes" in a room where four things moved.

So the goal I set was narrower. Report a change only when the second video *looked at the place* and what it saw contradicts the first video. In every other case, say "unchanged", "uncertain" or "not seen before". Most of the design follows from that.

## Geometry

I use Depth Anything 3 for reconstruction. One model gives depth, camera poses, intrinsics and a confidence map. The base model is Apache-2.0 and runs in about 4 GB of GPU memory.

The two visits have to end up in one coordinate frame, but their contents must stay separate. A chair that moved in the second video should not change the depth estimated in the first. I tried three setups:

- **Reconstruct each video separately, then align them.** This works, but two independent runs of a learned model disagree a little everywhere, not just by a rigid transform. The leftover misalignment was about 1.9% of the scene size.
- **Reconstruct both videos as one scene.** I rejected this without much testing. Mixing content from both visits is exactly what I wanted to avoid.
- **Estimate all camera poses jointly, but each visit's depth only from its own frames, conditioned on those poses.** This is the default. Misalignment drops to 1.6%, and the depth noise between visits drops by about 15%.

A feature-matching alignment step still runs afterwards as a check. If it fails, every change is downgraded to "uncertain".

Scale comes from DA3's metric depth model. It is a learned estimate, so the report says so and cross-checks it against the camera height it implies (1.30 m here). If you have a real measurement, you can pass it in.

## Finding objects

Grounding DINO detects objects from a list of everyday categories and SAM 2.1 cuts them out. I also segment regions where the geometry clearly changed, so an object with no category in the list still becomes an object. Masks are lifted into 3D with the depth maps and merged across frames in 3D, not in the image. Two masks from the same frame are never merged; otherwise two identical chairs standing side by side would turn into one.

Detection alone gives each object about 9 views and leaves duplicates. Tracking every object through its video with SAM 2.1's video mode raised that to about 16 views and merged 3 or 4 duplicates per run. Trackers drift, so a tracked mask is only kept if its 3D points stay inside the object's footprint.

## Deciding what changed

This part changed the most during development.

**First version: rules.** I started with hand-written rules: an object counts as removed if at least two distinct viewpoints see through its old position; a move is rejected if the old position still shows the same shape; thin objects get an exception; and so on. Each rule made sense on its own. The trouble was that every new test exposed a new failure, and every fix was another rule. When I swapped the input videos, or switched to a different reconstruction model, something new broke each time.

**Second version: one model.** Looking at the rules together, they were all special cases of one question: which state of the room best explains the depth the other video measured? So I replaced them with a model that asks that question directly.

For each object, every possible explanation (stayed, moved by some rigid motion, removed, new) predicts what each camera of the other video should see. I compute those predictions by rendering the scene into that camera, with the object present, absent, or moved. Each prediction is scored against the measured depth. The scoring uses a noise model fitted on the recording pair itself, from how much the static parts of the room disagree between the two reconstructions. A global assignment then picks the best explanation for all objects at once.

The old rules mostly fall out of this:

| Old rule | Where it went |
|---|---|
| occlusion doesn't mean removal | all explanations predict the occluder, so an occluded frame carries no evidence |
| need several viewpoints | each viewpoint's evidence is capped, and a change has a prior cost, so one viewpoint is never enough |
| a move is impossible if the old spot still shows the object | the "moved" explanation predicts an empty spot and scores badly there |
| thin objects are a special case | for a plate, "present" and "removed" predict almost the same depth, so the geometry stays neutral |

A few additions came from failures I actually ran into:

- **Rays the other video never observed count as no evidence.** Without this, an explanation could gain points just by filling empty space.
- **Each object may shift slightly in depth before scoring.** The two reconstructions are sometimes off by a few centimetres in one region. An empty spot is off by tens of centimetres, so this tolerance cannot explain away a real removal.
- **Appearance similarity, calibrated on the scene.** Objects that stayed in place give examples of "same object". Different objects in one video give examples of "different objects". Without this, a black backpack labelled "suitcase" got matched to the real suitcase.
- **Objects that touch and move by the same amount count as one move.** This is how the flat flower bowl ends up grouped with the flowers it carries.
- **Leftover fragments are explained by the move they sit on.** A second detection on a moved object's new spot is part of that move, not a separate "added" object.

Evidence from one learned reconstruction is correlated across its frames, so I scale it down (by a factor of 4) and cap each viewpoint's contribution. Otherwise thousands of pixels from one viewpoint would look like overwhelming proof. A change is reported only when it beats the next-best explanation by 19 to 1. These numbers were set once and not tuned per test.

The table below compares the two versions on the same checks:

| Test | Rules | Model |
|---|---|---|
| before vs. after | 4 moves plus one wrong "newly observed" armchair | 4 moves |
| inputs swapped | 4 moves, only after adding rules for this case | 4 moves |
| each video against itself | nothing | nothing |
| larger reconstruction model | 4 moves plus a wrong "flowers removed" | 4 moves |

## Things that didn't work

- **Comparing point clouds directly.** On the example this gives 23 "removed" regions, and 11 of them are spots that were hidden or never filmed in the second video.
- **Per-point appearance comparison** to catch flat objects. Image features of plain floor and rug vary too much from frame to frame. Appearance works at the level of whole objects, not single points.
- **Trusting every depth disagreement.** A far-away placemat, seen in only two frames, sat a few centimetres off in the other reconstruction. It showed up as a "book" that had been added, or removed when the inputs were swapped. Fitting the noise model on the comparison between videos instead of within one video was not enough on its own. Allowing each object a small, bounded depth shift before scoring fixed it.

## Robot navigation

I added a small navigation check because a robot cares less about which object moved than about whether its path is still free. Each visit gets a floor map with three states (occupied, free, not seen). The path walked in the first video serves as the route to check. The check models the robot as a disc, not a full body, and floor that was never filmed is reported as unknown rather than blocked.

## Runtime

About 6–9 minutes per pair on an RTX 5070:

| Step | Time |
|---|---|
| reconstruction | ≈ 25 s |
| detection and segmentation | ≈ 2 min |
| tracking | ≈ 3 min |
| decision | ≈ 2 min |

Speed was not a goal, but I capped each video at 32 frames at 504 px to keep runs short while iterating. Every step is cached, so most changes to the decision code rerun in under 3 minutes.

## What I would do next

1. **More recordings with known changes.** Above all ones with added and removed objects, a case where something is only hidden, and a second room. This matters more than any model change.
2. **Shape completion** for objects seen from one side only. The armchair in the example is seen from the front in one video and from behind in the other, and connecting the two currently takes a separate rule.
3. **Suggested viewpoints**: tell the user where to film next to resolve the uncertain cases.
