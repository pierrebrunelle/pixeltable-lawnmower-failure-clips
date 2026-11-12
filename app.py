"""Robot lawnmower failure-clip mining, built with Pixeltable.

A small robot-mower fleet uploads one front-camera video per mowing run. Pixeltable
stores the runs, cuts every video into fixed 3-second clips with video_splitter,
pulls a keyframe from each clip, labels the obvious failures with cheap color rules
(hose snag, tipping on a slope, stalling in the dark, fence bump) and indexes every
keyframe with CLIP so the fleet team can mine look-alike failure clips by text or by
example image. Truncated uploads and runs without a recording never block the batch.

Layout: UDFs live in udfs.py; this file holds table models and @pxt.query.

    python make_videos.py
    pxt schema update app.py mower
    python run.py
"""

from __future__ import annotations

import PIL.Image

import pixeltable as pxt
import pixeltable.functions as pxtf

from udfs import color_shares, failure_label, firmware_family, gray_thumb, luma, overlay_mask

TableModel = pxt.model_base()
CLIP = pxtf.huggingface.clip.using(model_id='openai/clip-vit-base-patch32')


class Runs(TableModel, name='runs'):
    """One mowing run: the mower, the yard and the front-camera video."""

    run_id = pxt.Column(type=pxt.String, primary_key=True)
    mower: pxt.String
    yard: pxt.String
    video: pxt.Video | None
    firmware: pxt.String | None
    run_secs = pxtf.video.get_duration(video)
    fw_family = firmware_family(firmware)


class Clips(
    TableModel,
    name='clips',
    base=Runs,
    iterator=pxtf.video.video_splitter(Runs.video, duration=3.0, min_segment_duration=1.0),
):
    """Fixed 3-second clips of every run, each with a keyframe, a rule label and a CLIP embedding."""

    keyframe = pxtf.video.extract_frame(video_segment, timestamp=1.5)  # type: ignore[name-defined]  # noqa: F821
    brightness = luma(keyframe)
    shares = color_shares(keyframe)
    label = failure_label(shares, brightness)
    thumb = gray_thumb(keyframe)  # mode 'L'
    review = overlay_mask(keyframe, label)  # RGBA tint on failure clips
    clip_secs = segment_end - segment_start  # type: ignore[name-defined]  # noqa: F821

    __indexes__ = [pxt.EmbeddingIndex(keyframe, embedding=CLIP, name='keyframe_clip')]


@pxt.query
def failure_queue(label: str):
    """Clips the rules flagged with a given failure label, in run order."""
    return (
        Clips.where(Clips.label == label)
        .order_by(Clips.run_id, Clips.segment_start)
        .select(Clips.run_id, Clips.yard, Clips.segment_start, Clips.segment_end, Clips.brightness)
    )


@pxt.query
def clips_like_text(description: str, k: int = 4):
    """Top-k clips whose keyframe best matches a text description (CLIP text->image)."""
    sim = Clips.keyframe.similarity(string=description)
    return (
        Clips.order_by(sim, asc=False)
        .select(Clips.run_id, Clips.segment_start, Clips.label, score=sim)
        .limit(k)
    )


@pxt.query
def clips_like_image(example: PIL.Image.Image, k: int = 4):
    """Top-k clips that look like an example failure frame (CLIP image->image)."""
    sim = Clips.keyframe.similarity(image=example)
    return (
        Clips.order_by(sim, asc=False)
        .select(Clips.run_id, Clips.segment_start, Clips.label, score=sim)
        .limit(k)
    )
