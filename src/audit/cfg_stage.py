
from src.audit.pair_validation import retain_complete_pairs


def build_and_localize_cfg(
    samples,
    cfg_builder,
    cfg_localizer,
    audit,
):
    """
    Build CFGs and localize changed lines.

    Failed members are excluded. After processing every member,
    any pair missing one of its revisions is excluded as well.
    """
    successful_samples = []

    for sample in samples:
        cfg = cfg_builder.build(sample)

        if cfg is None:
            audit.record(
                stage="cfg_build",
                reason="cfg_build_failed",
                sample=sample,
                details={
                    "source": "sample.source",
                    "function_source_available": (
                        getattr(sample, "function_source", None)
                        is not None
                    ),
                },
            )
            continue

        sample.cfg = cfg
        sample = cfg_localizer.localize(sample)

        if not sample.seed_nodes:
            audit.record(
                stage="cfg_localization",
                reason="no_seed_nodes",
                sample=sample,
                details={
                    "seed_lines": list(sample.seed_lines or []),
                    "cfg_node_count": len(cfg["nodes"]),
                },
            )
            continue

        successful_samples.append(sample)

    complete_samples, dropped_samples = retain_complete_pairs(
        successful_samples,
        audit,
        stage="after_cfg_localization",
    )

    return complete_samples
