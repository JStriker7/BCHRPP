# BCHRPP code versions

This repository preserves the existing local code snapshots for the MOSI
multimodal sentiment experiments. Each directory remains separate so older
pipelines can be inspected without pretending they share one configuration.

| Directory | Contents |
| --- | --- |
| `snapshots/111/` | Earlier compact handoff of RCSE-Lite and its MOSI extractor. |
| `BCHR++_Full_Code_v2.0_Experiment_Ready/` | Earlier full BCHR++ v2 implementation. |
| `BCHRPP_RCSE_Lite_Final_20261008/` | Current local RCSE-Lite code, MOSI metric protocol, and face-crop experiment scripts. |
| `extract/` | Offline feature extraction scripts used by the model projects. |

The `111` snapshot preserves the file layout supplied at the time; the active
RCSE-Lite project is nested under `BCHRPP_RCSE_Lite_Final_20261008/`.
Training data, extracted feature tensors, checkpoints, caches, generated
outputs, and server handoff notes are intentionally not tracked. The teacher's
`English/` reference project and method document are also withheld from this
public repository pending permission to redistribute them.

The three-seed whole-frame versus face-crop trial uses the same MOSI splits,
model settings, and `metricsTop.py` regression protocol. Its authoritative
per-seed CSV files currently remain on the wjy server; they are not replaced
by the local test scripts or historical weights in this repository.
