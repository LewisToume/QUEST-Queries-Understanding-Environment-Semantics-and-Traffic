# QUEST: Queries for Understanding Environment Semantics and Traffic

QUEST is a multi-camera perception research prototype for autonomous driving.
It studies how a single trainable scene representation can support **3D agent
detection, vectorized local mapping, semantic segmentation, and metric depth**.

**[Current implementation, experiment notes, and evaluation](https://github.com/LewisToume/QUEST-Queries-Understanding-Environment-Semantics-and-Traffic/tree/feature/multiview-quest)** are on the `feature/multiview-quest` branch. The default branch retains the initial prototype code.

The current model fuses eight calibrated camera views from the current and
previous frames. It uses a frozen DINOv2-S image encoder, trainable geometric
and temporal fusion, and a shared bird's-eye-view representation and scene-query
decoder. Four task readouts are trained jointly from nuPlan vector-map ground
truth and offline teacher pseudo-labels for the other tasks.

The shared model was evaluated on 120 held-out frames after training on 5,000
OpenScene/nuPlan frames. Evaluation distinguishes teacher agreement from
ground-truth Map metrics and compares the joint model with staged baselines.
Results are internal; the model has not passed all regression checks.

This is a research prototype, not a ready-to-run pretrained release.
Datasets, teacher weights, and trained QUEST checkpoints are not included.
