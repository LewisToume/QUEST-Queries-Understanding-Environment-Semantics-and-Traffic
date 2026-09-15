# Teacher Environment

StreamPETR and MapTRv2 remain external Agent and Vector Map teachers. Their
legacy OpenMMLab runtime is isolated from the QUEST `.venv`.

The attempted Windows legacy environment currently has Python 3.8.20, but the
requested CUDA-enabled PyTorch wheel and compiled OpenMMLab dependencies did not
install successfully. Use `envs/teacher_legacy.yml` on WSL2/Linux or a compatible
CUDA server.

Stage2 student training never imports or instantiates these teachers. Verified
teacher predictions must first be exported into the token-aligned offline label
format documented in `docs/offline_distillation.md`.
