## My Crowd Experiment

**Implementation Notes:**
- The loss function must be properly **normalized** to ensure stable training; however, **Sigmoid should not be applied indiscriminately**, as it may distort regression-based objectives.
- Input data follow the conversion pipeline:
  `PIL.Image → numpy.asarray → torch.Tensor`.
- During validation, testing, and visualization, **foreground** and **background** regions must be explicitly distinguished.
- Configuration management should be handled via `configs.yaml` to ensure reproducibility and clarity.
- The **LMDS** module outputs **points count, coordinates, labels, and confidence scores**, all in **list format**.
- **PointToMask** generates the ground-truth **hard disk mask** from point annotations.
