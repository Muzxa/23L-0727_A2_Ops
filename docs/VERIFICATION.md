# Completed real-data verification - 6 November 2026

The actual Colab T4 experiment is complete, beyond the earlier synthetic software checks.

- 973 original readable images; zero unreadable candidates; one duplicate removed; 972 retained.
- Stratified split sizes: 680/146/146. Original membership is identical in local and Colab manifests.
- Zero decoded-pixel hash overlap between splits.
- One Colab processed PNG repaired losslessly from its audited original; 972/972 pixel hashes verified.
- Scratch CNN: 10 epochs, best epoch 6. Frozen head: 10 epochs, best 10. Last-block stage: 15 epochs, best 11.
- Eight completed Fashion-MNIST toy trials and eight completed applied trials.
- Tuned transfer selected by validation loss; thresholds locked before test.
- Saved probabilities reproduce all reported test metrics without another model inference pass.
- Ten displayed errors per model inspected and annotated with visible observations/testable hypotheses.
- Local Git/DVC two-version push/delete-cache/pull recovery passed. Both toy commits are published on DagsHub's dvc-toy branch.
- Real raw/processed/split DVC objects pushed to a local remote and verified in sync.
- Four local unit tests passed after the input caching change.

Full evidence lives under outputs/23L-0727_A2_Ops. FINAL_VERIFICATION.json records the completed evidence checks. The report and submission archive are under submission/.

No remote MLflow runs, API deployment, Tasks 3-8, or Grad-CAM bonus are claimed.
