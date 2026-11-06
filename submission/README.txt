23L-0727 - VisionGuard - Tasks 1 and 2

Primary submission: 23L-0727_VisionGuard.pdf
The report includes executed Colab evidence, dataset/DVC evidence, model and tuning results, evaluation, and error analysis.

Project source: https://github.com/Muzxa/23L-0727_A2_Ops
DagsHub mirror: https://dagshub.com/Muzxa/23L-0727_A2_Ops
The report identifies the graded implementation commit. A later documentation commit packages the report and cropped screenshots.

The supporting submission ZIP contains:
- The PDF and the current project source.
- evidence/: compact CSV, JSON, plots, cropped screenshots, and annotated error-analysis tables.
- 23L-0727_VisionGuard_executed.ipynb: the downloaded executed Colab notebook. Earlier troubleshooting cells/errors remain in its execution history; the final training_cached run completed successfully.
- 23L-0727_colab_evidence.zip: the original unmodified Colab export, including checkpoints and tuner state. Its initial blank error-analysis CSVs are superseded by the manually annotated copies under evidence/colab/training_cached/evaluation/.

Screenshots are cropped to exclude the ChatGPT software cursor overlay. Displayed results were not edited. All reported test figures reuse the saved final predictions, with no new training or test inference during report preparation.

Dataset bytes and the local DVC remote are excluded from the submission ZIP. Their persistent local location is outputs/23L-0727_A2_Ops/versioned_storage/dvc_remote in the working project. Reproduction elsewhere requires that remote (or a copy) and an updated remote path. Git alone does not contain the large dataset objects.

Google Drive storage was full, so the runtime notebook could not save there. The local executed notebook and original export preserve the run. Review the name, roll number, report, and AI assistance disclosure before submitting.
