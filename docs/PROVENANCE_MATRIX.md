# Provenance matrix

| Material | Status | Meaning |
|---|---|---|
| Bundled RCS-OSA/BRA/loss source | `verified_from_local_source` | Directly inspectable code in this release |
| Model and ablation YAML files | `reconstructed_and_tested` | Structured from manuscript/local code; all four architectures build and run |
| Training protocol | `manuscript_reported` | Transcribed settings; final original `args.yaml` is missing |
| Dataset totals | `manuscript_reported` | Table/text transcription; private data not bundled |
| Split manifest template | `reconstructed_from_manuscript` | Empty schema, not recovered image assignments |
| Offline augmentation exporter | `reconstructed_from_manuscript` | Reimplementation with audit log; exact lost operation order is unknown |
| Paper comparison CSV | `manuscript_table_4` | Published/tabulated values, not rerun measurements |
| Measured-results template | `measured_run_pending` | Empty cells must be filled only by new runs |
| Grad-CAM and Pointing Game code | `reconstructed_protocol` | Produces new outputs only when weights/images are supplied |
| Synthetic example | `synthetic_smoke_test_only` | Software test data; never valid for scientific reporting |
| Environment YAML | `manuscript_reported; pending_author_confirmation` | Includes the unresolved version mismatch |

The words `reproduced`, `replicated`, and `regenerated` should be used only
after new runs create logs and metrics under a declared commit and environment.
