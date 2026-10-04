# Figures

[Back to the README](../README.md)

The builders read analysis outputs copied under `$FIG_ROOT/artifacts/` and write `$FIG_ROOT/figs/`:

| input under `artifacts/` | written by |
|---|---|
| `replication/REPLICATION_FINAL_decomp_payload_decomposition.json` | `stats/rl_payload_contrast_v2_rep.py` with `arms_rep_decomp_final.json` (its `payload_decomposition.json`) |
| `locked_forms/LOCKED_orderfill_POSTHOC.json` | `stats/rl_orderfill_contrast_locked.py` with `arms_locked_orderfill.json` (its `orderfill.json`) |
| `replication/REPLICATION_FINAL_endpoints.json`, `replication/LOCKED_FINAL_endpoints.json` | `stats/rl_rep_endpoints.py` (its `endpoints.json`) |
| `fig3/fig3_data_replication.json` | `stats/rl_fig3_data.py` |
| `conv/<checker>_<pop>/orderfill.json` | `stats/rl_orderfill_contrast_locked.py` with the convention arms |
| `train_curves/train_curves.json` | `stats/rl_train_curves.py` |
| `listed_hist/listed_hist.json` | `stats/rl_listed_hist.py` |
| `flip_example/flip_pick.json`, `flip_example/flip_true.png` | `figures/rl_flip_pick.py` |
| `qual_examples/qual_examples.json` | `figures/rl_qual_examples.py` |
| `pair_example/pair.json`, `report.png`, `partner.png` | `figures/rl_teaser_pick.py` |

`figures/build_fig1_npj.py` (Figure 1), `figures/build_fig2_npj.py` (Figure 2), `figures/build_fig3_npj.py` (Figure 3),
and `figures/build_supp_figs.py` (Supplementary Figures S1 to S4). Figures 1 and 2 show CheXpert Plus radiographs and a
report sentence, so their inputs exist only where the dataset is available; the pickers write them from your copy.
