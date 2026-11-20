# CCPO model of the paper

| File | md5 |
|---|---|
| `ckpt_big6f_ep0057.pt` (training checkpoint, epoch 57) | `6c2e875f97af9f6147e16a051ed75e09` |
| `ccpo_arm6_feas10k_200ep_ep57.scripted_batched.pt` (TorchScript export read by `bab_ccpo`) | `db80529547723c1054a6e8cc10702826` |
| `train.log` (full log of the 200-epoch run) | `e51bbf9e8fafff7da5dd94bfcbbb4baf` |

- **Architecture**: features v3 (CCMKP-style normalisations), CCMKP-style edge weights, hidden size 128.
- **Data**: the training set has 10 008 feasible nodes. These are the 8 008 stratified training nodes,
  minus 1 431 whose subtree is provably infeasible, plus 3 431 extra feasible nodes. VAL has 1 991
  nodes, of which 300 are used for the metrics. All nodes come from n ∈ {6, 8, 10} × N ∈ {20, 50, 100}.
- **Training settings**: 200 epochs; lr 3e-4 with a 2-epoch ramp, then cosine decay to 1e-5;
  batch 32; per-example clipping at 0.5; supervised warm-up (300 examples, T = 30, 10 epochs).
  The run took 8 h 40 on one CPU node.
- **Selection**: epoch 57 has the best mean VAL improvement of L(λ_GNN) over L(0) on feasible nodes
  (10.45 % mean, 6.81 % median).
- **Export check**: `validate_export_ckpt.py` on 12 real graphs gives max |TorchScript − eager| = 9.5e-7.
- **Reproduction**: see `gnn/ccpo/README.md` (pipeline steps 1–7, launcher `gnn/ccpo/run_train_ccpo.sh`).
