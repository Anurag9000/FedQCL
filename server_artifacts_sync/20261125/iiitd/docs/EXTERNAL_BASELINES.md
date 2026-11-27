# External baseline reproduction boundary

Selection adaptations in the current ledger are intentionally named mass_stratified and divfl_sketch.
Neither claims exact reproduction of another paper. Replay profiles likewise are not an implementation
of personalized Re-Fed importance scoring. FedWeIT requires parameter decomposition and adaptive
knowledge transfer rather than a scheduler flag.

Primary reference sources:

- Fraboni et al., Clustered Sampling: https://proceedings.mlr.press/v139/fraboni21a.html
- DivFL official implementation: https://github.com/melodi-lab/divfl
- Yoon et al., FedWeIT: https://proceedings.mlr.press/v139/yoon21b.html
- FedWeIT upstream: https://github.com/wyjeong/FedWeIT
- Li et al., Re-Fed: https://openaccess.thecvf.com/content/CVPR2024/html/Li_Towards_Efficient_Replay_in_Federated_Incremental_Learning_CVPR_2024_paper.html

Before an external method can enter the validated comparison matrix, pin its exact upstream revision,
verify its license/environment, implement a common data/task/evaluation interface, document its native
aggregation and transfer accounting, and run a reference sanity check. Do not silently change its
objective to fit the FEDQCL client. Upstream commands and hyperparameters must be derived from the
pinned upstream implementation, not guessed. No official baseline reproduction is claimed here.
