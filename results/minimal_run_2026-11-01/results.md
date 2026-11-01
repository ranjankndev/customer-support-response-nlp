# AspectForge v3 — minimal run results

Systems: s1_gemma2, s2_gemma3, s3_gemma3_crag, s4_raft_aspect, s5_raft_noaspect, s6_gemma3_noaspect  
Rows per system: 500  
Bootstrap: 1000 paired resamples, seed 42; a difference is called significant when the CI of the paired difference excludes 0.

## Overall

| System | rouge1 | rouge2 | rougeL | bertscore_f1 | lang_match | acs | aas |
|---|---|---|---|---|---|---|---|
| s1_gemma2 | 0.3895 [0.3767, 0.4017] | 0.1570 [0.1461, 0.1686] | 0.2824 [0.2702, 0.2946] | 0.9035 [0.9015, 0.9055] | 1.0000 [1.0000, 1.0000] | 0.7962 [0.7844, 0.8069] | 0.8209 [0.7953, 0.8476] |
| s2_gemma3 | 0.3561 [0.3454, 0.3668] | 0.1190 [0.1126, 0.1261] | 0.2325 [0.2245, 0.2410] | 0.8931 [0.8916, 0.8947] | 0.9980 [0.9940, 1.0000] | 0.8052 [0.7978, 0.8131] | 0.7890 [0.7720, 0.8058] |
| s3_gemma3_crag | 0.3561 [0.3454, 0.3668] | 0.1190 [0.1126, 0.1261] | 0.2325 [0.2245, 0.2410] | 0.8931 [0.8916, 0.8947] | 0.9980 [0.9940, 1.0000] | 0.8052 [0.7978, 0.8131] | 0.7890 [0.7720, 0.8058] |
| s4_raft_aspect | 0.5176 [0.4980, 0.5365] | 0.2951 [0.2746, 0.3167] | 0.4295 [0.4080, 0.4504] | 0.9272 [0.9240, 0.9302] | 1.0000 [1.0000, 1.0000] | 0.6929 [0.6836, 0.7018] | 0.6486 [0.6279, 0.6696] |
| s5_raft_noaspect | 0.5096 [0.4908, 0.5280] | 0.2903 [0.2702, 0.3105] | 0.4235 [0.4018, 0.4429] | 0.9263 [0.9232, 0.9292] | 1.0000 [1.0000, 1.0000] | — | — |
| s6_gemma3_noaspect | 0.3918 [0.3802, 0.4035] | 0.1462 [0.1377, 0.1548] | 0.2695 [0.2600, 0.2795] | 0.9019 [0.9002, 0.9035] | 1.0000 [1.0000, 1.0000] | — | — |

## DE

| System | rouge1 | rouge2 | rougeL | bertscore_f1 | lang_match | acs | aas |
|---|---|---|---|---|---|---|---|
| s1_gemma2 | 0.3720 [0.3544, 0.3886] | 0.1385 [0.1254, 0.1532] | 0.2691 [0.2533, 0.2851] | 0.9006 [0.8978, 0.9034] | 1.0000 [1.0000, 1.0000] | 0.8023 [0.7860, 0.8194] | 0.8393 [0.7938, 0.8797] |
| s2_gemma3 | 0.3452 [0.3306, 0.3583] | 0.1089 [0.1009, 0.1173] | 0.2249 [0.2140, 0.2353] | 0.8932 [0.8910, 0.8952] | 0.9960 [0.9880, 1.0000] | 0.8080 [0.7960, 0.8191] | 0.7872 [0.7590, 0.8160] |
| s3_gemma3_crag | 0.3452 [0.3306, 0.3583] | 0.1089 [0.1009, 0.1173] | 0.2249 [0.2140, 0.2353] | 0.8932 [0.8910, 0.8952] | 0.9960 [0.9880, 1.0000] | 0.8080 [0.7960, 0.8191] | 0.7872 [0.7590, 0.8160] |
| s4_raft_aspect | 0.4474 [0.4252, 0.4675] | 0.2065 [0.1877, 0.2264] | 0.3515 [0.3287, 0.3727] | 0.9175 [0.9136, 0.9209] | 1.0000 [1.0000, 1.0000] | 0.6843 [0.6705, 0.6975] | 0.6579 [0.6206, 0.6969] |
| s5_raft_noaspect | 0.4472 [0.4252, 0.4667] | 0.2101 [0.1913, 0.2285] | 0.3551 [0.3327, 0.3751] | 0.9180 [0.9142, 0.9214] | 1.0000 [1.0000, 1.0000] | — | — |
| s6_gemma3_noaspect | 0.3625 [0.3485, 0.3762] | 0.1203 [0.1105, 0.1301] | 0.2469 [0.2348, 0.2584] | 0.8979 [0.8956, 0.8999] | 1.0000 [1.0000, 1.0000] | — | — |

## EN

| System | rouge1 | rouge2 | rougeL | bertscore_f1 | lang_match | acs | aas |
|---|---|---|---|---|---|---|---|
| s1_gemma2 | 0.4070 [0.3900, 0.4235] | 0.1754 [0.1597, 0.1916] | 0.2957 [0.2792, 0.3127] | 0.9063 [0.9035, 0.9090] | 1.0000 [1.0000, 1.0000] | 0.7902 [0.7765, 0.8040] | 0.8061 [0.7696, 0.8406] |
| s2_gemma3 | 0.3670 [0.3514, 0.3827] | 0.1291 [0.1193, 0.1396] | 0.2400 [0.2289, 0.2524] | 0.8931 [0.8912, 0.8953] | 1.0000 [1.0000, 1.0000] | 0.8025 [0.7933, 0.8125] | 0.7904 [0.7678, 0.8119] |
| s3_gemma3_crag | 0.3670 [0.3514, 0.3827] | 0.1291 [0.1193, 0.1396] | 0.2400 [0.2289, 0.2524] | 0.8931 [0.8912, 0.8953] | 1.0000 [1.0000, 1.0000] | 0.8025 [0.7933, 0.8125] | 0.7904 [0.7678, 0.8119] |
| s4_raft_aspect | 0.5879 [0.5583, 0.6161] | 0.3837 [0.3498, 0.4153] | 0.5075 [0.4750, 0.5397] | 0.9368 [0.9322, 0.9414] | 1.0000 [1.0000, 1.0000] | 0.7014 [0.6889, 0.7140] | 0.6412 [0.6206, 0.6643] |
| s5_raft_noaspect | 0.5720 [0.5424, 0.6006] | 0.3706 [0.3395, 0.4021] | 0.4919 [0.4606, 0.5246] | 0.9346 [0.9301, 0.9392] | 1.0000 [1.0000, 1.0000] | — | — |
| s6_gemma3_noaspect | 0.4211 [0.4043, 0.4382] | 0.1721 [0.1614, 0.1834] | 0.2922 [0.2788, 0.3063] | 0.9058 [0.9036, 0.9083] | 1.0000 [1.0000, 1.0000] | — | — |

## Takeaways

- **s1_gemma2** is 0.0334 above s2_gemma3 on rouge1 (95% CI [0.0242, 0.0428], n=500) — significant.
- **s4_raft_aspect** is 0.1615 above s2_gemma3 on rouge1 (95% CI [0.1430, 0.1793], n=500) — significant.
- **s5_raft_noaspect** is 0.1536 above s2_gemma3 on rouge1 (95% CI [0.1347, 0.1707], n=500) — significant.
- **s6_gemma3_noaspect** is 0.0357 above s2_gemma3 on rouge1 (95% CI [0.0292, 0.0418], n=500) — significant.
- **s1_gemma2** is 0.0379 above s2_gemma3 on rouge2 (95% CI [0.0295, 0.0470], n=500) — significant.
- **s4_raft_aspect** is 0.1761 above s2_gemma3 on rouge2 (95% CI [0.1574, 0.1955], n=500) — significant.
- **s5_raft_noaspect** is 0.1713 above s2_gemma3 on rouge2 (95% CI [0.1527, 0.1895], n=500) — significant.
- **s6_gemma3_noaspect** is 0.0272 above s2_gemma3 on rouge2 (95% CI [0.0214, 0.0325], n=500) — significant.
- **s1_gemma2** is 0.0499 above s2_gemma3 on rougeL (95% CI [0.0405, 0.0593], n=500) — significant.
- **s4_raft_aspect** is 0.1970 above s2_gemma3 on rougeL (95% CI [0.1768, 0.2160], n=500) — significant.
- **s5_raft_noaspect** is 0.1910 above s2_gemma3 on rougeL (95% CI [0.1703, 0.2096], n=500) — significant.
- **s6_gemma3_noaspect** is 0.0371 above s2_gemma3 on rougeL (95% CI [0.0305, 0.0434], n=500) — significant.
- **s1_gemma2** is 0.0103 above s2_gemma3 on bertscore_f1 (95% CI [0.0087, 0.0118], n=500) — significant.
- **s4_raft_aspect** is 0.0340 above s2_gemma3 on bertscore_f1 (95% CI [0.0309, 0.0368], n=500) — significant.
- **s5_raft_noaspect** is 0.0331 above s2_gemma3 on bertscore_f1 (95% CI [0.0302, 0.0359], n=500) — significant.
- **s6_gemma3_noaspect** is 0.0087 above s2_gemma3 on bertscore_f1 (95% CI [0.0075, 0.0098], n=500) — significant.
- **s4_raft_aspect** is 0.1123 below s2_gemma3 on acs (95% CI [-0.1228, -0.1009], n=494) — significant.
- **s1_gemma2** is 0.0319 above s2_gemma3 on aas (95% CI [0.0021, 0.0604], n=130) — significant.
- **s4_raft_aspect** is 0.1403 below s2_gemma3 on aas (95% CI [-0.1684, -0.1124], n=130) — significant.

## Aspect contribution

S2 − S6 isolates the aspects without fine-tuning, S4 − S5 with it.

- s2_gemma3 − s6_gemma3_noaspect on rougeL: -0.0370 (0.2325 vs 0.2695).
- s2_gemma3 − s6_gemma3_noaspect on bertscore_f1: -0.0088 (0.8931 vs 0.9019).
- s4_raft_aspect − s5_raft_noaspect on rougeL: +0.0060 (0.4295 vs 0.4235).
- s4_raft_aspect − s5_raft_noaspect on bertscore_f1: +0.0009 (0.9272 vs 0.9263).

## Aspect coverage (ACS / AAS)

| System | ACS (all) | ACS (de) | ACS (en) | AAS (all) | AAS (de) | AAS (en) |
|---|---|---|---|---|---|---|
| s1_gemma2 | 0.7962 [0.7844, 0.8069] | 0.8023 [0.7860, 0.8194] | 0.7902 [0.7765, 0.8040] | 0.8209 [0.7953, 0.8476] | 0.8393 [0.7938, 0.8797] | 0.8061 [0.7696, 0.8406] |
| s2_gemma3 | 0.8052 [0.7978, 0.8131] | 0.8080 [0.7960, 0.8191] | 0.8025 [0.7933, 0.8125] | 0.7890 [0.7720, 0.8058] | 0.7872 [0.7590, 0.8160] | 0.7904 [0.7678, 0.8119] |
| s3_gemma3_crag | 0.8052 [0.7978, 0.8131] | 0.8080 [0.7960, 0.8191] | 0.8025 [0.7933, 0.8125] | 0.7890 [0.7720, 0.8058] | 0.7872 [0.7590, 0.8160] | 0.7904 [0.7678, 0.8119] |
| s4_raft_aspect | 0.6929 [0.6836, 0.7018] | 0.6843 [0.6705, 0.6975] | 0.7014 [0.6889, 0.7140] | 0.6486 [0.6279, 0.6696] | 0.6579 [0.6206, 0.6969] | 0.6412 [0.6206, 0.6643] |

## CRAG routes

- `s1_gemma2`: rag 100%
- `s2_gemma3`: rag 100%
- `s3_gemma3_crag`: rag 100%
- `s4_raft_aspect`: rag 100%
- `s5_raft_noaspect`: rag 100%
- `s6_gemma3_noaspect`: rag 100%

## Retrieval

| config           | split   | metric      |   mean |   ci_low |   ci_high |   n |
|:-----------------|:--------|:------------|-------:|---------:|----------:|----:|
| m3_hybrid_rerank | all     | top1_rougeL | 0.38   |   0.3549 |    0.4066 | 500 |
| m3_hybrid_rerank | all     | top1_cos    | 0.8573 |   0.8467 |    0.8679 | 500 |
| m3_hybrid_rerank | all     | intent@3    | 0.5513 |   0.5246 |    0.5813 | 500 |
| m3_hybrid_rerank | de      | top1_rougeL | 0.221  |   0.1921 |    0.2481 | 250 |
| m3_hybrid_rerank | de      | top1_cos    | 0.825  |   0.8113 |    0.839  | 250 |
| m3_hybrid_rerank | de      | intent@3    | 0.552  |   0.508  |    0.5933 | 250 |
| m3_hybrid_rerank | en      | top1_rougeL | 0.539  |   0.5032 |    0.5748 | 250 |
| m3_hybrid_rerank | en      | top1_cos    | 0.8895 |   0.8754 |    0.9039 | 250 |
| m3_hybrid_rerank | en      | intent@3    | 0.5507 |   0.5093 |    0.5947 | 250 |

## Aspect heads

| head        | gold_column   | split   |   n |   accuracy |   f1_macro | majority_label    |   majority_accuracy |   majority_f1_macro |   beats_majority | source     | label           |   distinct_values | top_value       |   top_share |   collapsed_share |
|:------------|:--------------|:--------|----:|-----------:|-----------:|:------------------|--------------------:|--------------------:|-----------------:|:-----------|:----------------|------------------:|:----------------|------------:|------------------:|
| intent      | queue         | all     | 500 |      0.3   |     0.256  | Technical Support |               0.3   |              0.0462 |                1 | heads      | nan             |               nan | nan             |     nan     |           nan     |
| intent      | queue         | de      | 250 |      0.28  |     0.227  | Technical Support |               0.308 |              0.0471 |                1 | heads      | nan             |               nan | nan             |     nan     |           nan     |
| intent      | queue         | en      | 250 |      0.32  |     0.2717 | Technical Support |               0.292 |              0.0452 |                1 | heads      | nan             |               nan | nan             |     nan     |           nan     |
| ticket_type | type          | all     | 500 |      0.804 |     0.8043 | Incident          |               0.402 |              0.1434 |                1 | heads      | nan             |               nan | nan             |     nan     |           nan     |
| ticket_type | type          | de      | 250 |      0.824 |     0.8122 | Incident          |               0.408 |              0.1449 |                1 | heads      | nan             |               nan | nan             |     nan     |           nan     |
| ticket_type | type          | en      | 250 |      0.784 |     0.7913 | Incident          |               0.396 |              0.1418 |                1 | heads      | nan             |               nan | nan             |     nan     |           nan     |
| urgency     | priority      | all     | 500 |      0.462 |     0.4481 | medium            |               0.362 |              0.1772 |                1 | heads      | nan             |               nan | nan             |     nan     |           nan     |
| urgency     | priority      | de      | 250 |      0.448 |     0.4337 | medium            |               0.368 |              0.1793 |                1 | heads      | nan             |               nan | nan             |     nan     |           nan     |
| urgency     | priority      | en      | 250 |      0.476 |     0.4603 | medium            |               0.356 |              0.175  |                1 | heads      | nan             |               nan | nan             |     nan     |           nan     |
| nan         | nan           | nan     | 500 |    nan     |   nan      | nan               |             nan     |            nan      |              nan | sft_labels | intent          |                 7 | general_inquiry |       0.406 |             0.406 |
| nan         | nan           | nan     | 500 |    nan     |   nan      | nan               |             nan     |            nan      |              nan | sft_labels | ticket_type_nli |                 4 | complaint       |       0.504 |           nan     |

## CRAG sweep over the real score range (offline, 150 val rows)

bge-reranker-v2-m3 top scores sit at median 0.998 (min 0.70), so the planned 0.05-0.60 grid never routes to no_context and S3 == S2. Re-scoring the saved rag / no-context pairs over the actual range:

|   threshold |   rougeL |   rag_share |   n | best   |
|------------:|---------:|------------:|----:|:-------|
|       0     |   0.238  |      1      | 150 | False  |
|       0.9   |   0.2381 |      0.98   | 150 | True   |
|       0.95  |   0.2372 |      0.9467 | 150 | False  |
|       0.97  |   0.2353 |      0.9067 | 150 | False  |
|       0.98  |   0.2354 |      0.8867 | 150 | False  |
|       0.99  |   0.2325 |      0.7933 | 150 | False  |
|       0.993 |   0.2317 |      0.7467 | 150 | False  |
|       0.995 |   0.2305 |      0.6933 | 150 | False  |
|       0.997 |   0.2301 |      0.6267 | 150 | False  |
|       0.999 |   0.2233 |      0.4533 | 150 | False  |
|       1.01  |   0.1903 |      0      | 150 | False  |

Always retrieving (0.238) is best; no context alone gives 0.190. CRAG brings no gain on this data.

## RAFT training

```
[RAFT] Training on 3000 examples
[RAFT] precision=fp32 (sm major 7)
[RAFT] Trainable tensors: 476
[RAFT] train=2850 val=150
[RAFT] SFTConfig does not accept warmup_ratio; dropped
[RAFT] UnslothSFTTrainer does not accept tokenizer; dropped
{'loss': '2.187', 'grad_norm': '0.7118', 'learning_rate': '0.0001999', 'epoch': '0.05614'}
{'loss': '1.213', 'grad_norm': '0.5746', 'learning_rate': '0.0001972', 'epoch': '0.1123'}
{'loss': '1.157', 'grad_norm': '0.7206', 'learning_rate': '0.0001914', 'epoch': '0.1684'}
{'loss': '1.166', 'grad_norm': '0.8924', 'learning_rate': '0.0001826', 'epoch': '0.2246'}
{'loss': '1.156', 'grad_norm': '1.246', 'learning_rate': '0.000171', 'epoch': '0.2807'}
Unsloth: Not an error, but `num_items_in_batch` could not be counted for Gemma3ForConditionalGeneration's labels, so each micro-batch loss is a mean.
{'loss': '1.153', 'grad_norm': '1.435', 'learning_rate': '0.0001572', 'epoch': '0.3368'}
{'loss': '1.154', 'grad_norm': '1.36', 'learning_rate': '0.0001414', 'epoch': '0.393'}
{'loss': '1.157', 'grad_norm': '1.512', 'learning_rate': '0.0001243', 'epoch': '0.4491'}
{'loss': '1.168', 'grad_norm': '1.351', 'learning_rate': '0.0001064', 'epoch': '0.5053'}
{'loss': '1.103', 'grad_norm': '1.531', 'learning_rate': '8.822e-05', 'epoch': '0.5614'}
{'loss': '1.095', 'grad_norm': '1.525', 'learning_rate': '7.048e-05', 'epoch': '0.6175'}
{'loss': '1.08', 'grad_norm': '1.432', 'learning_rate': '5.371e-05', 'epoch': '0.6737'}
{'loss': '1.154', 'grad_norm': '1.757', 'learning_rate': '3.847e-05', 'epoch': '0.7298'}
{'loss': '1.062', 'grad_norm': '1.473', 'learning_rate': '2.524e-05', 'epoch': '0.786'}
{'loss': '1.083', 'grad_norm': '1.617', 'learning_rate': '1.448e-05', 'epoch': '0.8421'}
{'loss': '1.106', 'grad_norm': '1.477', 'learning_rate': '6.523e-06', 'epoch': '0.8982'}
{'loss': '1.113', 'grad_norm': '1.834', 'learning_rate': '1.644e-06', 'epoch': '0.9544'}
{'eval_loss': '1.127', 'eval_runtime': '74.53', 'eval_samples_per_second': '2.013', 'eval_steps_per_second': '0.51', 'epoch': '1'}
{'train_runtime': '4368', 'train_samples_per_second': '0.652', 'train_steps_per_second': '0.041', 'train_loss': '1.189', 'epoch': '1'}
[RAFT] Adapter saved -> output/raft/gemma3-4b-aspect-raft
[RAFT] --- en distractor=False with_aspects=True ---
[RAFT] gold : To resolve your scanner issue, could you kindly share your computer and scanner model along with the version of macOS you are running so that I can guide you on the appropriate drivers and updates?
[RAFT] model: I am here to assist you with your scanner issue. Please provide the model number of your computer and scanner, along with your macOS version, so I can offer you the correct drivers and updates.
[RAFT] --- de distractor=False with_aspects=True ---
[RAFT] gold : Wir entschuldigen uns für die dadurch verursachten Unannehmlichkeiten. Unser Team beschäftigt sich bereits damit, den Fehler so schnell wie möglich zu beheben. Bitte starten Sie die Werkzeuge neu und
[RAFT] model: Wir entschuldigen uns für die Unannehmlichkeiten, die durch den Netzwerkausfall verursacht wurden. Unser Team arbeitet daran, das Problem zu beheben, und wird es so schnell wie möglich lösen. Bitte ve
[RAFT] --- de distractor=True with_aspects=True ---
[RAFT] gold : Wir freuen uns, Ihnen zu raten, Sicherung medizinischer Daten in HubSpot CRM PostgreSQL-Umgebungen durchzuführen. Um mit den Vorschriften von HIPAA in Einklang zu bleiben, ist es wichtig, umfassende S
[RAFT] model: Ratung für Sicherung medizinischer Daten in HubSpot CRM PostgreSQL-Umgebungen.
```
