# Additional References - Project 11: NASA Turbofan RUL Prediction

Independent literature scout pass for Phase 1 of the C-MAPSS RUL scaffold. Search performed 2026-05-08 against CrossRef (api.crossref.org/works) and verified live per DOI. Papers below were chosen independently before the final SOTA-gap check against `reports/references.md`. Every entry below resolves under `https://api.crossref.org/works/{doi}` with title and journal matching CrossRef metadata. Per project rule, only Author / Title / Journal / Year / DOI fields are kept; volume / issue / page numbers are intentionally stripped.

## State-of-the-art callout (gaps in current references.md)

The current `reports/references.md` covers CNN / LSTM / vanilla Bayesian baselines well but is missing five areas that are now standard in 2024-2026 C-MAPSS / aero-engine RUL papers and that this project SHOULD cite:

1. **Pure transformer encoders for C-MAPSS RUL.** The current bibliography cites Vaswani only inside the manuscript prose, not as a formal reference, and has no transformer-specific RUL paper. Add Fan 2024 (Sensors) and Kim 2025 (IEEE TIM, Koopman dual-branch transformer).
2. **Conformal / distribution-free uncertainty quantification.** Existing refs cover Bayesian DL uncertainty (Li 2021) but not conformal prediction, which is the dominant 2025-2026 paradigm for safety-critical PHM. Add Robinson 2026 (IJPHM) and Hattar 2025 (IEEE ISPCE).
3. **Physics-informed neural networks for aero-engine RUL.** No PINN reference is currently cited. Add Zhang 2024 (ASME GT2024) and Wang 2025 (PHM-Xian).
4. **Unsupervised domain adaptation benchmark on C-MAPSS.** No DA benchmark currently. Krokotsch 2026 (IEEE TASE) is the canonical "from inconsistency to unity" benchmark and should be cited as the comparator for any FD001-to-FD004 transfer claim.
5. **Recent C-MAPSS deep-learning survey.** Existing list cites Wu 2024 (Sensors) but not the parallel Isbilen 2025 (Aeronautical Journal) deep-learning + similarity-models survey, which is the most recent direct C-MAPSS-only review.

## Architectures - transformers and attention (2024-2026)

1. Fan Z, Li W, Chang K. A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine Remaining Useful Life Prediction. Sensors. 2024. DOI: 10.3390/s24030824
2. Kim E, Park S, Lee H, Ko S, Hwang E. Enhanced Remaining Useful Life Prediction for Turbofan Engines Using Spatiotemporal Koopman Dual-Branch Transformer. IEEE Transactions on Instrumentation and Measurement. 2025. DOI: 10.1109/tim.2025.3625335
3. Chen X. A novel transformer-based DL model enhanced by position-sensitive attention and gated hierarchical LSTM for aero-engine RUL prediction. Scientific Reports. 2024. DOI: 10.1038/s41598-024-59095-3
4. Ren S, Wu M, Zhou H. A transformer-based method for aircraft engine RUL prediction integrating dual-layer attention with BiLSTM. Results in Engineering. 2026. DOI: 10.1016/j.rineng.2026.109187
5. Hatipoglu A, Yilmaz E. A Matrix-Statistics-Aware Attention Mechanism for Robust RUL Estimation in Aero-Engines. Applied Sciences. 2025. DOI: 10.3390/app16010169
6. Dida M, Cheriet A, Belhadj M. Remaining Useful Life Prediction Using Attention-LSTM Neural Network of Aircraft Engines. International Journal of Prognostics and Health Management. 2025. DOI: 10.36001/ijphm.2025.v16i2.4274

## Physics-informed and hybrid models (2024-2026)

7. Zhang X, Lou J, Zhang J, Zheng Y, Xia Y. Aero-Engine Remaining Useful Life Prediction via Physics-Informed Self-Attention Encoder. Volume 4: Controls, Diagnostics, and Instrumentation, ASME Turbo Expo 2024. 2024. DOI: 10.1115/gt2024-124893
8. Wang A, Shi D, Wu B, Li X, Li Y. Physics-Informed and Data-driven Feature Fusion with Gated Convolution and Transformer for RUL Prediction of Aero-Engines. PHM-Xian. 2025. DOI: 10.1109/phm-xian66756.2025.11427557
9. Zhao Z, Qin X, Zhang J. A Spatiotemporal Fusion Enhanced Physics-Informed Neural Network for RUL Prediction of Aero-Engine. MAIC. 2025. DOI: 10.1109/maic67965.2025.11468727
10. Jiang F, Hou X, Xia M. Spatio-Temporal Attention-Based Hidden Physics-Informed Neural Network for Remaining Useful Life Prediction. SSRN preprint. 2024. DOI: 10.2139/ssrn.4845675

## Graph neural networks for aero-engine RUL (2024-2026)

11. Sun S, Ding H, Zhao Z. Spatio-Temporal Information Fusion with Graph Neural Networks for Aero-Engine Remaining Useful Life Prediction. PHM-Beijing. 2024. DOI: 10.1109/phm-beijing63284.2024.10874697
12. Tian R, Sun H, Mei B, Fu Y, Zhu W. A heterogeneous graph neural network with spatial-temporal and operating condition-aware message passing mechanism for RUL prediction of aero-engines. Advanced Engineering Informatics. 2026. DOI: 10.1016/j.aei.2026.104507

## Uncertainty quantification and conformal prediction (2024-2026)

13. Robinson C. Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression. International Journal of Prognostics and Health Management. 2026. DOI: 10.36001/ijphm.2026.v17i1.4724
14. Hattar H, Vincent J. Advancing Remaining Useful Life Estimation for Turbofan Engines: A Multi-Model Approach with Conformal Prediction. IEEE ISPCE. 2025. DOI: 10.1109/ispce64260.2025.11044907
15. Onilede M. Distribution-Free Uncertainty Quantification for Turbofan Engine Remaining Useful Life Using Conformalized Quantile Regression. TechRxiv preprint. 2026. DOI: 10.36227/techrxiv.177220374.46388084/v1
16. Xu W, Zio E. Uncertainty-Aware Prediction of Remaining Useful Life in Complex Systems. Annual Conference of the PHM Society. 2024. DOI: 10.36001/phmconf.2024.v16i1.4178
17. Li R, Zhan H, Yu J, Wang R. Turbofan Engine Remaining Useful Life Prediction Based on Multi-Task Uncertainty Analysis Model. SSRN preprint. 2024. DOI: 10.2139/ssrn.4853896
18. Alomari K. A Unified Uncertainty-Aware Multi-Task Framework for Robust Remaining Useful Life Prediction Under Distribution Shift. IEEE Access. 2026. DOI: 10.1109/access.2026.3685622

## Domain adaptation, transfer learning, and benchmarking on C-MAPSS

19. Krokotsch T, Ragab M, Wu M, Li X, Chen Z. From Inconsistency to Unity: Benchmarking Deep Learning-Based Unsupervised Domain Adaptation for RUL. IEEE Transactions on Automation Science and Engineering. 2026. DOI: 10.1109/tase.2025.3528469
20. Zeng Y, Liu F, Han G. Domain Adaptive Network Based on Transformer for Remaining Useful Life Prediction in Turbofan Engine. SSRN preprint. 2024. DOI: 10.2139/ssrn.5060158
21. Chen Y, Liu C. Turbofan Engine Remaining Useful Life Prediction Based on Sample Efficient Transfer Learning and Leveraging Large Language Model. SSRN preprint. 2024. DOI: 10.2139/ssrn.5036910

## Comparative studies, ensembles, and surveys (2024-2026)

22. Isbilen F, Bektas O, Konar M. Deep learning and similarity-based models for predicting turbofan engine remaining useful life: insights from the CMAPSS dataset. The Aeronautical Journal. 2025. DOI: 10.1017/aer.2025.25
23. Omer Z, Al Seiari A, Cen Z, Bilendo F. Machine Learning Regression Models for Turbofan Engines: A Comparative Study on Remaining Useful Life Prediction. ASME Turbo Expo 2024 Volume 1. 2024. DOI: 10.1115/gt2024-127544
24. Limone M, Marchili C, Ciolli L. MoE-FTR: A Mixture of Experts Framework for Turbofan engines Remaining Useful Life estimation. IJCNN. 2025. DOI: 10.1109/ijcnn64981.2025.11227920
25. Nimri S, Kohtz S. An Enhanced Feature Engineering and Stacking Ensemble Framework for Remaining Useful Life Prediction. IEEE ICPHM. 2025. DOI: 10.1109/icphm65385.2025.11062026

## Sensor-failure robustness, anomaly fusion, and digital-twin extensions

26. Tang D. A deep learning framework for remaining useful life prediction of turbofan engines with partial sensor failure. PLOS One. 2026. DOI: 10.1371/journal.pone.0347312
27. Duan Y, Zhang T, Shi D. Anomaly Detection and Remaining Useful Life Prediction for Turbofan Engines with a Key Point-Based Approach to Secure Health Management. Sensors. 2024. DOI: 10.3390/s24248022
28. Wan A, Zhu Z, Al-Bukhaiti K, Yin R, Yuan J. Digital Twin Framework for Interpretable Aero-Engine RUL Prediction via Multi-Branch LSTM and Unity Visualization. IEEE Transactions on Reliability. 2026. DOI: 10.1109/tr.2026.3686412
29. Seo J. Fault-Type-Aware Remaining Useful Life Prediction of Aircraft Engines Using an Integrated Deep Learning Framework. International Journal of Prognostics and Health Management. 2025. DOI: 10.36001/ijphm.2025.v16i2.4305

---

**Total NEW verified references: 29**
**Verification date: 2026-05-08**
**Verification source: api.crossref.org/works/{doi} (every DOI above resolved live with matching title and journal)**
