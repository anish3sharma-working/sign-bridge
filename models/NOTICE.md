# Third-party hand landmark detector notice

SignBridge optionally uses Google's MediaPipe Hand Landmarker task asset at `models/hand_landmarker.task`.

- Official download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
- Official task overview and model page: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- Official model card: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf
- The official model card states the hand tracking model is licensed under Apache License 2.0. The model card describes intended use around prominently displayed hands and cautions that the model is experimental and can degrade under conditions such as low light, occlusion, and motion blur.
- The detector estimates hand landmarks only. It is **not** an ISL sign recognizer, does not provide a trained ISL vocabulary, and must not be presented as such.

The model asset is distributed separately from SignBridge's MIT-licensed original source. Retain this notice and the model card with redistributions. No runtime download is performed. If you remove or replace the asset, ensure the replacement has a compatible license and document it here.


Asset integrity: the downloaded official task file is 7,819,105 bytes with SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
