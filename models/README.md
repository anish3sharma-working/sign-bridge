# Model files

`hand_landmarker.task` is an upstream Google MediaPipe hand-landmark detector (Apache 2.0 per its official model card), used only to locate up to two hands and their 21 landmarks each. Its official source, notice, limitations, and checksum are in [NOTICE.md](NOTICE.md). It is not an ISL vocabulary or sign-recognition model.

No trained ISL recognition model is included. A successful bilateral SignBridge training run writes `signbridge_static_v4.joblib` and a matching JSON metadata sidecar into `%LOCALAPPDATA%\SignBridge\models\` (not this source folder).

The version-4 model input has **136 float values**: a 68-value slot for the signer’s anatomical left hand and another for the right hand. Each side slot has a presence flag, 63 wrist-anchored hand-pose values, frame-relative wrist x/y/z, and apparent scale. A missing side is represented by an all-zero slot. When `mirrored_input` is enabled (the default), source pixels are physically flipped horizontally before detection. `swap_handedness` applies an independent camera-specific correction to the MediaPipe side label and is enabled by default for this release. Webcam preview, training and inference use the same orientation and correction. Both settings are stored in model metadata and must match at inference.

The application checks artifact version, feature version, dimensions, estimator feature count, class mapping, mirrored-input orientation, and Left/Right correction. Earlier single-hand and bilateral v2/v3 artifacts are intentionally incompatible with v4 and must be retrained from local labeled samples. Evaluation metrics are computed from actual group-independent partitions.

**Security:** joblib uses Python object serialization. Only load model artifacts trained by your own trusted SignBridge installation. Do not open untrusted `.joblib` files.
