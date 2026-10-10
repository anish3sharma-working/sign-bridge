# SignBridge

**SignBridge** is a Windows-first, local desktop application for collecting and recognizing a small vocabulary of static Indian Sign Language (ISL) hand gestures. It uses Python and native Qt Widgets. The modules are separated so a future mobile client can reuse stable data and model interfaces without changing this desktop-first product direction.

> **Honest scope:** no trained ISL sign-recognition model or dataset is included. Until you collect properly labeled examples and train a compatible model, the application will show a setup state and will not produce sign predictions. The detector now locates up to two hands and labels the signer’s anatomical left/right hand; static model input includes side-specific pose, wrist position, and scale features. It does not translate unrestricted continuous ISL, model movement, facial/non-manual signals, spatial grammar, or produce grammatically complete English/Hindi sentences.

## Implemented features

SignBridge includes a native dashboard, live-recognition screen, dataset manager, model-training screen, transcript history, settings, and privacy section. Camera capture uses a background Qt worker; the user starts it explicitly. The worker keeps a one-frame UI handoff, overlays landmarks optionally, and reports approximate FPS and inference latency.

The computer-vision path uses the official MediaPipe Hand Landmarker task asset with `num_hands=2`. MediaPipe documents its handedness labels assuming mirrored selfie input, so SignBridge physically flips frames before detection and shows that same selfie-mirrored preview by default ([MediaPipe handedness guidance](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)). Each detected hand receives 21 wrist-anchored normalized landmarks; the model input is a 136-value left/right-slot feature vector that also preserves frame-relative wrist position and hand scale. Live frames show separate colored overlays with side labels and detector scores. A configurable threshold balances missed detections against false positives; lowering it can increase false detections. The gesture-recognition baseline is a locally trained scikit-learn Random Forest. It uses actual user-provided media, skips and reports samples that fail processing, and keeps participants or recordings together across train, validation, and test partitions. No scores or predictions are fabricated.

Other implemented features include local image/video imports with still/first-frame preview, webcam frame collection, class and sample metadata, dataset metadata export, group leakage checks and insufficient-group warnings, versioned model metadata, confidence rejection, temporal stability, and explicit manual prediction acceptance. The **Flip camera/media horizontally (selfie orientation)** setting physically flips pixels before detection and mirrors the live preview. The separate **Swap detected Left/Right labels** correction is enabled by default in response to the reported reversed labels, and can be switched off if calibration shows that your camera already reports sides as expected. Both controls affect training and inference; changing either requires retraining. The bilateral feature schema is version 4: earlier models are intentionally rejected. Transcript editing, undo/clear/copy/save, text export, basic punctuation/spacing normalization, optional local text-to-speech, SQLite persistence, adjustable 5–30 FPS inference frequency, JSON settings, history deletion, and local privacy disclosures are also included.

## Architecture

```text
PySide6 Qt Widgets
  ├─ Dashboard / Live / Dataset / Training / History / Settings & Privacy
  ├─ QThread camera worker ─ OpenCV ─ MediaPipe Hand Landmarker
  ├─ Recognition: feature contract → group-trained Random Forest → rejection/stability filter
  └─ Local services: SQLite, JSON settings, text export, pyttsx3

Training: local media → MediaPipe → normalized hand features → grouped partitions
          → Random Forest → validation + held-out test report → local model artifact
```

The desktop UI communicates with worker results through Qt signals. Recognition and training are separated from UI widgets. The app remains a desktop application: no web server, cloud service, account, external inference, or mobile app is set up by this implementation. A separate future mobile client can be designed against stable interfaces after the feature and model formats mature.

## Runtime and dependencies

The project targets **64-bit CPython 3.14 on Windows**, the accessible Python runtime on the development device. The validated direct package versions are PySide6 6.12.0, MediaPipe 1.1.0, OpenCV-contrib-python 4.14.0.94, NumPy 2.5.3, scikit-learn 1.9.1, pyttsx3 2.99, joblib 1.6.0, pytest 9.1.1, and PyInstaller 6.22.3. These exact pins are listed in `requirements.txt`. The standard PySide6 wheel is used because the Essentials-only wheel did not provide a working Windows module loader in validation; PySide6 also bundles browser/PDF Qt add-ons that this app does not use. The camera utility tries DirectShow, Media Foundation, and OpenCV’s default backend; this development device exposed no usable camera at indices 0–4, so camera capture remains unverified here. MediaPipe 1.1.0 uses the Tasks API and requires a local hand-landmarker model asset; the project includes Google’s official task file and its license/source notice in `models/NOTICE.md`. It is a hand detector only, not a sign-recognition model. No model is downloaded at runtime.

The project does not use PyTorch because this release has no sequence model. Dynamic-sign training should be separately implemented and evaluated before that mode is exposed.


SignBridge’s MIT license applies only to original project code. The MediaPipe task asset is Apache-2.0 per its official model card. Qt for Python, OpenCV, NumPy, scikit-learn, pyttsx3, and PyInstaller retain their own licenses and notices. No dataset or pretrained ISL sign-recognition model is distributed. Check each component’s terms before redistribution.

## Requirements

- Windows 10/11, 64-bit, PowerShell.
- 64-bit CPython 3.14.
- A webcam for live capture (optional for the UI and unit tests).
- Local Windows speech voices for speech output (optional; available language depends on Windows configuration).
- Local disk space for dependencies, user samples, model artifacts, database, and logs.

## Install and launch (PowerShell)

From the project folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\check_environment.py
python app.py
```

If PowerShell blocks activation, do not change the machine execution policy. Call the virtual-environment interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe app.py
```

## One-click desktop launch

For subsequent launches, double-click `Launch SignBridge.bat` in the project folder or use the **SignBridge** desktop shortcut. The launcher creates `.venv` if needed, verifies it is 64-bit Python 3.14, installs/verifies the pinned requirements, checks the environment, and then opens the app. The first dependency installation needs an internet connection; no account or paid service is required.

To create the desktop shortcut from your own Windows session, double-click `Create SignBridge Desktop Shortcut.bat` once. It creates `SignBridge.lnk` on that Windows account’s Desktop without administrator access or execution-policy changes. If a different shortcut with that name already exists, it will not overwrite it. You can also double-click `Launch SignBridge.bat` directly from the project folder without creating a shortcut.

To probe camera indices and verify frame access, run `python scripts\check_camera.py`. It opens each candidate camera briefly and releases it. If no device is found, check Windows **Settings → Privacy & security → Camera**, choose the available camera index in SignBridge, and close other apps that may be using the device.

## Dataset organization and collection

No ISL dataset is distributed. Add gesture classes in **Dataset Manager**, then collect examples using the webcam or import media that you are permitted to use.

For webcam collection, open **Live recognition**, choose a label, provide a recording group and (when useful) a non-identifying participant code, start the camera, and save a frame. The preview is horizontally mirrored by default to behave like a selfie camera. Detection runs on that same mirrored view, and each detected hand is labeled as anatomical **Left** or **Right**. If the app labels your known left hand as Right, open **Settings & Privacy** and toggle **Swap detected Left/Right labels**; verify using one known hand before retraining. Capture both hands for signs that use two hands and keep settings consistent. Both **Flip camera/media horizontally** and **Swap detected Left/Right labels** affect training and inference; changing either requires retraining. Older settings are migrated automatically. Change the recording group for each distinct recording session. For local imports, source files remain at their current paths and SignBridge stores metadata and references. A batch of imported files shares one recording-group value by default; set the field explicitly to reuse a group across class-specific imports from the same recording.

Use labels reviewed by an ISL-fluent person where possible. Participant codes take precedence for split grouping; otherwise the recording group is used. If participant identity is missing, participant-independent evaluation cannot be guaranteed. The split routine requires at least three distinct groups per class and class coverage in each partition. It may require more than three groups to create class-complete training, validation, and test data; the app reports failure rather than generating misleading metrics.

Supported image extensions are JPG/JPEG, PNG, BMP, and WEBP. Supported video extensions are MP4, AVI, MOV, MKV, and WEBM, subject to local codec support. For static training, an image is processed directly and an imported video contributes only its **middle frame**. Video import does not make the system capable of recognizing movement or dynamic signs.

Sample metadata includes label, source path, media type, recording group, optional participant code, creation time, and current split. Newly captured webcam files are saved under `%LOCALAPPDATA%\SignBridge\dataset\raw\`. Imported media is not copied. Deleting a sample entry deletes a webcam-captured file stored in SignBridge’s raw folder, but leaves imported source files untouched. Dataset metadata export can include participant codes and source paths; review it before sharing.

## Train and evaluate

1. Add at least two static gesture classes.
2. Add real, correctly labeled media for each class across several independent recording groups. Include both hands for classes that depend on two-hand pose or relative placement.
3. In **Model Training**, select **Start training**.
4. Training extracts hand landmarks locally, reports unreadable media or missing hands, and uses group-disjoint train, validation, and test partitions.
5. The app fits a Random Forest on the training partition and computes validation and held-out test metrics from actual predictions. Split assignments are stored with the sample metadata.
6. The model and JSON metadata sidecar are saved under `%LOCALAPPDATA%\SignBridge\models\`; the SQLite database records the model and report.

The displayed test metrics describe only the collected dataset and its split. They are not claims about general ISL performance. Random Forest vote fractions are uncalibrated scores, not probabilities of correctness. Use an appropriate threshold and manual confirmation.

Model artifacts use Python joblib serialization. **Only load artifacts created by a trusted SignBridge run. Untrusted joblib files may execute code.** The application has no external model-import control; it only loads the latest recorded artifact from its own local model directory.

## Live recognition, transcripts, and speech

Start the camera explicitly. The detector labels up to two hands (anatomical Left/Right) and shows each hand’s detection score; adjust **Hand detector confidence** in settings if detections are missed, keeping in mind that a lower threshold may increase false detections. When a compatible local ISL model has been trained, the UI can show top candidates and scores. Temporal stability and the configured model threshold filter predictions; low scores are shown as uncertain. A stable sign still requires the user to click **Add prediction**. No repeated video-frame prediction is automatically added to the transcript.

Transcripts are editable and saved locally. The optional normalizer changes spacing and punctuation only; it does not infer grammar or add words. Local text-to-speech uses `pyttsx3` and Windows voices in a background thread. It makes no online request. The app does not claim Hindi/Indian-language speech unless a suitable voice is installed locally. The camera is not recorded by default.

## Storage and privacy

App data is stored in `%LOCALAPPDATA%\SignBridge\`: SQLite database, JSON settings, local logs, webcam sample images, and trained recognition models. Imported media remains at its original path. The camera opens only after the user selects **Start camera** and is released on stop, when leaving the live screen, or on application close. Webcam frames, landmarks, training, and inference remain local. The app has no telemetry, analytics, account, external API, cloud inference, or automatic video recording. It does not access the microphone.

Avoid personal names or identifying details in labels and participant codes. Use Dataset Manager and History to remove samples and saved transcripts. SQLite stores only the metadata needed for those functions. Do not share databases or sample folders publicly if they contain sensitive or identifiable information.

## Tests

Run the automated tests without a webcam:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The current suite passes **29 automated tests**, including mocked two-hand results, left/right correction, physical input flipping, legacy settings migration, feature-slot encoding, mirrored-model and correction compatibility, a real local task-model initialization on a controlled blank frame, and mocked backend-fallback checks. The one-click launcher setup-only check and shortcut helper (tested in a disposable folder) also passed. The actual Desktop shortcut was not placed by this session because Windows denied direct Desktop-folder writes; run `Create SignBridge Desktop Shortcut.bat` from your Windows account to create it. Numerical fixtures and mocked hands are software-test data only, not genuine ISL examples or recognition-accuracy evidence. An offscreen UI smoke test initialized all six sections and the hand correction control. The host camera probe found no usable camera at indices 0–4 via DirectShow, Media Foundation, or the default backend, so real left/right detection quality, capture, and overlay remain hardware-unverified. A local speech engine and voice are available, but audio playback was not triggered.

## Optional Windows executable

Packaging is optional and must be tested on a clean Windows machine. From the activated environment:

```powershell
python -m PyInstaller --noconfirm --clean --windowed --name SignBridge --add-data "models\hand_landmarker.task;models" app.py
```

The output is in `dist\SignBridge\` (one-folder mode). Qt plugins, MediaPipe, OpenCV DLLs, and installed speech voices require packaging checks. Do not include user datasets, camera captures, databases, or trained `.joblib` models. No standalone executable is claimed until it has been built and tested separately.

## Limitations and next steps

The detector supports up to two hands and the static model input preserves both hands separately, but two-hand sign classification still requires sufficient genuine labeled two-hand data and has not been hardware-validated here. The system does not model sign movement, body/face or non-manual markers, sign grammar, continuous segmentation, or unrestricted translation. No evaluated ISL vocabulary is included. Training can be refused when group data is insufficient. Stability filtering is a simple frame-window rule, not linguistic modeling. Camera access, handedness on mirrored input, performance, and available speech voices depend on the target device and must be checked there.

Next steps are to collect consented, reviewed recordings from multiple participants; improve group-aware dynamic datasets and sequence evaluation; calibrate rejection thresholds; validate with ISL users; and design a distinct mobile client only after the local desktop contracts and recognition scope are established.

## Compatibility and model references

- MediaPipe Python setup: <https://developers.google.com/edge/mediapipe/solutions/setup_python>
- MediaPipe Hand Landmarker task documentation and official model bundle: <https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker>
- Official model card and Apache-2.0 notice: <https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf>
- PySide6 package information: <https://pypi.org/project/PySide6/>
