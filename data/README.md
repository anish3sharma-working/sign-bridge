# Local dataset notes

No ISL training dataset is bundled. Place only samples you are allowed to use in the app through its Dataset Manager. Imports are referenced in place; webcam captures are saved under `%LOCALAPPDATA%\SignBridge\dataset\raw\<label>\`.

Each database sample stores its label, media path/type, recording group, optional participant code, and creation time. Keep all frames from the same recording together. Use a stable, non-identifying participant code when you can; avoid names, phone numbers, or other personal details. The split routine groups by participant code when supplied, otherwise by recording group.

The baseline consumes one hand pose per image. For imported video, it extracts the middle frame only. Video import does not make the model dynamic-sign capable. Review labels and delete unusable entries in the app. Do not commit personal recordings or identifiable data to a public repository.
