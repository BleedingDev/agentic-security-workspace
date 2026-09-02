---
name: android-security-assessment
description: Android security assessment for an APK, AAB, application, emulator, or device.
---

Run `/scope-security-test` unless the scope ledger matches the target identity, assessment ID, and review date; run `/plan-security-assessment` unless its target identity, assessment ID, and review date match that ledger. Select a versioned MASVS/MASTG profile. Run `/mobsf`, `/jadx`, and `/apktool` for independent static views. Run `/native-binary-assessment` on native libraries and `/api-security-assessment` on remote services. When dynamic testing is in scope, record device and app state, then run `/adb`, `/frida`, and `/burp` as needed. Run `/verify-security-finding` on every material candidate.

Complete when every selected mobile control has correlated static or runtime evidence, or a reasoned terminal gap with impact; device cleanup is recorded.
