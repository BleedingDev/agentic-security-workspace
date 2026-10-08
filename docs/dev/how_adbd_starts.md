# How adbd starts

The `adbd` service life cycle is managed by the [init](../../../../../system/core/init/README.md) process.
The daemon will be started when the two following conditions are true.
1. The device is in developer mode.
2. Adb over USB or adb over Wifi are enabled.

`init` itself doesn't have any special knowledge about adbd. Everything it needs to know comes from the various .rc files
(telling it what to do when various properties are set) and the processes that set those properties.

There are two main scenarios where init will start `adbd`. When the device boots and when a user runs a device into
"developer mode".

## When the device boots

The behavior of `init` is controlled by `.rc` files, commands, and system properties.

- The `adbd` service is described [here](https://cs.android.com/android/platform/superproject/main/+/main:packages/modules/adb/apex/adbd.rc;drc=a9b3987d2a42a40de0d67fcecb50c9716639ef03).
- The [rc language](../../../../../system/core/init/README.md) tie together properties, commands, and services.

When a device boots, the script init.usb.rc [checks](https://cs.android.com/android/platform/superproject/main/+/main:system/core/rootdir/init.usb.rc;l=109;drc=e34549af332e4be13a2ffb385455280d4736c1a9)
if persistent property `persist.sys.usb.config` is set, in which case the values is copied into `sys.usb.config`.
When this value is written, it [triggers](https://cs.android.com/android/platform/superproject/main/+/main:system/core/rootdir/init.usb.rc;l=47;drc=e34549af332e4be13a2ffb385455280d4736c1a9) `init` to run `start adbd`.

## When the device is already booted

When the device is up and running, it could be in "Developer Mode" but `adbd` service may not be running. It is only
after the user toggles "Developer options" -> "USB debugging" or "Developer options" -> "Wireless debugging" via the GUI that `adbd` starts.

Note that the previous description is valid for `user` builds. In the case of `userdebug` and `eng`, properties set
at build-time, such as `ro.adb.secure` or `persist.sys.usb.config`, will automate adbd starting up and disable authentication.

Four layers are involved.

1. GUI USB / GUI Wireless
2. AdbSettingsObserver
2. AdbService
3. init process


### GUI (USB)

1. The confirmation dialog is displayed from [AdbPreferenceController.showConfirmationDialog](https://cs.android.com/android/platform/superproject/main/+/main:packages/apps/Settings/src/com/android/settings/development/AdbPreferenceController.java;l=48;drc=1b8c0fdfdb9a36f691402513258b26036c41667f).
2. Validation is performed in [AdbPreferenceController.writeAdbSettings](https://cs.android.com/android/platform/superproject/main/+/main:frameworks/base/packages/SettingsLib/src/com/android/settingslib/development/AbstractEnableAdbPreferenceController.java;l=133;drc=aaea2d2266d29b3881f452899b79fb9e71525c3b) once the dialog is validated by user
3. `Settings.Global.ADB_ENABLED` is set.

```
Settings.Global.putInt(mContext.getContentResolver(),
Settings.Global.ADB_ENABLED, enabled ? ADB_SETTING_ON : ADB_SETTING_OFF);
```

### GUI (Wireless)
In the case of "Wireless debugging" toggle, the same kind of interaction leads to `ADB_WIFI_ENABLED` being set.

```
Settings.Global.putInt(mContext.getContentResolver(), Settings.Global.ADB_WIFI_ENABLED , 1);
```
### AdbSettingsObserver
1. Both `ADB_ENABLED` and `ADB_WIFI_ENABLED` are monitored by [AdbSettingsObserver](https://cs.android.com/android/platform/superproject/main/+/main:frameworks/base/services/core/java/com/android/server/adb/AdbService.java;l=208;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b).

2. When a change is detected, the Observers calls [AdbService::setAdbEnabled](https://cs.android.com/android/platform/superproject/main/+/main:frameworks/base/services/core/java/com/android/server/adb/AdbService.java;l=213;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b).

### AdbService

1. [AdbService.startAdbd](https://cs.android.com/android/platform/superproject/main/+/main:frameworks/base/services/core/java/com/android/server/adb/AdbService.java;l=480;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b) is called. This talks to the `init` process by setting `ctl.start` or `ctl.stop` to "adbd".
This step is equivalent to `.rc` files `start adbd` and `stop adbd`.

### USBDeviceManager (USB only)

If USB is involved (as opposed to ADB Wifi), ([USBDeviceManager.onAdbEnabled](https://cs.android.com/android/platform/superproject/main/+/main:frameworks/base/services/usb/java/com/android/server/usb/UsbDeviceManager.java;l=1090;drc=e36f88c420fe00112e11e85634851d047c0b623e)
) is called to recompose the gadget functions. As a side effect, persistent property `persist.sys.usb.config`
is set so `init` will automatically start `adbd` service on the next device start.

1. `MSG_ENABLE_ADB` message is sent from [onAdbEnabled](https://cs.android.com/android/platform/superproject/main/+/main:frameworks/base/services/usb/java/com/android/server/usb/UsbDeviceManager.java;l=1090;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b).

2. In [UsbDeviceManager.setAdbEnabled](https://cs.android.com/android/platform/superproject/main/+/main:frameworks/base/services/usb/java/com/android/server/usb/UsbDeviceManager.java;l=780;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b) property `persist.sys.usb.config` is set.

3. The manager needs to recompose the functions into a gadget.
    1. [UsbDeviceManager.setEnabledFunctions](https://cs.android.com/android/platform/superproject/main/+/main:frameworks/base/services/usb/java/com/android/server/usb/UsbDeviceManager.java;l=2422;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b).
    2. [UsbDeviceManager.setUsbConfig()](https://cs.android.com/android/platform/superproject/main/+/main:frameworks/base/services/usb/java/com/android/server/usb/UsbDeviceManager.java;l=2376;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b).

### init

`init` [monitors](https://cs.android.com/android/platform/superproject/main/+/main:system/core/init/property_service.cpp;l=551;drc=8067bd819f42be5512cdab8aaa3b0e9b4dba2369)
properties `ctl.start` and `ctl.stop` and interprets changes
as requests to start/stop a service. See `init` built-in commands (such as `start` and `stop`) [here](https://cs.android.com/android/platform/superproject/main/+/main:system/core/init/builtins.cpp;l=1334;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b).

To let other systems observe services' lifecycle, `init` [sets properties](https://cs.android.com/android/platform/superproject/main/+/main:system/core/init/service.cpp;l=179;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b) with known prefixes.
- Lifecycle: `init.svc.SERVICE_NAME` (`init.svc.adbd`), which can be set to "running", "stopped", "stopping" (see [Service::NotifyStateChange](https://cs.android.com/android/platform/superproject/main/+/main:system/core/init/service.cpp;l=172;drc=6474abd265cae9ccbe4e5d9ad37959215dcf564b) for all possible values).
- Bootime: `ro.boottime.SERVICE_NAME` (`ro.boottime.adbd`).
