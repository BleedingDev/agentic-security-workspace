# ADB Debugging page

## Address sanitizer

### Host

When you build you not only get an `adb` executable, you also get an `adb_asan`
which is built with clang's address sanitizer.

### Device

Use HWASan (Hardware-assisted AddressSanitizer). This is done via `lunch`
with an `hwasan` suffixed <product> (e.g.: `lunch aosp_panther_hwasan-trunk_staging-userdebug`)
(for reminder, the lunch format is <product>-<release>-<variant>).

## Logs

### Host

Enable logs and cycle the server.

```
$ export ADB_TRACE=all
$ adb server nodaemon
```

The environment variable `ADB_TRACE` is also checked by the adb client.

#### libusb
Libusb log level can be increased via environment variable `LIBUSB_DEBUG=4` (and restarting the server).
See libusb documentation for available [log levels](https://libusb.sourceforge.io/api-1.0/group__libusb__lib.html#ga2d6144203f0fc6d373677f6e2e89d2d2).

### Device

#### adbd
On the device, `adbd` does not read `ADB_TRACE` env variable. Instead it checks property `persist.adb.trace_mask`.
Set it and then cycle `adbd`.

```
$ adb shell su 0 setprop persist.adb.trace_mask 1
$ adb shell su 0 pkill adbd
```

`adbd` will write logs in `/data/adb`. The filename depends on what time `adbd` started (e.g.:`adb-2024-10-08-17-06-21-4611`).

#### Framework

To log components living in Framework, several methods are available depending on how much
needs to be seen.

The log level of each component can be changed.
```
adb shell setprop log.tag.all VERBOSE
adb shell setprop log.tag.AdbDebuggingManager D
```

Alternatively, components' log levels can be set directly on `logcat` command-line.
```
adb logcat AdbDebuggingManager:D *:S
```

#### mdnsResponder

mdnsResponder is the lib in charge of mDNS service publishing on the device. Enabling logs
requires recompiling it with the following changes.

Change `mDNSDebug.c`.
```
mDNSexport int mDNS_LoggingEnabled = 1;
```
Change `Android.bp`.
```
-DMDNS_DEBUGMSGS=2
```




