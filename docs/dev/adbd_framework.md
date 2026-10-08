# How ADBd and Framework communicate

## adbd_auth

The recommended way is to use `libadbd_auth` (frameworks/native/libs/adbd_auth).
It is a bidirectional socket originally used to handle authentication messages (hence the name).
It has since  evolved to carry other categories of messages.

```
        ┌────────────┐               ┌─────────────────────┐
        │ ADBService ◄───────────────► AdbDebuggingManager │
        └────────────┘               └──────────▲──────────┘
                                                │
                                     ┌──────────▼──────────┐
                                     │  AdbDebuggingThread │
                                     └──────────▲──────────┘
                                                │
   Framework                            ┌───────▼───────┐
   ─────────────────────────────────────┤ "adbd" socket ├─────────
   ADBd                                 └───────▲───────┘
                                                │
           ┌───────┐                     ┌──────▼─────┐
           │ ADBd  ◄─────────────────────► adbd_auth  │
           └───────┘                     └────────────┘
```

Example of usages (adbd-framework direction, packet header):

- [>> DD] Upon authentication, prompt user with a window to accept/refuse adb server's public key.
- [<< OK] Upon authentication, tell adbd the user accepted the key.
- [<< KO] Upon authentication, tell adbd the user refused the key.
- [>> DC] When a device disconnects.
- [>> TP] When the TLS Server starts, advertise its TLS port.
- [>> WE] When a TLS device connects.
- [>> WF] When a TLS device disconnects.

## System properties

A hacky way which should be avoided as much as possible is to use system property setter + getter. There
are threads listening on system property changes in both adbd and framework. See examples as follows.

- adbd writes `service.adb.tls.port`, framework uses a thread to monitor it.
- framework writes `persist.adb.tls_server.enable`, adbd uses a thread to monitor it.

If you are an ADB maintainer or/and have a few spare cycles, it would not be a bad idea to remove
these in favor of using `adbd_auth`.
