#include <windows.h>
#include "rc4.h"

unsigned char key[] = { 'A', 'B', 'C', 'D' };
unsigned char message[22] = {
    0x70, 0x01, 0x96, 0xDB, 0x60, 0xF4, 0x4C, 0x2F, 
    0xAF, 0x49, 0x46, 0xA1, 0xC9, 0xD6, 0xAA, 0x27,
    0x72, 0x2E, 0xB4, 0x5C, 0xE4, 0x67 
};

__declspec(dllexport) void __stdcall Demo()
{
    RC4_STATE rc4;

    rc4_init(&rc4, (unsigned char*)key, sizeof(key));
    rc4_crypt(&rc4, message, sizeof(message));
    MessageBoxA(0, message, message, MB_OK);
}

// DllMain not needed because CorDllMain will be called as entry point