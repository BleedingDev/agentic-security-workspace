#ifndef _RC4_H_
#define _RC4_H_
typedef struct RC4_STATE_
{
    int x, y, m[256];
}RC4_STATE, *PRC4_STATE;

void rc4_init(PRC4_STATE s, unsigned char* key, int length);
void rc4_crypt(PRC4_STATE s, unsigned char* data, int length);
#endif