/*
Copyright (c) 2014, Michael Borisov
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of rsa_embedded nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/




#include <stdint.h>
#include <string.h>
#include "rsa.h"


/*
 * Single <= Single+Single Addition (mpi+mpi)
 * *dest += *src
 * Returns carry flag
 */
static uint8_t mpi_add(uint16_t* dest, const uint16_t* src)
{
	uint32_t c = 0;
    uint8_t count = MPI_NUMBER_SIZE;
	do
	{
		c += *dest + *src;
		*dest = c&0xFFFF;
		src++;
		dest++;
		c>>=16;
	} while(--count);
    return (uint8_t)(c&1);
}

/*
 * Single <= Single-Single Subtraction (mpi-mpi)
 * *dest -= *src
 */
static void mpi_sub(uint16_t* dest, const uint16_t* src)
{
	uint32_t c = 1;
    uint8_t count = MPI_NUMBER_SIZE;
	do
	{
		c += 65535 + *dest - *src;
		*dest = c&0xFFFF;
		src++;
		dest++;
		c>>=16;
		c&=1;
	} while(--count);
}

/*
 * Unsigned compare (mpi-mpi), always uses the default single precision length
 * returns sign of *a - *b
 */
static int8_t mpi_cmp(const uint16_t* a, const uint16_t* b)
{
	uint8_t count=MPI_NUMBER_SIZE;
	int32_t dif;
	a+=MPI_NUMBER_SIZE-1;
	b+=MPI_NUMBER_SIZE-1;
	do
	{
		dif = *a-- - *b--;
		if(dif)
			return dif>0 ? 1 : -1;
	} while(--count);
	return 0;
}



/*
 * Single+1 - Single*short -> Single+1 unsigned multiplication/subtraction (mpi-mpi*ushort)
 * *c -= *a * b
 * returns /nborrow flag from subtraction
 */
static uint8_t mpi_mulsubuuk(uint16_t* c, const uint16_t* a, uint16_t b)
{
	uint32_t nborrow = 1;
    uint32_t mcarry = 0;
    uint8_t count=MPI_NUMBER_SIZE;
    do
    {
        mcarry += *a++ * b;
        nborrow += 65535 + *c - (mcarry&0xFFFF);
        *c++ = nborrow&0xFFFF;
        mcarry>>=16;
        nborrow>>=16;
        nborrow&=1;
    } while(--count);
    // Subtract from the N+1th digit
    nborrow += 65535 + *c - (mcarry&0xFFFF);
    *c = nborrow&0xFFFF;
    nborrow >>= 16;
    return (uint8_t)(nborrow&1);
}

/*
 * Single*Single-> Double unsigned multiplication (mpi*mpi)
 * *c -= *a * b
 */

void mpi_muluu(uint16_t* c, const uint16_t* a, const uint16_t* b)
{
    uint64_t mcarry = 0;
    uint8_t count = MPI_NUMBER_SIZE;
    uint8_t niter2 = 1; // Inner loop starts with 1 iteration
    uint8_t count2;
    const uint16_t* aa;
    const uint16_t* bb;
    // First outer loop - the LSWs
    do
    {
        count2 = niter2++;
        aa = a;
        bb = b++;
        do
        {
            mcarry += (unsigned)*aa++ * (unsigned)*bb--;
        } while(--count2);
        *c++ = (uint16_t)(mcarry&0xFFFF);
        mcarry>>=16;
    } while(--count);
    // Second outer loop - the MSWs
    count = MPI_NUMBER_SIZE-1;
    b--;
    niter2--;
    do
    {
        count2 = --niter2;
        aa = ++a;
        bb = b;
        do
        {
            mcarry += (unsigned)*aa++ * (unsigned)*bb--;
        } while(--count2);
        *c++ = (uint16_t)(mcarry&0xFFFF);
        mcarry>>=16;
    } while(--count);
    // Store the last carry digit
    *c = (uint16_t)(mcarry&0xFFFF);
}

/*
 * Double/Single->Single modulo division (mpi/mpi)
 * *a %= *b
 * Divisor's MSW must be >= 0x8000
 */
void mpi_moduu(uint16_t* a, const uint16_t* b)
{
    uint16_t* pr = a + MPI_NUMBER_SIZE;// Initial partial remainder is the dividend's upper half
    uint8_t count = MPI_NUMBER_SIZE+1;
    uint8_t flag = 0; // If PR has N+1 significant words, this flag = 1
    uint32_t pq; // Partial quotient
    uint32_t carry;

    // Loop through the lower half of the dividend, bringing digits down to the PR
    do
    {
        if(flag)
        {
            // Number of words in PR is 1 more than in the divisor. Apply pq guessing
            pq = (((uint32_t)pr[MPI_NUMBER_SIZE])<<16) | (uint32_t)pr[MPI_NUMBER_SIZE-1];
            pq /= (uint32_t)b[MPI_NUMBER_SIZE-1];
            if(pq>0xFFFF)
                pq = 0xFFFF; // Divide overflow, use pq=0xFFFF
            // Multiply the divisor by pq, subtract the result from PR
            if(!mpi_mulsubuuk(pr,b,pq))
            {
                // trial pq was too high (max 1 or 2 too high), add the divisor back and check
                carry = mpi_add(pr,b);
                carry += pr[MPI_NUMBER_SIZE];
                pr[MPI_NUMBER_SIZE] = carry&0xFFFF;
                if(!(carry&0x10000))
                {
                    // Trial pq was still too high, add the divisor back again
                    carry = mpi_add(pr,b);
                    carry += pr[MPI_NUMBER_SIZE];
                    pr[MPI_NUMBER_SIZE] = carry&0xFFFF;
                }
            }
            // pr[MPI_NUMBER_SIZE] must be 0 here, but another digit will come at the next loop iteration
            flag = pr[MPI_NUMBER_SIZE-1]!=0;
        }
        else
        {
            // PR has N significant words or less
            if(pr[MPI_NUMBER_SIZE-1]) // MSW of PR
            {
                // PR has N significant words, same as dividend
                if(mpi_cmp(pr,b)>=0)
                {
                    // PR is >= divisor, pq=1, update PR
                    mpi_sub(pr,b);
                    // If MSW of PR after subtraction is >0, there will be N+1 words
                    flag = pr[MPI_NUMBER_SIZE-1]!=0;
                }
                else
                {
                    // PR is < divisor, pq=0, there will be N+1 words
                    flag = 1;
                }
            }
            else
            {
                // PR has less than N significant words. pq=0, leave the flag at 0
            }
        }
        pr--; // Bring in another digit from the dividend into PR
    } while(--count);
}

/*
 * Single modulo exponentiation M^65537 mod N (mpi^65537 mod mpi)
 * *m = (*m^65537) % n
 * Modulus's MSW must be >= 0x8000
 */

void mpi_powm65537(uint16_t* m, const uint16_t* n)
{
    uint8_t count = 16;    // First, square the number M modulo N 16 times
    uint16_t bufprod[MPI_NUMBER_SIZE*2]; // Squaring buffer, double size
    uint16_t bufmod[MPI_NUMBER_SIZE]; // Modulo buffer, single size
    memcpy(bufmod,m,MPI_NUMBER_SIZE*2);
    do
    {
        mpi_muluu(bufprod,bufmod,bufmod);
        mpi_moduu(bufprod,n);
        memcpy(bufmod,bufprod,MPI_NUMBER_SIZE*2);
    } while(--count);
    // Final multiplication with the original M
    mpi_muluu(bufprod,bufmod,m);
    mpi_moduu(bufprod,n);
    memcpy(m,bufprod,MPI_NUMBER_SIZE*2);
}
