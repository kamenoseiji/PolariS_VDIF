//	cuda_fft_xspec.c : FFT using CuFFT
//
//	Author : Seiji Kameno
//	Created: 2012/12/6
//
#include <cuda.h>
#include <cufft.h>
#include <string.h>
#include <math.h>
#include </usr/local/cuda-6.0/samples/common/inc/timer.h>
#include "cuda_polaris.inc"
#define SCALEFACT 1.0/(NFFT* NsegSec)

extern int segment_offset(struct SHM_PARAM *,	int *);
// extern int fileRecOpen(struct SHM_PARAM *, int, int, char *, char *, FILE **);
extern int gaussBit(int, unsigned int *, double *, double *);
extern int bitDist16st2bit(int, unsigned char *, unsigned int *);

main(
	int		argc,			// Number of Arguments
	char	**argv )		// Pointer to Arguments
{
	int		shrd_param_id;				// Shared Memory ID
	int		index;						// General Index
	int		cycle_index;				// Index for cycle of buffer (4 cycles per sec)
	int		page_index;					// Index for page in buffer (2 pages per cycle)
	int		seg_index;					// Index for Segment
	int		offset[16384];				// Segment offset position
	int		IF_index;
	struct	SHM_PARAM	*param_ptr;		// Pointer to the Shared Param
	struct	sembuf		sops;			// Semaphore for data access
	unsigned char	*vdifhead_ptr;		// Pointer to the VDIF header
	unsigned char	*vdifdata_ptr;		// Pointer to shared VDIF data
	float	*xspec_ptr;					// Pointer to 1-sec-integrated Power Spectrum
	FILE	*file_ptr[6];				// File Pointer to write
	FILE	*power_ptr[4];				// Power File Pointer to write
	char	fname_pre[16];
	unsigned int		bitDist[64];	// 16 IF x 4 level
	double	param[2], param_err[2];		// Gaussian parameters derived from bit distribution

	dim3			Dg, Db(512,1, 1);	// Grid and Block size
	unsigned char	*cuvdifdata_ptr;	// Pointer to VDIF data in GPU
	cufftHandle		cufft_plan;			// 1-D FFT Plan, to be used in cufft
	cufftReal		*cuRealData;		// Time-beased data before FFT, every IF, every segment
	cufftComplex	*cuSpecData;		// FFTed spectrum, every IF, every segment
	float			*cuPowerSpec;		// (autocorrelation) Power Spectrum
	float2			*cuXSpec;

//------------------------------------------ Access to the SHARED MEMORY
	shrd_param_id = shmget( SHM_PARAM_KEY, sizeof(struct SHM_PARAM), 0444);
	param_ptr  = (struct SHM_PARAM *)shmat(shrd_param_id, NULL, 0);
	vdifhead_ptr = (unsigned char *)shmat(param_ptr->shrd_vdifhead_id, NULL, SHM_RDONLY);
	vdifdata_ptr = (unsigned char *)shmat(param_ptr->shrd_vdifdata_id, NULL, SHM_RDONLY);
	xspec_ptr  = (float *)shmat(param_ptr->shrd_xspec_id, NULL, 0);
//------------------------------------------ Prepare for CuFFT
	cudaMalloc( (void **)&cuvdifdata_ptr, MAX_SAMPLE_BUF);
	cudaMalloc( (void **)&cuRealData, NST* NsegPart* NFFT * sizeof(cufftReal) );
	cudaMalloc( (void **)&cuSpecData, NST* NsegPart* NFFTC* sizeof(cufftComplex) );
	cudaMalloc( (void **)&cuPowerSpec, NST* NFFT2* sizeof(float));
	// cudaMalloc( (void **)&cuXSpec, 2* NFFT2* sizeof(float2));

	// if(cudaGetLastError() != cudaSuccess){
	// 	fprintf(stderr, "Cuda Error : Failed to allocate memory.\n"); return(-1); }

 	if(cufftPlan1d(&cufft_plan, NFFT, CUFFT_R2C, NST* NsegPart ) != CUFFT_SUCCESS){
 		fprintf(stderr, "Cuda Error : Failed to create plan.\n"); return(-1); }
//------------------------------------------ Parameters for S-part format
	// printf("NsegPart = %d\n", NsegPart);
 	segment_offset(param_ptr, offset);
	for(index=0; index< 2*NsegPart; index++){	printf("Offset[%d] = %d\n", index, offset[index]);}
//------------------------------------------ K5 Header and Data
 	param_ptr->current_rec = 0;
	setvbuf(stdout, (char *)NULL, _IONBF, 0);   // Disable stdout cache
	while(param_ptr->validity & ACTIVE){
		if( param_ptr->validity & (FINISH + ABSFIN) ){  break; }

		//-------- Initial setup for cycles
		if( param_ptr->part_index == 0){
			cudaMemset( cuPowerSpec, 0, NST* NFFT2* sizeof(float));		// Clear Power Spectrum to accumulate
		}
		cycle_index = param_ptr->part_index / 2;	// 4 cycles per 1 sec
		page_index  = param_ptr->part_index % 2;	// 2 pages per cycle

		//-------- Loop for half-sec period
		// memset(bitDist, 0, sizeof(bitDist));

		//-------- Wait for the first half in the S-part
		sops.sem_num = (ushort)1; sops.sem_op = (short)-1; sops.sem_flg = (short)0;
		semop( param_ptr->sem_data_id, &sops, 1);
		// printf("Ready to process Part=%d Cycle=%d Page=%d\n", param_ptr->part_index, cycle_index, page_index);
		StartTimer();

		//-------- SHM -> GPU memory transfer
		cudaMemcpy( &cuvdifdata_ptr[HALFBUF* page_index], &vdifdata_ptr[HALFBUF* page_index], HALFBUF, cudaMemcpyHostToDevice);

		//-------- Segment Format
		Dg.x=NFFT/512; Dg.y=1; Dg.z=1;
		for(index=0; index < NsegPart; index ++){
			seg_index = page_index* NsegPart + index;
			segform_16st_2bit<<<Dg, Db>>>( &cuvdifdata_ptr[offset[seg_index]], &cuRealData[index* NST* NFFT], NFFT);	// 
			// segform_8st_2bit<<<Dg, Db>>>( &cuvdifdata_ptr[offset[seg_index]], &cuRealData[index* NST* NFFT], NFFT);	// 
			// segform_4st_2bit<<<Dg, Db>>>( &cuvdifdata_ptr[offset[seg_index]], &cuRealData[index* NST* NFFT], NFFT);	// 
			// segform_2st_2bit<<<Dg, Db>>>( &cuvdifdata_ptr[offset[seg_index]], &cuRealData[index* NST* NFFT], NFFT);	// 
		}

		//-------- FFT Real -> Complex spectrum
		cudaThreadSynchronize();
		cufftExecR2C(cufft_plan, cuRealData, cuSpecData);		// FFT Time -> Freq
		cudaThreadSynchronize();

		//---- Auto Corr
		Dg.x= NFFTC/512; Dg.y=1; Dg.z=1;
		for(seg_index=0; seg_index<NsegPart; seg_index++){
			 for(index=0; index<NST; index++){
				accumPowerSpec<<<Dg, Db>>>( &cuSpecData[(seg_index* NST + index)* NFFTC], &cuPowerSpec[index* NFFT2],  NFFT2);
			}
		}

		//-------- Dump cross spectra to shared memory
		if( param_ptr->part_index == 7){
			// scalePowerSpec<<<Dg, Db>>>(cuPowerSpec, SCALEFACT, NST* NFFT2);
			cudaMemcpy(xspec_ptr, cuPowerSpec, NST* NFFT2* sizeof(float), cudaMemcpyDeviceToHost);
			sops.sem_num = (ushort)SEM_FX; sops.sem_op = (short)1; sops.sem_flg = (short)0; semop( param_ptr->sem_data_id, &sops, 1);
		}

		//-------- BitDist
		// bitDist16st2bit(HALFBUF, &vdifdata_ptr[HALFBUF* (param_ptr->part_index % 2)], bitDist); 
		// for(IF_index=0; IF_index<16; IF_index ++){
		// 	gaussBit(4, &bitDist[4* IF_index], param, param_err );
		// 	param_ptr->power[IF_index] = 1.0 / (param[0]* param[0]);
		// 	// printf("%6.2f ", 10.0* log10(param_ptr->power[IF_index]));
		// }
		// printf("\n");
			
		printf("%lf [msec]\n", GetTimer());
		param_ptr->current_rec ++;

	}	// End of part loop
/*
-------------------------------------------- RELEASE the SHM
*/
	// for(index=0; index<Nif+2; index++){ if( file_ptr[index] != NULL){	fclose(file_ptr[index]);} }
	// for(index=0; index<Nif; index++){ if( power_ptr[index] != NULL){	fclose(power_ptr[index]);} }
	// cufftDestroy(cufft_plan);
	cudaFree(cuvdifdata_ptr); cudaFree(cuRealData); cudaFree(cuSpecData); cudaFree(cuPowerSpec); // cudaFree(cuXSpec);

    return(0);
}

