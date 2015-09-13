#include <stdio.h>
#include <stdlib.h>
int main( int argc, char **argv){
	printf("Size of int = %d\n", sizeof(int));
	printf("Size of long = %d\n", sizeof(long));
	printf("Size of float = %d\n", sizeof(float));
	printf("Size of double = %d\n", sizeof(double));
	printf("Size of void * = %d\n", sizeof(void *));
	printf("Size of int * = %d\n", sizeof(int *));
	printf("Size of long * = %d\n", sizeof(long *));
	printf("Size of float * = %d\n", sizeof(float *));
}
