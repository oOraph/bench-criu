/* Probe the checkpoint API against a pid with and without a prior cuInit() in the caller. */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv)
{
	int pid = atoi(argv[1]);
	void *h = dlopen("libcuda.so.1", RTLD_NOW);
	CUresult (*init)(unsigned) = dlsym(h, "cuInit");
	CUresult (*gs)(int, CUprocessState *) = dlsym(h, "cuCheckpointProcessGetState");
	CUresult (*grt)(int, int *) = dlsym(h, "cuCheckpointProcessGetRestoreThreadId");
	CUresult (*ver)(int *) = dlsym(h, "cuDriverGetVersion");
	CUresult (*lock)(int, CUcheckpointLockArgs *) = dlsym(h, "cuCheckpointProcessLock");
	CUresult (*unlock)(int, CUcheckpointUnlockArgs *) = dlsym(h, "cuCheckpointProcessUnlock");
	CUprocessState st; int tid, v;
	printf("cuDriverGetVersion -> %d (api %d)\n", ver(&v), v);
	printf("before cuInit: GetState(%d) -> %d state=%d\n", pid, gs(pid, &st), st);
	printf("before cuInit: GetRestoreThreadId(%d) -> %d tid=%d\n", pid, grt(pid, &tid), tid);
	if (argc > 2 && argv[2][0] == 'l') {
		printf("before cuInit: Lock -> %d\n", lock(pid, NULL));
		printf("before cuInit: GetState -> %d state=%d\n", gs(pid, &st), st);
		printf("before cuInit: Unlock -> %d\n", unlock(pid, NULL));
	}
	printf("cuInit(0) -> %d\n", init(0));
	printf("after cuInit: GetState(%d) -> %d state=%d\n", pid, gs(pid, &st), st);
	printf("after cuInit: GetRestoreThreadId(%d) -> %d tid=%d\n", pid, grt(pid, &tid), tid);
	return 0;
}
