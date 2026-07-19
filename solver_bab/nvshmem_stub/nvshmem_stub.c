/* Stub libnvshmem_host.so.3 — jamais appelé en pratique : notre usage est
 * inférence GNN mono-GPU (pas de communication multi-GPU distribuée).
 * libtorch_cuda.so a un NEEDED dur sur libtorch_nvshmem.so (bundlé dans le
 * zip libtorch 2.13.0), qui lui-même a un NEEDED dur sur libnvshmem_host.so.3
 * (absent du cluster, pas de module nvshmem). Ce stub fournit juste les
 * symboles versionnés @NVSHMEM nécessaires à la résolution du chargeur
 * dynamique au démarrage — n'importe quel appel réel plantera, mais aucun
 * chemin de notre code n'appelle ces fonctions. */
int nvshmem_free(void) { return 0; }
int nvshmem_malloc(void) { return 0; }
int nvshmem_ptr(void) { return 0; }
int nvshmem_team_n_pes(void) { return 0; }
int nvshmem_team_split_strided(void) { return 0; }
int nvshmemid_hostlib_finalize(void) { return 0; }
int nvshmemid_hostlib_init_attr(void) { return 0; }
int nvshmemid_init_status(void) { return 0; }
int nvshmem_info_get_version(void) { return 0; }
int nvshmemx_alltoallmem_on_stream(void) { return 0; }
int nvshmemx_broadcastmem_on_stream(void) { return 0; }
int nvshmemx_cumodule_init(void) { return 0; }
int nvshmemx_getmem_on_stream(void) { return 0; }
int nvshmemx_get_uniqueid(void) { return 0; }
int nvshmemx_mc_ptr(void) { return 0; }
int nvshmemx_putmem_on_stream(void) { return 0; }
int nvshmemx_putmem_signal_on_stream(void) { return 0; }
int nvshmemx_set_attr_uniqueid_args(void) { return 0; }
int nvshmemx_signal_wait_until_on_stream(void) { return 0; }
