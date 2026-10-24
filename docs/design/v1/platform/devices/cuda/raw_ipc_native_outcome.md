# Quarantine ambiguous native IPC outcomes

Status: native-outcome quarantine and process-local reference serialization are
implemented; hardware qualification is pending. Earlier checked-release controls
inject failure before native effects and do not establish recovery after an
actual driver error or interruption at an unknown native side-effect boundary.

The [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html)
permits device-address reuse and gives a new exported allocation a distinct IPC
handle even when its address repeats. As a conservative inference, retrying an
ambiguous close by its old pointer could close an unrelated later import. An
error that can also report previous asynchronous work does not establish that
the attempted close had no effect. Never use pointer presence or absence to
manufacture a successful close receipt.

Separate a failure known to precede the native call from an unknown native
outcome. A device-context preflight refusal leaves a known live mapping and can
be retried. Once native close may have started, retain a quarantine marker
before submitting it. Clear that marker only after successful close and complete
reference bookkeeping. A native error or interruption before this transition
completes requires importer-process teardown; do not resubmit close by pointer.
Keep new raw imports and positive release inspection refused for that handle.

Apply the same rule to import and alias-reference bookkeeping. Mark its outcome
unresolved before opening or modifying a mapping. If interruption prevents
recording the pointer and owner references coherently, retain uncertainty even
when no completed mapping record exists. Checked close and raw release inspection
must not treat an empty record as proof that no import occurred.

Wrapper reference counts and uncertainty belong to the importing process.
Serialization must reset them on receipt, including for older metadata without
the new fields. Sending an imported wrapper does not transfer its sender's
physical obligations to another process.

Qualify native error quarantine with a controlled refusal before driver effects:
the test knows the old mapping remains live, while the API conservatively refuses
another close attempt. Keep the exporter alive until the importing child exits.
Separately qualify retry of a preflight failure that cannot have submitted native
close. Retain success and alias-retirement controls. Actual device faults remain
unqualified; these tests establish conservative refusal and process lifetime,
not automatic recovery from an unknown driver outcome.
