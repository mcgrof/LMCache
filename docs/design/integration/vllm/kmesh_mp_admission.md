# KMesh multiprocess admission

Status: external MP construction refusal and its public handoff validator are
implemented and tested on CPU for scheduler and worker roles. The engine-side
inspection/arming boundary is separate. Ordinary multiprocess prefix caching
works independently of the experimental KMesh position-recovery profiles;
the MP connector does not provide a qualified KMesh metadata/span consumer.

The engine can inspect the named MP child only if it also demands a positive
handoff contract before construction completes and before KMesh arms. Base
connector lifecycle methods are ordinary no-ops; inheriting them cannot prove
that request lookup evidence or worker plans are consumed.

The external MP connector must reject any explicit KMesh target profile before
creating adapters or contacting a server. Its public handoff validator must
also refuse when asked for proof, including on an ordinary instance. Profile
absence does not turn an unsupported handoff into a supported one. Existing
in-process KMesh and unprofiled MP behavior remain separate accepted paths.

CPU tests must exercise the actual constructor for both scheduler and worker
roles and prove that no adapter/network construction occurs on refusal. The
engine's exact-class checks and handoff probe need their own tests. A future
positive validator requires actual server/domain/context evidence, explicit
span execution, and completion/exporter lifetime qualification. Standalone
native kernels or CPU ownership controls do not provide that qualification.
