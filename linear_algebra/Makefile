main:
	/path-to-duo-sdk/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-gcc -o benchmark benchmark.c linear_algebra.c -mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d -Wall -Wextra -DRVV
clean:
	rm benchmark
