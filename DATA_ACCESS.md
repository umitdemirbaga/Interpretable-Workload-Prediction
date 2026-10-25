# Data availability

The monitoring records originate from Hadoop workloads executed on Linux-based AWS infrastructure.

The repository contains the processed infrastructure-monitoring table `data/processed/All_with_group.csv` (24,000 task profiles; 800 consecutive job groups). Its fields and label construction are described in `data/processed/README.md`.

The four healthcare workloads of 250K, 500K, 750K and 1M records, the NEWS reference/generated source records, and raw SmartMonit acquisition logs are not distributed. The processed table does not include patient-level NEWS data, monitoring timestamps, original Hadoop job identifiers or workload-scale assignments.

Reproduction of the original Hadoop data acquisition requires the corresponding input datasets and execution environment. The NEWS fidelity and leave-one-scale-out experiments also require the additional inputs described in the corresponding Python scripts.
