# Workload-scale metadata

The processed monitoring table contains positional `group` identifiers but not the original Hadoop job IDs or workload-scale labels. The leave-one-scale-out evaluation uses `job_scale_map.csv` with columns:

`group,scale,original_job_id`

Each of the 800 groups requires an authentic mapping to a Hadoop job and one of the scales `250K`, `500K`, `750K` or `1M`. Group order is not a substitute for this metadata.
