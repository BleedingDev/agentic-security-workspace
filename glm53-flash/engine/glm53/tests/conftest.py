import os

# 8 virtual CPU devices so the TP engine can be tested without a TPU (must be set before jax is imported).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
