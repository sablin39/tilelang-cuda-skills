"""Copy into the task root, then pass to measure_calibration.py module --factory.

This small example exercises fit/holdout and fusion-family comparisons. The
copy-rate estimate is a hypothesis; validation can reject it. It is not a
hardware lower bound or a recommended model for all pointwise kernels.
"""


def calibration_cases(device_profile):
    import torch

    anchors = [r for r in device_profile["measurements"]
               if r["quantity"] == "copy_effective_bandwidth" and r["status"] == "measured"]
    if not anchors:
        raise ValueError("example needs a measured copy anchor")
    anchor = max(anchors, key=lambda r: r["scope"]["working_set_bytes"])
    bandwidth = anchor["summary"]["median"]
    cases = []
    for size, partition in [(2 ** 20, "fit"), (2 ** 21, "holdout"), (2 ** 22, "holdout")]:
        for implementation in ("fused", "separate"):
            def setup(size=size, implementation=implementation):
                a, b, c = (torch.rand(size, device="cuda", dtype=torch.float32) for _ in range(3))
                output, temporary = torch.empty_like(a), torch.empty_like(a)

                def run():
                    if implementation == "fused":
                        torch.addcmul(c, a, b, out=output)
                    else:
                        torch.mul(a, b, out=temporary)
                        torch.add(temporary, c, out=output)

                def check():
                    run()
                    expected = a.double() * b.double() + c.double()
                    torch.testing.assert_close(output.double(), expected, rtol=1e-6, atol=1e-7)
                    return True

                return {"run": run, "check": check}

            # Fused: 3 reads + 1 write. Separate: 4 reads + 2 writes.
            traffic = size * 4 * (4 if implementation == "fused" else 6)
            cases.append({"id": f"{implementation}-{size}", "group": implementation, "partition": partition,
                          "workload": {"operation": "a*b+c", "elements": size, "dtype": "float32", "contiguous": True},
                          "configuration": {"implementation": implementation, "preallocated_output": True},
                          "predicted_ns": traffic / bandwidth * 1e9, "setup": setup})
    return {"module": "pointwise-triad", "regime": {
        "implementation_families": ["torch.addcmul", "torch.mul+torch.add"],
        "elements": [2 ** 20, 2 ** 22], "dtype": "float32", "layout": "contiguous",
        "cache_policy": "repeated inputs; residency unverified", "allocation": "preallocated output and temporary",
        "initial_model": "logical bytes divided by a copy anchor; no assumed HBM bound",
        "anchor_id": anchor["id"]}, "cases": cases}
