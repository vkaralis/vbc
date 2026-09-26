from pathlib import Path

import numpy as np

from vbc import decompose

data = np.genfromtxt(
    Path(__file__).with_name("endpoints.csv"),
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)
for endpoint in ("B", "C", "D"):
    result = decompose(data["A"], data[endpoint], preprocessing="none")
    print(f"A vs {endpoint}: {result.angle_degrees:.3f} degrees")
    print(f"  perpendicular component: {result.perpendicular}")
