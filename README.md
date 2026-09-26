# VBC — Vector-Based Comparison

`vbc` is a small Python implementation of Vector-Based Comparison (VBC), a method for
decomposing clinical endpoints relative to a selected primary endpoint. It accompanies:

> Karalis, V. D. (2024). *A Vector Theory of Assessing Clinical Trials: An Application to
> Bioequivalence*. Journal of Cardiovascular Development and Disease, 11(7), 185.
> [https://doi.org/10.3390/jcdd11070185](https://doi.org/10.3390/jcdd11070185)

This package implements a generalized VBC framework inspired by Karalis (2024), using
explicit orthogonal vector projection for endpoint decomposition.

## Method in brief

Given a primary endpoint **A** and another endpoint **B**, VBC computes

```text
cos(theta) = dot(A, B) / (norm(A) * norm(B))
B_parallel = (dot(A, B) / dot(A, A)) * A
B_perpendicular = B - B_parallel
```

This construction guarantees `dot(A, B_perpendicular) = 0` apart from floating-point rounding.
It also satisfies `norm(B_perpendicular) = norm(B) * sin(theta)`: the sine expression gives the
magnitude of the perpendicular component, while the residual formula constructs the component
vector itself. For several endpoints, each is decomposed separately relative to the selected
primary endpoint.

This release implements the orthogonal projection interpretation agreed with the author.
It differs from the published elementwise expression `B * sin(theta)` and from the original
MATLAB routine. It does not claim to reproduce the paper's simulation results. The formula
with the sine applies to the residual's norm, not to each observation.
Residuals of different secondary endpoints are all orthogonal to A, but are not necessarily
orthogonal to one another. The primary endpoint is retained unchanged in the selected
preprocessing domain; the API returns only the secondary endpoints' components.

## Install

Python 3.9 or later is required.

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## Python example

```python
from vbc import transform_endpoints

endpoints = {
    "A": [10.0, 12.0, 9.0, 14.0],  # primary endpoint
    "B": [5.0, 7.0, 6.0, 8.0],
    "C": [20.0, 17.0, 22.0, 16.0],
}

results = transform_endpoints(endpoints, primary="A")

print(results["B"].angle_degrees)
print(results["B"].perpendicular)
print(results["C"].angle_degrees)
print(results["C"].perpendicular)
```

`A` denotes the selected primary endpoint. `B`, `C`, ..., `N` denote all remaining
endpoints. The number and scientific meaning of the endpoints are not restricted by the
software. Each non-primary endpoint is decomposed independently relative to `A`.

### Mapping for the bioequivalence example

In the pharmacokinetic bioequivalence application presented in the accompanying article, the
generic endpoint notation maps as follows:

- `A` = AUC, the selected primary endpoint;
- `B` = Cmax;
- `C` = AS (average slope).

Consequently, both Cmax (`B`) and AS (`C`) are decomposed separately relative to AUC (`A`).
These endpoint labels refer to the roles of Cmaxz and ASy in the article; the current
orthogonal residuals are not numerically identical to its scalar transformation.
This mapping is an application example and is not a restriction of the software: in another
clinical study, `A`, `B`, `C`, ..., `N` may represent entirely different endpoints.

By default, VBC uses the endpoint values exactly as supplied (`preprocessing="none"`). This
preserves the original angle. Cosine similarity is already invariant to multiplication by a
positive constant, so changing measurement units does not change the angle:

```text
cos(A, cB) = cos(A, B), for c > 0
```

Available preprocessing options are:

- `none` (default): preserve the original values and angle;
- `zscore`: subtract the mean and divide by the sample standard deviation (`ddof=1`).
  This centers each endpoint and gives it unit sample variance. Cosine similarity then
  equals sample Pearson correlation, making this option appropriate when the intended
  quantity is linear endpoint association. Constant endpoints are rejected.
- `l2`: divide each endpoint by its Euclidean norm, making it unit length while preserving
  its angle exactly;
- `minmax`: map each endpoint to `[0, 1]`. Because it subtracts the minimum and therefore
  moves the origin, this option can change the angle and should be used only deliberately.

An orthogonal residual can contain negative values even when all input observations are positive.
This is expected and is required for exact orthogonality.

The default `none` is not inherently preferable to standardization: it preserves geometry
relative to the original origin, whereas `zscore` measures centered linear association and
follows the standardization step described in the paper. The orthogonal decomposition
remains the generalized implementation described above, rather than an exact reproduction
of the paper. With `zscore`, returned components are in standardized units and reconstruct
the standardized endpoint, not the original values.

```python
results = transform_endpoints(endpoints, primary="A", preprocessing="zscore")
```

From the command line:

```bash
vbc examples/endpoints.csv --id-column participant_id --primary A --preprocess zscore
```

For a domain that mandates another transformation, apply it before calling VBC. For example,
the article's bioequivalence application uses logarithmic pharmacokinetic endpoints.
Given strictly positive arrays `auc`, `cmax`, and `average_slope`, transform them first:

```python
import numpy as np
from vbc import transform_endpoints

log_endpoints = {
    "A": np.log(auc),
    "B": np.log(cmax),
    "C": np.log(average_slope),
}
results = transform_endpoints(log_endpoints, primary="A", preprocessing="none")
```

The logarithm is therefore a requirement of that application and is not part of the general
VBC transformation. Use `preprocessing="none"` (or the angle-preserving `"l2"`) after the
logarithm when the original log-domain angle must be preserved.

### Matching observations

Position `i` must refer to the same participant in every endpoint vector. For example, `A[0]`,
`B[0]`, and `C[0]` must all belong to the same participant. Reordering one endpoint independently
invalidates the dot products, angles, and decomposition. Missing observations must be handled
consistently across all endpoints so that complete rows remain aligned.

The CSV interface can validate a participant-ID column. IDs must be present and unique; the ID
column is used only for validation and is excluded from the VBC calculation.
IDs are retained in output order. This check cannot detect values that were already assigned
to the wrong participant before importing the CSV.

### Orthogonality and statistical independence

The VBC perpendicular component is geometrically orthogonal to the primary endpoint. This
means their dot product is zero. Orthogonality alone should not be described as unconditional
statistical independence: that stronger conclusion requires assumptions about the joint
distribution. When the analyzed vectors are centered, orthogonality implies zero sample
covariance. Zero population covariance implies independence for jointly normal variables;
zero sample covariance by itself does not establish population independence. With the default
uncentered inputs, cosine similarity is not Pearson correlation.

## Command line

Input is a CSV file with one numeric endpoint per column and one participant per row:

```bash
vbc examples/endpoints.csv --id-column participant_id --primary A --output results.json
```

The JSON output contains the cosine similarity, angle, parallel component, and perpendicular
component for every non-primary endpoint.
The example contains six synthetic participants and four generic endpoints, not clinical data.

## Reproducibility and scope

The package intentionally implements only the reusable VBC numerical core: preprocessing,
cosine similarity, angle estimation, and decomposition of each non-primary endpoint relative
to the selected primary endpoint. It is endpoint-agnostic and is not restricted to
pharmacokinetics or bioequivalence. Domain-specific calculations, statistical tests, simulation
models, GUIs, and publication plotting routines are outside its scope.

VBC is research software. Validate the implementation and analysis plan before using results for
clinical or regulatory decisions.

## Repository layout

```text
src/vbc/           Python package
tests/             automated tests
examples/          minimal input and runnable example
```

## Citation

Please cite the article above. Citation metadata is also provided in `CITATION.cff`.

## License

The software in this repository is released under the [MIT License](LICENSE). The linked article
has its own CC BY 4.0 license and is not included in the public repository.
