# Mcanon acquisition and deterministic parsing

The V15 canonical support mask is the DECA asset
`data/uv_face_eye_mask.png`. This package does not redistribute it. Obtain DECA
from its official project under the provider's terms:

```text
https://github.com/YadiraF/DECA
```

The author-held checkout is bound to commit
`a11554ae2a2b0f3998cf1fa94dd4db03babb34a2`. The required source file SHA-256
is `a5069d8ffaf020008ae92d5062c4e98600f723aac4eb869731f190e4630467b5`.

Parsing is fixed:

1. convert to single-channel `L` mode;
2. resize to 64×64 by nearest-neighbour resampling;
3. convert to `uint8` and threshold with `value > 127`;
4. store a C-contiguous boolean array.

The result contains 1,515 true texels and has canonical array SHA-256
`7a3f9bd59eebcaf3471892c4569f3e4aa5a0510d9d5c003a1ddce3977fd2fd69`.

After official acquisition, verify the file without copying it into this
package:

```text
python3 -B tools/verify_mcanon.py /path/to/uv_face_eye_mask.png
```

A passing hash check establishes byte and parse identity only; it does not
grant redistribution rights.
