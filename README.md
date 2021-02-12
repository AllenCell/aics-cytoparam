# Cytoplasm Parameterization

[![Build Status](https://github.com/AllenCell/aics-cytoparam/workflows/Build%20Main/badge.svg)](https://github.com/AllenCell/aics-cytoparam/actions)
[![Documentation](https://github.com/AllenCell/aics-cytoparam/workflows/Documentation/badge.svg)](https://AllenCell.github.io/aics-cytoparam/)
[![Code Coverage](https://codecov.io/gh/AllenCell/aics-cytoparam/branch/main/graph/badge.svg)](https://codecov.io/gh/AllenCell/aics-cytoparam)

Spherical harmonics expansion coefficients-based parameterization of the cytoplasm and nucleoplasm for 3D cells
---
## Installation

**Stable Release:** `pip install aicscytoparam`<br>
**Development Head:** `pip install git+https://github.com/AllenCell/aics-cytoparam.git`

## How to use

Here you have an example of how to use `aicscytoparam` to cretae a parameterization of a 3D cell. In this case the 3D cells will be represented by a cell segementation, nuclear segmentation a FP image representing the fluorescent signal of a tagged protein.

```python
# First create a cuboid cell with a not centered cuboid nucleus
# and get their SHE coefficients
w = 100
mem = np.zeros((w, w, w), dtype = np.uint8)
mem[20:80, 20:80, 20:80] = 1
nuc = np.zeros((w, w, w), dtype = np.uint8)
nuc[40:60, 40:60, 30:50] = 1

# Create a FP signal located at the top half of the cell and outside the
# nucleus.
gfp = np.random.rand(w**3).reshape(w,w,w)
gfp[mem==0] = 0
gfp[:,w//2:] = 0
gfp[nuc>0] = 0

# Vizualize a center cross section of our cell
plt.imshow((mem+nuc)[w//2], cmap='gray')
plt.imshow(gfp[w//2], cmap='gray', alpha=0.25)
plt.axis('off')
```

For full package documentation please visit [AllenCell.github.io/aics-cytoparam](https://AllenCell.github.io/aics-cytoparam).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## Questions?

If you have any questions, feel free to leave a comment in our Allen Cell forum: [https://forum.allencell.org/](https://forum.allencell.org/). 


***Free software: Allen Institute Software License***