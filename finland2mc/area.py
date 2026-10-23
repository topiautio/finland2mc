"""The square of land being turned into a world, and the grid it is sampled on."""
from dataclasses import dataclass

from rasterio.transform import from_origin

MARGIN = 16  # extra metres sampled around the playable area


@dataclass(frozen=True)
class Area:
    """A square centred on ETRS-TM35FIN point (ce, cn).

    Minecraft x = x0 + col, z = x0 + row (1 block = 1 m). E = ce + x + 0.5, N = cn - z - 0.5.
    """
    ce: int
    cn: int
    half: int  # half the side length in metres

    @property
    def x0(self):
        return -(self.half + MARGIN)

    @property
    def size(self):
        return 2 * (self.half + MARGIN)

    @property
    def west(self):
        return self.ce + self.x0

    @property
    def north(self):
        return self.cn - self.x0

    @property
    def bounds(self):
        """(minE, minN, maxE, maxN) of the sampled grid."""
        return (self.west, self.north - self.size, self.west + self.size, self.north)

    @property
    def transform(self):
        return from_origin(self.west, self.north, 1, 1)

    def to_dict(self):
        return dict(ce=self.ce, cn=self.cn, half=self.half)
