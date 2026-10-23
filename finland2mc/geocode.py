"""Find a place in Finland by name (OpenStreetMap Nominatim) and convert to ETRS-TM35FIN."""
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

from pyproj import Transformer

from . import __version__

USER_AGENT = f"finland2mc/{__version__} (+https://github.com/topiautio/finland2mc)"
_TO_TM35 = Transformer.from_crs(4326, 3067, always_xy=True)
_TO_WGS = Transformer.from_crs(3067, 4326, always_xy=True)


@dataclass
class Place:
    name: str
    description: str
    lat: float
    lon: float

    @property
    def etrs(self):
        e, n = _TO_TM35.transform(self.lon, self.lat)
        return int(round(e)), int(round(n))


def search(query, limit=5):
    """Places in Finland matching the query, best match first."""
    params = urllib.parse.urlencode(dict(q=query, countrycodes="fi", format="jsonv2", limit=limit,
                                         **{"accept-language": "fi"}))
    req = urllib.request.Request(f"https://nominatim.openstreetmap.org/search?{params}",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        results = json.load(r)
    return [Place(r.get("name") or query, r["display_name"], float(r["lat"]), float(r["lon"])) for r in results]


def from_coordinates(text):
    """'62.24,25.75' (latitude, longitude) or 'E381000,N6905000' (ETRS-TM35FIN) to a Place."""
    a, b = [t.strip() for t in text.split(",")]
    if a.upper().startswith("E") and b.upper().startswith("N"):
        lon, lat = _TO_WGS.transform(float(a[1:]), float(b[1:]))
    else:
        lat, lon = float(a), float(b)
    return Place(f"{lat:.4f}, {lon:.4f}", "given coordinates", lat, lon)
