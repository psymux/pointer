from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import requests
from skyfield import named_stars
from skyfield.api import (
    Loader,
    Star,
    EarthSatellite,
    load_constellation_map,
    load_constellation_names,
    position_of_radec,
    wgs84,
)


CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
HORIZONS_LOOKUP_URL = "https://ssd.jpl.nasa.gov/api/horizons_lookup.api"
DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "pointer"
TLE_CACHE_TTL_SECONDS = 6 * 3600
LOOKUP_CACHE_TTL_SECONDS = 24 * 3600
SPACECRAFT_WINDOW_HOURS = 12
SPACECRAFT_STEP_MINUTES = 5

SOLAR_SYSTEM_BODIES = {
    "sun": ("sun", "Sun"),
    "moon": ("moon", "Moon"),
    "mercury": ("mercury", "Mercury"),
    "venus": ("venus", "Venus"),
    "mars": ("mars", "Mars"),
    "jupiter": ("jupiter barycenter", "Jupiter"),
    "saturn": ("saturn barycenter", "Saturn"),
    "uranus": ("uranus barycenter", "Uranus"),
    "neptune": ("neptune barycenter", "Neptune"),
    "pluto": ("pluto barycenter", "Pluto"),
}

DEEP_SKY_OBJECTS = [
    {
        "name": "Andromeda Galaxy",
        "aliases": ["M31", "NGC 224", "Andromeda"],
        "ra_hours": 0.712,
        "dec_degrees": 41.269,
    },
    {
        "name": "Triangulum Galaxy",
        "aliases": ["M33", "NGC 598", "Triangulum"],
        "ra_hours": 1.564,
        "dec_degrees": 30.66,
    },
    {
        "name": "Pleiades",
        "aliases": ["M45", "Seven Sisters"],
        "ra_hours": 3.79,
        "dec_degrees": 24.1167,
    },
    {
        "name": "Orion Nebula",
        "aliases": ["M42", "NGC 1976"],
        "ra_hours": 5.588,
        "dec_degrees": -5.391,
    },
    {
        "name": "Crab Nebula",
        "aliases": ["M1", "NGC 1952"],
        "ra_hours": 5.5756,
        "dec_degrees": 22.0145,
    },
    {
        "name": "Beehive Cluster",
        "aliases": ["M44", "Praesepe", "NGC 2632"],
        "ra_hours": 8.666,
        "dec_degrees": 19.666,
    },
    {
        "name": "Omega Centauri",
        "aliases": ["NGC 5139", "Caldwell 80"],
        "ra_hours": 13.446,
        "dec_degrees": -47.479,
    },
    {
        "name": "Hercules Globular Cluster",
        "aliases": ["M13", "NGC 6205"],
        "ra_hours": 16.695,
        "dec_degrees": 36.467,
    },
    {
        "name": "Lagoon Nebula",
        "aliases": ["M8", "NGC 6523"],
        "ra_hours": 18.061,
        "dec_degrees": -24.387,
    },
    {
        "name": "Trifid Nebula",
        "aliases": ["M20", "NGC 6514"],
        "ra_hours": 18.032,
        "dec_degrees": -22.971,
    },
    {
        "name": "Eagle Nebula",
        "aliases": ["M16", "NGC 6611"],
        "ra_hours": 18.313,
        "dec_degrees": -13.806,
    },
    {
        "name": "Omega Nebula",
        "aliases": ["M17", "Swan Nebula", "NGC 6618"],
        "ra_hours": 18.34,
        "dec_degrees": -16.171,
    },
    {
        "name": "Ring Nebula",
        "aliases": ["M57", "NGC 6720"],
        "ra_hours": 18.893,
        "dec_degrees": 33.03,
    },
    {
        "name": "Dumbbell Nebula",
        "aliases": ["M27", "NGC 6853"],
        "ra_hours": 19.993,
        "dec_degrees": 22.721,
    },
    {
        "name": "Carina Nebula",
        "aliases": ["NGC 3372", "Eta Carinae Nebula"],
        "ra_hours": 10.75,
        "dec_degrees": -59.52,
    },
    {
        "name": "47 Tucanae",
        "aliases": ["NGC 104", "47 Tuc"],
        "ra_hours": 0.403,
        "dec_degrees": -72.081,
    },
    {
        "name": "Whirlpool Galaxy",
        "aliases": ["M51", "NGC 5194"],
        "ra_hours": 13.497,
        "dec_degrees": 47.195,
    },
    {
        "name": "Pinwheel Galaxy",
        "aliases": ["M101", "NGC 5457"],
        "ra_hours": 14.054,
        "dec_degrees": 54.349,
    },
    {
        "name": "Bode's Galaxy",
        "aliases": ["M81", "NGC 3031"],
        "ra_hours": 9.926,
        "dec_degrees": 69.065,
    },
    {
        "name": "Cigar Galaxy",
        "aliases": ["M82", "NGC 3034"],
        "ra_hours": 9.928,
        "dec_degrees": 69.679,
    },
    {
        "name": "Sombrero Galaxy",
        "aliases": ["M104", "NGC 4594"],
        "ra_hours": 12.666,
        "dec_degrees": -11.623,
    },
    {
        "name": "Large Magellanic Cloud",
        "aliases": ["LMC"],
        "ra_hours": 5.391,
        "dec_degrees": -69.756,
    },
    {
        "name": "Small Magellanic Cloud",
        "aliases": ["SMC"],
        "ra_hours": 0.877,
        "dec_degrees": -72.829,
    },
]


def _normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _signed(value: float, *, is_ra: bool = False) -> float:
    return value % (360.0 if is_ra else value)


def _sexagesimal_ra_to_degrees(hours: str, minutes: str, seconds: str) -> float:
    return 15.0 * (float(hours) + float(minutes) / 60.0 + float(seconds) / 3600.0)


def _sexagesimal_dec_to_degrees(degrees: str, minutes: str, seconds: str) -> float:
    sign = -1.0 if degrees.strip().startswith("-") else 1.0
    whole = abs(float(degrees))
    return sign * (whole + float(minutes) / 60.0 + float(seconds) / 3600.0)


def _format_cache_key(value: str) -> str:
    return _normalize_name(value) or "blank"


def _to_iso_utc(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _wrap_degrees_360(angle: float) -> float:
    return angle % 360.0


def _looks_like_ra_hour_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return False
    if stripped.startswith(("+", "-")):
        return False
    return stripped.isdigit()


def _parse_tle_triplets(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return []
    triplets = []
    for index in range(0, len(lines) - 2, 3):
        name, line1, line2 = lines[index:index + 3]
        if line1.startswith("1 ") and line2.startswith("2 "):
            triplets.append({"name": name, "line1": line1, "line2": line2})
    return triplets


def _json_default(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


@dataclass
class TargetSpec:
    kind: str
    query: str
    source: str | None = None
    identifier: str | None = None


@dataclass
class TargetMatch:
    kind: str
    query: str
    display_name: str
    identifier: str | None = None
    source: str | None = None
    description: str | None = None

    def to_spec(self) -> TargetSpec:
        return TargetSpec(
            kind=self.kind,
            query=self.query,
            source=self.source,
            identifier=self.identifier,
        )


@dataclass
class TargetState:
    kind: str
    display_name: str
    az_deg: float
    alt_deg: float
    ra_deg: float
    dec_deg: float
    distance: float | None
    distance_unit: str | None
    is_visible: bool
    plot_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class TargetResolutionError(RuntimeError):
    pass


class AmbiguousTargetError(TargetResolutionError):
    def __init__(self, query: str, matches: list[TargetMatch]):
        super().__init__(f"Ambiguous target '{query}'.")
        self.matches = matches


@dataclass
class ObserverContext:
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    surface: Any
    topos: Any


class CacheStore:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, bucket: str, key: str) -> pathlib.Path:
        path = self.root / bucket
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{key}.json"

    def load_json(self, bucket: str, key: str, *, ttl_seconds: int | None = None) -> dict[str, Any] | None:
        path = self._path(bucket, key)
        if not path.exists():
            return None
        if ttl_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > ttl_seconds:
                return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def save_json(self, bucket: str, key: str, payload: dict[str, Any]) -> pathlib.Path:
        path = self._path(bucket, key)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
        return path


class ResolvedTarget:
    kind: str
    display_name: str

    def state_at(self, target_time: Any, observer_context: ObserverContext) -> TargetState:
        raise NotImplementedError


class EarthSatelliteTarget(ResolvedTarget):
    def __init__(self, *, display_name: str, satellite: EarthSatellite, source: str):
        self.kind = "satellite"
        self.display_name = display_name
        self.satellite = satellite
        self.source = source

    def state_at(self, target_time: Any, observer_context: ObserverContext) -> TargetState:
        geocentric = self.satellite.at(target_time)
        subpoint = wgs84.subpoint(geocentric)
        topocentric = (self.satellite - observer_context.surface).at(target_time)
        alt, az, distance = topocentric.altaz()
        ra, dec, _ = topocentric.radec()
        return TargetState(
            kind=self.kind,
            display_name=self.display_name,
            az_deg=float(az.degrees),
            alt_deg=float(alt.degrees),
            ra_deg=float(ra.degrees),
            dec_deg=float(dec.degrees),
            distance=float(distance.km),
            distance_unit="km",
            is_visible=float(alt.degrees) >= 0.0,
            plot_mode="earth",
            metadata={
                "subpoint_lat_deg": float(subpoint.latitude.degrees),
                "subpoint_lon_deg": float(subpoint.longitude.degrees),
                "subpoint_elevation_km": float(subpoint.elevation.km),
                "source": self.source,
            },
        )


class SkyObjectTarget(ResolvedTarget):
    def __init__(self, *, kind: str, display_name: str, body: Any, metadata: dict[str, Any] | None = None):
        self.kind = kind
        self.display_name = display_name
        self.body = body
        self.metadata = metadata or {}

    def state_at(self, target_time: Any, observer_context: ObserverContext) -> TargetState:
        apparent = observer_context.topos.at(target_time).observe(self.body).apparent()
        alt, az, distance = apparent.altaz()
        ra, dec, _ = apparent.radec()
        distance_value = None
        distance_unit = None
        if distance is not None:
            distance_value = float(distance.au)
            distance_unit = "au"
        return TargetState(
            kind=self.kind,
            display_name=self.display_name,
            az_deg=float(az.degrees),
            alt_deg=float(alt.degrees),
            ra_deg=float(ra.degrees),
            dec_deg=float(dec.degrees),
            distance=distance_value,
            distance_unit=distance_unit,
            is_visible=float(alt.degrees) >= 0.0,
            plot_mode="sky",
            metadata=dict(self.metadata),
        )


class HorizonsSpacecraftTarget(ResolvedTarget):
    def __init__(
        self,
        *,
        display_name: str,
        spacecraft_id: str,
        samples: list[dict[str, float | str]],
        source: str,
    ):
        self.kind = "spacecraft"
        self.display_name = display_name
        self.spacecraft_id = spacecraft_id
        self.source = source
        self._samples = samples
        self._epoch_seconds = np.array([sample["epoch"] for sample in samples], dtype=float)
        self._az_unwrapped = np.unwrap(np.radians([sample["az_deg"] for sample in samples]))
        self._alt = np.array([sample["alt_deg"] for sample in samples], dtype=float)
        self._ra = np.unwrap(np.radians([sample["ra_deg"] for sample in samples]))
        self._dec = np.array([sample["dec_deg"] for sample in samples], dtype=float)
        self._distance = np.array([sample["distance_au"] for sample in samples], dtype=float)

    @property
    def start_epoch(self) -> float:
        return float(self._epoch_seconds[0])

    @property
    def end_epoch(self) -> float:
        return float(self._epoch_seconds[-1])

    def covers(self, target_time: Any) -> bool:
        epoch = float(target_time.utc_datetime().timestamp())
        return self.start_epoch <= epoch <= self.end_epoch

    def state_at(self, target_time: Any, observer_context: ObserverContext) -> TargetState:
        if not self.covers(target_time):
            start = dt.datetime.fromtimestamp(self.start_epoch, tz=dt.timezone.utc).isoformat()
            end = dt.datetime.fromtimestamp(self.end_epoch, tz=dt.timezone.utc).isoformat()
            raise TargetResolutionError(
                f"Cached spacecraft ephemeris for {self.display_name} only covers {start} to {end}."
            )

        epoch = float(target_time.utc_datetime().timestamp())
        az_deg = math.degrees(float(np.interp(epoch, self._epoch_seconds, self._az_unwrapped))) % 360.0
        ra_deg = math.degrees(float(np.interp(epoch, self._epoch_seconds, self._ra))) % 360.0
        alt_deg = float(np.interp(epoch, self._epoch_seconds, self._alt))
        dec_deg = float(np.interp(epoch, self._epoch_seconds, self._dec))
        distance_au = float(np.interp(epoch, self._epoch_seconds, self._distance))
        return TargetState(
            kind=self.kind,
            display_name=self.display_name,
            az_deg=az_deg,
            alt_deg=alt_deg,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            distance=distance_au,
            distance_unit="au",
            is_visible=alt_deg >= 0.0,
            plot_mode="sky",
            metadata={"source": self.source, "spacecraft_id": self.spacecraft_id},
        )


class TargetResolver:
    def __init__(self, cache_dir: pathlib.Path | None = None):
        self.cache_dir = pathlib.Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache = CacheStore(self.cache_dir)
        self.loader = Loader(str(self.cache_dir / "skyfield"))
        self.ts = self.loader.timescale()
        self._ephemeris = None
        self._constellation_centroids = None
        self._constellation_names = dict(load_constellation_names())
        self._deep_sky_index = self._build_alias_index(DEEP_SKY_OBJECTS)

    def build_observer_context(self, observer_cfg: dict[str, float]) -> ObserverContext:
        surface = wgs84.latlon(
            observer_cfg["lat"],
            observer_cfg["lon"],
            elevation_m=observer_cfg.get("elevation_m", 0.0),
        )
        eph = self._planetary_ephemeris()
        topos = eph["earth"] + surface
        return ObserverContext(
            latitude_deg=float(observer_cfg["lat"]),
            longitude_deg=float(observer_cfg["lon"]),
            elevation_m=float(observer_cfg.get("elevation_m", 0.0)),
            surface=surface,
            topos=topos,
        )

    def search(self, query: str, *, kind: str = "auto", limit: int = 20) -> list[TargetMatch]:
        query = query.strip()
        if not query:
            return []

        if kind == "auto":
            results: list[TargetMatch] = []
            for candidate in ("solar-system", "star", "constellation", "dso", "spacecraft", "satellite"):
                results.extend(self.search(query, kind=candidate, limit=limit))
            deduped = []
            seen = set()
            for result in results:
                key = (result.kind, result.display_name, result.identifier)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(result)
            return deduped[:limit]

        if kind == "solar-system":
            return self._search_solar_system(query, limit=limit)
        if kind == "star":
            return self._search_named_stars(query, limit=limit)
        if kind == "constellation":
            return self._search_constellations(query, limit=limit)
        if kind == "dso":
            return self._search_deep_sky(query, limit=limit)
        if kind == "spacecraft":
            return self._search_spacecraft(query, limit=limit)
        if kind == "satellite":
            return self._search_satellites(query, limit=limit)

        raise TargetResolutionError(f"Unsupported target kind '{kind}'.")

    def resolve(self, spec: TargetSpec, observer_cfg: dict[str, float]) -> ResolvedTarget:
        kind = spec.kind or "auto"
        if kind == "auto":
            kind_candidates = ("solar-system", "star", "constellation", "dso", "spacecraft", "satellite")
            for candidate in kind_candidates:
                try:
                    return self.resolve(TargetSpec(candidate, spec.query, spec.source, spec.identifier), observer_cfg)
                except AmbiguousTargetError:
                    raise
                except TargetResolutionError:
                    continue
            raise TargetResolutionError(f"No supported target matched '{spec.query}'.")

        if kind == "satellite":
            return self._resolve_satellite(spec)
        if kind == "solar-system":
            return self._resolve_solar_system(spec)
        if kind == "spacecraft":
            return self._resolve_spacecraft(spec, observer_cfg)
        if kind == "star":
            return self._resolve_star(spec)
        if kind == "constellation":
            return self._resolve_constellation(spec)
        if kind == "dso":
            return self._resolve_deep_sky(spec)

        raise TargetResolutionError(f"Unsupported target kind '{kind}'.")

    def _planetary_ephemeris(self):
        if self._ephemeris is None:
            self._ephemeris = self.loader("de421.bsp")
        return self._ephemeris

    def _build_alias_index(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        index = {}
        for row in rows:
            index[_normalize_name(row["name"])] = row
            for alias in row.get("aliases", []):
                index[_normalize_name(alias)] = row
        return index

    def _search_solar_system(self, query: str, *, limit: int) -> list[TargetMatch]:
        matches = []
        needle = _normalize_name(query)
        for key, (_eph_key, display_name) in SOLAR_SYSTEM_BODIES.items():
            if needle in _normalize_name(display_name) or needle in key:
                matches.append(
                    TargetMatch(
                        kind="solar-system",
                        query=display_name,
                        display_name=display_name,
                        identifier=key,
                        source="de421",
                    )
                )
        return matches[:limit]

    def _search_named_stars(self, query: str, *, limit: int) -> list[TargetMatch]:
        needle = _normalize_name(query)
        matches = []
        for name, hip_id in named_stars.named_star_dict.items():
            if needle in _normalize_name(name):
                matches.append(
                    TargetMatch(
                        kind="star",
                        query=name,
                        display_name=name,
                        identifier=str(hip_id),
                        source="hipparcos",
                    )
                )
        matches.sort(key=lambda item: item.display_name)
        return matches[:limit]

    def _search_constellations(self, query: str, *, limit: int) -> list[TargetMatch]:
        needle = _normalize_name(query)
        matches = []
        for abbreviation, full_name in self._constellation_names.items():
            if needle in _normalize_name(full_name) or needle == _normalize_name(abbreviation):
                matches.append(
                    TargetMatch(
                        kind="constellation",
                        query=full_name,
                        display_name=full_name,
                        identifier=abbreviation,
                        source="skyfield",
                    )
                )
        matches.sort(key=lambda item: item.display_name)
        return matches[:limit]

    def _search_deep_sky(self, query: str, *, limit: int) -> list[TargetMatch]:
        needle = _normalize_name(query)
        seen = set()
        matches = []
        for row in DEEP_SKY_OBJECTS:
            aliases = [row["name"], *row.get("aliases", [])]
            if any(needle in _normalize_name(alias) for alias in aliases):
                key = row["name"]
                if key in seen:
                    continue
                seen.add(key)
                matches.append(
                    TargetMatch(
                        kind="dso",
                        query=row["name"],
                        display_name=row["name"],
                        identifier=row["aliases"][0] if row.get("aliases") else row["name"],
                        source="curated",
                    )
                )
        matches.sort(key=lambda item: item.display_name)
        return matches[:limit]

    def _search_spacecraft(self, query: str, *, limit: int) -> list[TargetMatch]:
        payload = self._cached_json_request(
            "lookups",
            f"spacecraft-{_format_cache_key(query)}",
            HORIZONS_LOOKUP_URL,
            params={"format": "json", "sstr": query, "group": "sct"},
            ttl_seconds=LOOKUP_CACHE_TTL_SECONDS,
        )
        matches = []
        for row in payload.get("result", [])[:limit]:
            matches.append(
                TargetMatch(
                    kind="spacecraft",
                    query=row["name"],
                    display_name=row["name"],
                    identifier=str(row["spkid"]),
                    source="horizons",
                    description=row.get("pdes"),
                )
            )
        return matches

    def _search_satellites(self, query: str, *, limit: int) -> list[TargetMatch]:
        if query.strip().isdigit():
            return [
                TargetMatch(
                    kind="satellite",
                    query=query.strip(),
                    display_name=f"NORAD {query.strip()}",
                    identifier=query.strip(),
                    source="celestrak",
                )
            ]

        encoded = quote_plus(query.strip())
        payload = self._cached_text_request(
            "satellite-search",
            _format_cache_key(query),
            f"{CELESTRAK_URL}?NAME={encoded}&FORMAT=TLE",
            ttl_seconds=TLE_CACHE_TTL_SECONDS,
        )
        triplets = _parse_tle_triplets(payload)
        matches = []
        for triplet in triplets[:limit]:
            catnr = triplet["line1"][2:7].strip()
            matches.append(
                TargetMatch(
                    kind="satellite",
                    query=triplet["name"],
                    display_name=triplet["name"],
                    identifier=catnr,
                    source="celestrak",
                )
            )
        return matches

    def _resolve_satellite(self, spec: TargetSpec) -> ResolvedTarget:
        if spec.identifier:
            triplet = self._fetch_tle_by_catnr(spec.identifier)
        elif spec.query.strip().isdigit():
            triplet = self._fetch_tle_by_catnr(spec.query.strip())
        else:
            triplet = self._fetch_tle_by_name(spec.query)
        satellite = EarthSatellite(triplet["line1"], triplet["line2"], triplet["name"], self.ts)
        return EarthSatelliteTarget(display_name=triplet["name"], satellite=satellite, source="celestrak")

    def _resolve_solar_system(self, spec: TargetSpec) -> ResolvedTarget:
        key = _normalize_name(spec.identifier or spec.query)
        for body_key, (ephemeris_name, display_name) in SOLAR_SYSTEM_BODIES.items():
            if key in {_normalize_name(body_key), _normalize_name(display_name)}:
                body = self._planetary_ephemeris()[ephemeris_name]
                return SkyObjectTarget(
                    kind="solar-system",
                    display_name=display_name,
                    body=body,
                    metadata={"source": "de421", "ephemeris_name": ephemeris_name},
                )
        raise TargetResolutionError(f"No supported solar-system body matched '{spec.query}'.")

    def _resolve_spacecraft(self, spec: TargetSpec, observer_cfg: dict[str, float]) -> ResolvedTarget:
        matches = []
        spacecraft_id = spec.identifier
        display_name = spec.query
        if spacecraft_id is None:
            matches = self._search_spacecraft(spec.query, limit=20)
            if not matches:
                raise TargetResolutionError(f"No interplanetary spacecraft matched '{spec.query}'.")
            exact = [match for match in matches if _normalize_name(match.display_name) == _normalize_name(spec.query)]
            if len(exact) == 1:
                selected = exact[0]
            elif len(matches) == 1:
                selected = matches[0]
            else:
                raise AmbiguousTargetError(spec.query, matches)
            spacecraft_id = selected.identifier
            display_name = selected.display_name

        samples = self._fetch_spacecraft_ephemeris(
            spacecraft_id=spacecraft_id,
            display_name=display_name,
            observer_cfg=observer_cfg,
        )
        return HorizonsSpacecraftTarget(
            display_name=display_name,
            spacecraft_id=spacecraft_id,
            samples=samples,
            source="horizons",
        )

    def _resolve_star(self, spec: TargetSpec) -> ResolvedTarget:
        key = spec.query
        if spec.identifier:
            hip_id = int(spec.identifier)
            display_name = spec.query
        else:
            matches = self._search_named_stars(spec.query, limit=20)
            if not matches:
                raise TargetResolutionError(f"No named star matched '{spec.query}'.")
            exact = [match for match in matches if _normalize_name(match.display_name) == _normalize_name(spec.query)]
            if len(exact) == 1:
                selected = exact[0]
            elif len(matches) == 1:
                selected = matches[0]
            else:
                raise AmbiguousTargetError(spec.query, matches)
            hip_id = int(selected.identifier)
            display_name = selected.display_name
            key = selected.display_name

        star_row = self._load_hipparcos_star(hip_id)
        star = Star(ra_hours=star_row["ra_deg"] / 15.0, dec_degrees=star_row["dec_deg"])
        return SkyObjectTarget(
            kind="star",
            display_name=display_name,
            body=star,
            metadata={"source": "hipparcos", "hip": hip_id, "query": key},
        )

    def _resolve_constellation(self, spec: TargetSpec) -> ResolvedTarget:
        centroids = self._constellation_centroids_map()
        key = spec.identifier or spec.query
        normalized = _normalize_name(key)
        for abbreviation, payload in centroids.items():
            full_name = payload["name"]
            if normalized in {_normalize_name(abbreviation), _normalize_name(full_name)}:
                star = Star(
                    ra_hours=float(payload["ra_deg"]) / 15.0,
                    dec_degrees=float(payload["dec_deg"]),
                )
                return SkyObjectTarget(
                    kind="constellation",
                    display_name=full_name,
                    body=star,
                    metadata={"source": "skyfield", "abbreviation": abbreviation},
                )
        raise TargetResolutionError(f"No constellation matched '{spec.query}'.")

    def _resolve_deep_sky(self, spec: TargetSpec) -> ResolvedTarget:
        row = self._deep_sky_index.get(_normalize_name(spec.identifier or spec.query))
        if row is None:
            raise TargetResolutionError(f"No curated deep-sky object matched '{spec.query}'.")
        star = Star(ra_hours=float(row["ra_hours"]), dec_degrees=float(row["dec_degrees"]))
        return SkyObjectTarget(
            kind="dso",
            display_name=row["name"],
            body=star,
            metadata={"source": "curated", "aliases": row.get("aliases", [])},
        )

    def _cached_text_request(self, bucket: str, key: str, url: str, *, ttl_seconds: int) -> str:
        payload = self.cache.load_json(bucket, key, ttl_seconds=ttl_seconds)
        if payload is not None and "text" in payload:
            return payload["text"]
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = {"text": response.text}
        self.cache.save_json(bucket, key, payload)
        return response.text

    def _cached_json_request(
        self,
        bucket: str,
        key: str,
        url: str,
        *,
        params: dict[str, Any],
        ttl_seconds: int,
    ) -> dict[str, Any]:
        payload = self.cache.load_json(bucket, key, ttl_seconds=ttl_seconds)
        if payload is not None:
            return payload
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        self.cache.save_json(bucket, key, payload)
        return payload

    def _fetch_tle_by_catnr(self, catnr: str) -> dict[str, str]:
        text = self._cached_text_request(
            "tle",
            f"catnr-{catnr}",
            f"{CELESTRAK_URL}?CATNR={int(catnr)}&FORMAT=TLE",
            ttl_seconds=TLE_CACHE_TTL_SECONDS,
        )
        triplets = _parse_tle_triplets(text)
        if not triplets:
            raise TargetResolutionError(f"No TLE found for catalog number {catnr}.")
        return triplets[0]

    def _fetch_tle_by_name(self, name: str) -> dict[str, str]:
        encoded = quote_plus(name.strip())
        text = self._cached_text_request(
            "tle",
            f"name-{_format_cache_key(name)}",
            f"{CELESTRAK_URL}?NAME={encoded}&FORMAT=TLE",
            ttl_seconds=TLE_CACHE_TTL_SECONDS,
        )
        triplets = _parse_tle_triplets(text)
        if not triplets:
            raise TargetResolutionError(f"No TLE found for '{name}'.")
        exact = [triplet for triplet in triplets if _normalize_name(triplet["name"]) == _normalize_name(name)]
        if len(exact) == 1:
            return exact[0]
        if len(triplets) == 1:
            return triplets[0]
        matches = [
            TargetMatch(
                kind="satellite",
                query=triplet["name"],
                display_name=triplet["name"],
                identifier=triplet["line1"][2:7].strip(),
                source="celestrak",
            )
            for triplet in triplets[:20]
        ]
        raise AmbiguousTargetError(name, matches)

    def _constellation_centroids_map(self) -> dict[str, dict[str, float | str]]:
        if self._constellation_centroids is not None:
            return self._constellation_centroids

        payload = self.cache.load_json("catalogs", "constellation-centroids")
        if payload is not None:
            self._constellation_centroids = payload
            return payload

        constellation_at = load_constellation_map()
        centroids: dict[str, list[np.ndarray]] = {abbr: [] for abbr in self._constellation_names}
        for ra_deg in np.arange(0.0, 360.0, 2.0):
            for dec_deg in np.arange(-88.0, 90.0, 2.0):
                position = position_of_radec(ra_hours=ra_deg / 15.0, dec_degrees=dec_deg)
                abbreviation = constellation_at(position)
                ra_rad = math.radians(ra_deg)
                dec_rad = math.radians(dec_deg)
                vector = np.array(
                    [
                        math.cos(dec_rad) * math.cos(ra_rad),
                        math.cos(dec_rad) * math.sin(ra_rad),
                        math.sin(dec_rad),
                    ]
                )
                centroids[abbreviation].append(vector)

        payload = {}
        for abbreviation, vectors in centroids.items():
            mean_vector = np.mean(vectors, axis=0)
            mean_vector = mean_vector / np.linalg.norm(mean_vector)
            ra_deg = math.degrees(math.atan2(mean_vector[1], mean_vector[0])) % 360.0
            dec_deg = math.degrees(math.asin(mean_vector[2]))
            payload[abbreviation] = {
                "name": self._constellation_names[abbreviation],
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
            }

        self.cache.save_json("catalogs", "constellation-centroids", payload)
        self._constellation_centroids = payload
        return payload

    def _load_hipparcos_star(self, hip_id: int) -> dict[str, float]:
        cache_key = f"hip-{hip_id}"
        payload = self.cache.load_json("hipparcos", cache_key)
        if payload is not None:
            return payload

        catalog_path = self.cache_dir / "hip_main.dat"
        if not catalog_path.exists():
            response = requests.get("https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat", timeout=120)
            response.raise_for_status()
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            catalog_path.write_bytes(response.content)

        with catalog_path.open("r", encoding="latin-1") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("|")
                if len(fields) < 10:
                    continue
                if fields[1].strip() != str(hip_id):
                    continue
                payload = {
                    "hip": hip_id,
                    "ra_deg": float(fields[8]),
                    "dec_deg": float(fields[9]),
                }
                self.cache.save_json("hipparcos", cache_key, payload)
                return payload

        raise TargetResolutionError(f"HIP {hip_id} was not found in the Hipparcos catalog.")

    def _fetch_spacecraft_ephemeris(
        self,
        *,
        spacecraft_id: str,
        display_name: str,
        observer_cfg: dict[str, float],
    ) -> list[dict[str, float | str]]:
        now = dt.datetime.now(dt.timezone.utc)
        window_start = (now - dt.timedelta(hours=SPACECRAFT_WINDOW_HOURS)).replace(minute=0, second=0, microsecond=0)
        window_stop = window_start + dt.timedelta(hours=SPACECRAFT_WINDOW_HOURS * 2)
        cache_key = "-".join(
            [
                f"id{spacecraft_id}",
                f"lat{observer_cfg['lat']:.4f}",
                f"lon{observer_cfg['lon']:.4f}",
                f"elev{observer_cfg.get('elevation_m', 0.0):.1f}",
                window_start.strftime("%Y%m%d%H"),
            ]
        )
        cached = self.cache.load_json("spacecraft", cache_key)
        if cached is not None:
            return cached["samples"]

        lon = float(observer_cfg["lon"])
        lat = float(observer_cfg["lat"])
        elevation_km = float(observer_cfg.get("elevation_m", 0.0)) / 1000.0
        params = {
            "format": "json",
            "COMMAND": f"'{spacecraft_id}'",
            "OBJ_DATA": "NO",
            "MAKE_EPHEM": "YES",
            "EPHEM_TYPE": "OBSERVER",
            "CENTER": "'coord@399'",
            "SITE_COORD": f"'{lon:.6f},{lat:.6f},{elevation_km:.6f}'",
            "START_TIME": f"'{window_start.strftime('%Y-%m-%d %H:%M')}'",
            "STOP_TIME": f"'{window_stop.strftime('%Y-%m-%d %H:%M')}'",
            "STEP_SIZE": f"'{SPACECRAFT_STEP_MINUTES} min'",
            "QUANTITIES": "'1,4,20'",
        }
        response = requests.get(HORIZONS_URL, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        samples = parse_horizons_ephemeris(payload["result"])
        if not samples:
            raise TargetResolutionError(f"Horizons returned no ephemeris samples for {display_name}.")

        self.cache.save_json(
            "spacecraft",
            cache_key,
            {
                "display_name": display_name,
                "spacecraft_id": spacecraft_id,
                "generated_at": _to_iso_utc(now),
                "window_start": _to_iso_utc(window_start),
                "window_stop": _to_iso_utc(window_stop),
                "samples": samples,
            },
        )
        return samples


def parse_horizons_ephemeris(text: str) -> list[dict[str, float | str]]:
    start = text.find("$$SOE")
    end = text.find("$$EOE")
    if start == -1 or end == -1 or end <= start:
        return []

    samples = []
    for line in text[start + 5:end].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 12:
            continue
        timestamp_text = " ".join(parts[0:2])
        epoch = dt.datetime.strptime(timestamp_text, "%Y-%b-%d %H:%M").replace(tzinfo=dt.timezone.utc)
        cursor = 2
        while cursor < len(parts) and not _looks_like_ra_hour_token(parts[cursor]):
            cursor += 1
        if cursor + 8 >= len(parts):
            continue
        ra_deg = _sexagesimal_ra_to_degrees(parts[cursor], parts[cursor + 1], parts[cursor + 2])
        dec_deg = _sexagesimal_dec_to_degrees(parts[cursor + 3], parts[cursor + 4], parts[cursor + 5])
        az_deg = float(parts[cursor + 6])
        alt_deg = float(parts[cursor + 7])
        distance_au = float(parts[cursor + 8])
        samples.append(
            {
                "timestamp": _to_iso_utc(epoch),
                "epoch": float(epoch.timestamp()),
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "az_deg": az_deg,
                "alt_deg": alt_deg,
                "distance_au": distance_au,
            }
        )
    return samples
