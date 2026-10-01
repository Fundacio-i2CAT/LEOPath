"""Build population points for North America from the national statistics offices.

Downloads three public files, records their SHA-256, and writes one CSV of
population points (country, lat, lon, population, spread_deg) for
``terminal_population.py --layout census``:

- United States: US Census Bureau county population estimates (vintage 2024,
  ``co-est2024-alldata.csv``) joined with the 2024 Gazetteer county
  internal points. One point per county; counties are large, so terminals
  drawn from one spread 0.2 degrees (about 20 km) around it.
- Mexico: INEGI, Censo de Poblacion y Vivienda 2020, ITER localities: every
  locality with its coordinates and total population. Spread 0.01 degrees.
- Canada: Statistics Canada, 2021 Census Geographic Attribute File
  (92-151-X): dissemination-block population, placed at its dissemination
  area's representative point. Spread 0.01 degrees.

Usage: python scripts/fetch_population_points.py OUTPUT_DIR
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "usa_population": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv",
    "usa_gazetteer": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_counties_national.zip",
    "mexico_iter": "https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/iter/iter_00_cpv2020_csv.zip",
    "canada_gaf": "https://www12.statcan.gc.ca/census-recensement/2021/geo/aip-pia/attribute-attribs/files-fichiers/2021_92-151_X.zip",
}


def fetch(name: str, cache: Path) -> bytes:
    path = cache / Path(SOURCES[name]).name
    if not path.exists():
        request = urllib.request.Request(SOURCES[name], headers={"User-Agent": "LEOPath"})
        with urllib.request.urlopen(request, timeout=300) as response:
            path.write_bytes(response.read())
    return path.read_bytes()


def zipped_csv(data: bytes, pattern: str) -> io.TextIOWrapper:
    archive = zipfile.ZipFile(io.BytesIO(data))
    member = next(n for n in archive.namelist() if re.search(pattern, n, re.I))
    return io.TextIOWrapper(archive.open(member), encoding="latin-1", newline="")


def usa(cache: Path) -> list[tuple]:
    population = {}
    rows = csv.DictReader(io.StringIO(fetch("usa_population", cache).decode("latin-1")))
    for row in rows:
        if row["SUMLEV"] == "050":
            population[row["STATE"].zfill(2) + row["COUNTY"].zfill(3)] = int(row["POPESTIMATE2024"])
    points = []
    gazetteer = zipped_csv(fetch("usa_gazetteer", cache), r"\.txt$")
    for row in csv.DictReader(gazetteer, delimiter="\t"):
        row = {k.strip(): v.strip() for k, v in row.items()}
        pop = population.get(row["GEOID"])
        if pop:
            points.append(("usa", float(row["INTPTLAT"]), float(row["INTPTLONG"]), pop, 0.2))
    return points


def _dms(value: str) -> float | None:
    """INEGI writes coordinates as degrees, minutes and seconds, e.g. 102°17'45.768" W."""
    numbers = re.findall(r"[\d.]+", value)
    if len(numbers) < 3:
        return None
    degrees = float(numbers[0]) + float(numbers[1]) / 60 + float(numbers[2]) / 3600
    return -degrees if re.search(r"[SWO]", value) else degrees


def mexico(cache: Path) -> list[tuple]:
    points = []
    reader = csv.DictReader(zipped_csv(fetch("mexico_iter", cache), r"conjunto_de_datos.*\.csv$"))
    for row in reader:
        # Locality rows only: 0 is a state or municipal total, 9998/9999 groups of small ones.
        if row["LOC"] in ("0000", "0", "9998", "9999"):
            continue
        lat, lon = _dms(row["LATITUD"]), _dms(row["LONGITUD"])
        try:
            pop = int(row["POBTOT"])
        except ValueError:
            continue
        if lat is not None and lon is not None and pop > 0:
            points.append(("mexico", lat, lon, pop, 0.01))
    return points


def canada(cache: Path) -> list[tuple]:
    points = []
    for row in csv.DictReader(zipped_csv(fetch("canada_gaf", cache), r"\.csv$")):
        try:
            pop = int(row["DBPOP2021_IDPOP2021"])
            # Blocks carry population; coordinates come at the dissemination-area
            # level (a few hundred people), so each block sits at its area's point.
            lat, lon = float(row["DARPLAT_ADLAT"]), float(row["DARPLONG_ADLONG"])
        except (KeyError, ValueError):
            continue
        if pop > 0:
            points.append(("canada", lat, lon, pop, 0.01))
    return points


def main() -> None:
    out = Path(sys.argv[1])
    cache = out / "sources"
    cache.mkdir(parents=True, exist_ok=True)
    points = usa(cache) + mexico(cache) + canada(cache)
    with open(out / "population_points.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["country", "lat", "lon", "population", "spread_deg"])
        writer.writerows(points)
    with open(out / "SOURCES.txt", "w") as handle:
        for name, url in SOURCES.items():
            digest = hashlib.sha256((cache / Path(url).name).read_bytes()).hexdigest()
            handle.write(f"{name}\t{url}\tsha256:{digest}\n")
    for country in ("usa", "mexico", "canada"):
        rows = [p for p in points if p[0] == country]
        print(f"{country}: {len(rows)} points, population {sum(p[3] for p in rows):,}")


if __name__ == "__main__":
    main()
