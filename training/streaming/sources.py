"""Remote image sources. Every source streams bytes straight into the ImageStore (RAM only).

Sources
-------
SDSSSource     SDSS SkyServer spectroscopic classes (GALAXY / STAR / QSO) + image cutouts.
DecalsSource   Hugging Face ``Smith42/galaxies`` (DECaLS galaxy images, parquet range reads).
NASASource     NASA Images API (images-api.nasa.gov), keyword queries + metadata filters.
ESASource      Hugging Face ``Supermaxman/esa-hubble`` with ESA's own object-type metadata.
CommonsSource  Wikimedia Commons categories (thumbnails only).
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional

import requests

from . import CLASS_TO_IDX
from .store import ImageStore

USER_AGENT = "AstroClassAI-Research/1.0 (NASA deep space classifier training; contact: rafa@tarracoimportexport.com)"

_tls = threading.local()


def session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _tls.s = s
    return s


def _retry_after(value, default: float) -> float:
    """Retry-After may be seconds or an HTTP date; never trust it blindly (cap at 60 s)."""
    try:
        return min(60.0, max(0.5, float(value)))
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone

        return min(60.0, max(0.5, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()))
    except Exception:  # noqa: BLE001
        return min(60.0, default)


def http_json(url: str, **kw):
    r = http_get(url, **kw)
    if r is None:
        return None
    try:
        return r.json()
    except ValueError:  # an HTML error page served with status 200
        return None


def http_get(url: str, *, params=None, timeout: float = 30, tries: int = 4, stream_ok=(200,)) -> Optional[requests.Response]:
    """GET with exponential backoff. Returns None on permanent failure."""
    delay = 1.0
    for attempt in range(tries):
        try:
            r = session().get(url, params=params, timeout=timeout)
            if r.status_code in stream_ok:
                return r
            if r.status_code in (403, 404, 410):
                return None
            if r.status_code == 429:
                time.sleep(_retry_after(r.headers.get("Retry-After"), delay * 2))
        except requests.RequestException:
            pass
        time.sleep(delay)
        delay = min(delay * 2, 20)
    return None


class Source:
    """Base class. Subclasses implement ``run``; ``progress`` is read by the status line."""

    def __init__(self, name: str, cls: str, group: str, quota: int):
        self.name, self.cls, self.group, self.quota = name, cls, group, quota
        self.cls_idx = CLASS_TO_IDX[cls]
        self.attempted = 0
        self.done = False
        self.error: Optional[str] = None

    def accepted(self, store: ImageStore) -> int:
        with store.lock:
            return store.counts[(self.cls_idx, self.group)]

    def full(self, store: ImageStore) -> bool:
        return self.accepted(store) >= self.quota

    def run(self, store: ImageStore, stop: threading.Event) -> None:  # pragma: no cover
        raise NotImplementedError


def _bounded_pool(workers: int):
    return ThreadPoolExecutor(max_workers=workers), threading.BoundedSemaphore(workers * 3)


def _submit(pool, sem, stop, fn, *args):
    sem.acquire()
    if stop.is_set():
        sem.release()
        return None
    fut = pool.submit(fn, *args)
    fut.add_done_callback(lambda _f: sem.release())
    return fut


# SDSS
SDSS_SQL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
SDSS_CUT = "https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"


class SDSSSource(Source):
    """Spectroscopically *confirmed* classes from SDSS DR18 (SpecPhoto), rendered as image cutouts."""

    def __init__(self, name, cls, group, quota, sdss_class, where, scale_mode="fixed", scale=0.4, workers=20, modulus=7):
        super().__init__(name, cls, group, quota)
        self.sdss_class, self.where, self.scale_mode, self.scale = sdss_class, where, scale_mode, scale
        self.workers, self.modulus = workers, modulus

    def _rows(self, rng: random.Random) -> List[dict]:
        k = rng.randrange(self.modulus)
        top = int(self.quota * 1.6) + 200
        cmd = (
            f"SELECT TOP {top} s.specObjID AS sid, s.ra, s.dec, p.petroR90_r AS r90 "
            f"FROM SpecPhoto s JOIN PhotoObjAll p ON p.objID = s.objID "
            f"WHERE s.class='{self.sdss_class}' AND s.zWarning=0 AND {self.where} "
            f"AND (s.specObjID % {self.modulus}) = {k}"
        )
        r = http_get(SDSS_SQL, params={"cmd": cmd, "format": "json"}, timeout=600, tries=3)
        if r is None:
            raise RuntimeError("SDSS SQL failed")
        data = r.json()
        if isinstance(data, dict):
            raise RuntimeError(f"SDSS SQL error: {data.get('ErrorMessage')}")
        rows = data[0]["Rows"]
        rng.shuffle(rows)
        return rows

    def _scale(self, row) -> float:
        if self.scale_mode == "r90":  # frame ~5.5x the Petrosian R90 so the galaxy fills the picture
            r90 = row.get("r90") or 5.0
            return float(min(1.6, max(0.4, r90 * 5.5 / 256.0)))
        return self.scale

    def _one(self, store: ImageStore, row: dict) -> None:
        if not store.wants(f"sdss:{row['sid']}"):
            return
        self.attempted += 1
        params = {"ra": row["ra"], "dec": row["dec"], "scale": f"{self._scale(row):.4f}", "width": 256, "height": 256}
        r = http_get(SDSS_CUT, params=params, timeout=40, tries=3)
        if r is not None and len(r.content) > 1500:
            store.add_raw(self.cls_idx, self.group, f"sdss:{row['sid']}", r.content)

    def run(self, store, stop):
        try:
            rows = self._rows(random.Random(hash(self.name) & 0xFFFF))
            pool, sem = _bounded_pool(self.workers)
            with pool:
                for row in rows:
                    if stop.is_set() or self.full(store):
                        break
                    _submit(pool, sem, stop, self._one, store, row)
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        finally:
            self.done = True


# DECaLS (HF)
class DecalsSource(Source):
    """DECaLS galaxy images from the Hugging Face dataset Smith42/galaxies (parquet range reads)."""

    REPO = "datasets/Smith42/galaxies/data"

    def __init__(self, name, quota, shards=14, workers=6):
        super().__init__(name, "galaxy", "decals", quota)
        self.n_shards, self.workers = shards, workers

    def _read_shard(self, store, stop, path, rng):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        try:
            pf = pq.ParquetFile(fs.open(path, "rb"))
            groups = list(range(pf.metadata.num_row_groups))
            rng.shuffle(groups)
            per_shard = max(8, self.quota // self.n_shards // 100 + 6)
            for gi in groups[:per_shard]:
                if stop.is_set() or self.full(store):
                    return
                tb = pf.read_row_group(gi, columns=["image", "dr8_id"])
                imgs, ids = tb.column("image").to_pylist(), tb.column("dr8_id").to_pylist()
                for im, did in zip(imgs, ids):
                    if self.full(store):
                        return
                    if not store.wants(f"decals:{did}"):
                        continue
                    self.attempted += 1
                    store.add_raw(self.cls_idx, self.group, f"decals:{did}", im["bytes"])
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)

    def run(self, store, stop):
        try:
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()
            files = sorted(p for p in fs.ls(self.REPO, detail=False) if "/train-" in p)
            rng = random.Random(7)
            picks = rng.sample(files, min(self.n_shards, len(files)))
            with ThreadPoolExecutor(self.workers) as pool:
                futs = [pool.submit(self._read_shard, store, stop, p, random.Random(i)) for i, p in enumerate(picks)]
                for f in futs:
                    f.result()
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        finally:
            self.done = True


# NASA API
NASA_SEARCH = "https://images-api.nasa.gov/search"

# Junk signals. Title/keywords only for the broad list (descriptions of good pictures routinely mention
# "team", "instrument", "photograph"...), plus a short list checked in the first 180 chars of the description.
NASA_JUNK_TITLE = re.compile(
    r"astronaut|artist|\bconcept\b|illustrat|diagram|\bchart\b|\blogo\b|poster|infograph|animation|simulation|"
    r"\blaunch|rocket|\bcrew\b|engineer|technician|\btesting\b|facility|\brover\b|\blander\b|portrait|"
    r"press conference|ceremony|spectrum|spectra|\bplot\b|\bmap\b|\bglobe\b|painting|drawing|cartoon|"
    r"space ?suit|shuttle|space station|clean ?room|\bassembly\b|\bvideo\b|lecture|\baward\b|medal|\bpatch\b|"
    r"mission control|nustar|payload|hardware|hangar|launch pad|\btruck\b|test image|mirror segment|building",
    re.I,
)
NASA_JUNK_HEAD = re.compile(
    r"astronaut|artist.?s (concept|rendering|illustration|impression)|illustration|animation|simulation|rocket|"
    r"\bcrew\b|clean ?room|technician|engineers? (at|work|inspect)|mission control|payload|"
    r"\blaunch(ed)? (of|from|pad)|press conference",
    re.I,
)
_nasa_seen: set = set()
_nasa_seen_lock = threading.Lock()


class NASASource(Source):
    def __init__(self, name, cls, quota, queries: Iterable[str], must_match: Optional[re.Pattern] = None,
                 exclude_title: Optional[re.Pattern] = None, exclude_head: Optional[re.Pattern] = None,
                 workers=14, max_pages=100):
        super().__init__(name, cls, "nasa", quota)
        self.queries, self.must_match, self.workers, self.max_pages = list(queries), must_match, workers, max_pages
        self.exclude_title = exclude_title  # applied to title + keywords only (descriptions mention neighbours)
        self.exclude_head = exclude_head  # title + keywords + first 180 chars of the description

    def _fetch(self, store, nasa_id: str, href: str) -> None:
        if not store.wants(f"nasa:{nasa_id}"):
            return
        self.attempted += 1
        urls = []
        if "~thumb" in href:
            urls.append(href.replace("~thumb", "~small"))
            urls.append(href.replace("~thumb", "~medium"))
        urls.append(href)
        for u in urls:
            r = http_get(u, timeout=40, tries=2)
            if r is not None and len(r.content) > 4000:
                store.add_raw(self.cls_idx, self.group, f"nasa:{nasa_id}", r.content)
                return

    def run(self, store, stop):
        try:
            pool, sem = _bounded_pool(self.workers)
            with pool:
                for q in self.queries:
                    for page in range(1, self.max_pages + 1):
                        if stop.is_set() or self.full(store):
                            return
                        j = http_json(
                            NASA_SEARCH,
                            params={"q": q, "media_type": "image", "page": page, "page_size": 100},
                            timeout=60,
                            tries=4,
                        )
                        if j is None:
                            break
                        items = j.get("collection", {}).get("items", [])
                        if not items:
                            break
                        for it in items:
                            d = it.get("data", [{}])[0]
                            links = it.get("links") or []
                            nid = d.get("nasa_id")
                            if not nid or not links:
                                continue
                            kw = " ".join(str(k) for k in (d.get("keywords") or []))
                            ttl = f"{d.get('title', '')} {kw}"
                            desc = str(d.get("description", ""))
                            head180, head300 = f"{ttl} {desc[:180]}", f"{ttl} {desc[:300]}"
                            if NASA_JUNK_TITLE.search(ttl) or NASA_JUNK_HEAD.search(head180):
                                continue
                            if self.exclude_head is not None and self.exclude_head.search(head180):
                                continue
                            if self.exclude_title is not None and self.exclude_title.search(ttl):
                                continue
                            # class evidence must be in the title/keywords or the opening of the description
                            if self.must_match is not None and not self.must_match.search(head300):
                                continue
                            with _nasa_seen_lock:
                                if nid in _nasa_seen:
                                    continue
                                _nasa_seen.add(nid)
                            _submit(pool, sem, stop, self._fetch, store, nid, links[0]["href"])
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        finally:
            self.done = True


# ESA (HF)
class ESASource(Source):
    """ESA/Hubble images with ESA's own object-type taxonomy -> project classes."""

    REPO = "datasets/Supermaxman/esa-hubble/data"

    def __init__(self, name, workers=3):  # shards are 30-250 MB each: keep memory in check
        super().__init__(name, "galaxy", "esa", 10**9)  # class is decided per row
        self.workers = workers
        self.n_by_class: Dict[str, int] = {}

    @staticmethod
    def classify(type_str: str, category: str) -> Optional[str]:
        t = (type_str or "")
        hits = set()
        if "Nebula" in t:
            hits.add("nebula")
        if "Galaxy" in t:
            hits.add("quasar" if re.search(r"Quasar|AGN : Quasar", t) else "galaxy")
        if re.search(r"Cluster : (Open|Globular)|Star Cluster", t) or category == "Star Clusters":
            hits.add("star")
        if "Solar System" in t and re.search(r"\bPlanet\b(?!ary)|\bMoon\b", t) and not re.search(r"Exoplanet|Planetary System", t):
            hits.add("planet")
        if len(hits) == 1:
            return next(iter(hits))
        return None

    def _shard(self, store, stop, path):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        try:
            if stop.is_set():
                return
            pf = pq.ParquetFile(HfFileSystem().open(path, "rb"))
            names = pf.schema_arrow.names
            cols = [c for c in ("id", "Type", "Category", "image") if c in names]
            df = pf.read(columns=cols).to_pydict()
            for i in range(len(df["id"])):
                if stop.is_set():
                    return
                cls = self.classify(df.get("Type", [""])[i], df.get("Category", [""])[i])
                if cls is None:
                    continue
                img = df["image"][i]
                if not img or not img.get("bytes"):
                    continue
                self.attempted += 1
                if store.add_raw(CLASS_TO_IDX[cls], self.group, f"esa:{df['id'][i]}", img["bytes"]) == "ok":
                    self.n_by_class[cls] = self.n_by_class.get(cls, 0) + 1
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)

    def run(self, store, stop):
        try:
            from huggingface_hub import HfFileSystem

            files = sorted(p for p in HfFileSystem().ls(self.REPO, detail=False) if p.endswith(".parquet"))
            pool, sem = _bounded_pool(self.workers)
            with pool:
                for p in files:  # bounded submission: stop_fetch really stops the downloads
                    if stop.is_set():
                        break
                    _submit(pool, sem, stop, self._shard, store, stop, p)
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        finally:
            self.done = True

    def full(self, store):  # never "full"
        return False


# Wikimedia Commons
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_BAD_TITLE = re.compile(
    r"map|diagram|chart|logo|illustrat|artist|concept|drawing|painting|cartoon|stamp|poster|infograph|"
    r"comparison|animation|labell?ed|annotated|graph|spectrum|sketch|simulation|render|orbit|"
    r"telescope|observatory|spacecraft|rocket|astronaut|screenshot|diagrama|\.svg|\.gif|\.tif|\.pdf|\.webm|\.ogv",
    re.I,
)
COMMONS_BAD_CAT = re.compile(
    r"\bart\b|painting|drawing|cartoon|stamp|poster|logo|mathemat|diagram|\bmaps?\b|film|video|stub|book|"
    r"people|astronaut|telescopes?\b|observator|spacecraft|rocket|constellations?$|sculpture|"
    r"model|toy|game|fiction|popular culture|mythology|coin|literature|music|science fiction|"
    r"meteor|aurora|eclipse|earth|sun\b|solar (flare|eclipse)|comet|asteroid",
    re.I,
)


class CommonsSource(Source):
    def __init__(self, name, cls, quota, roots: Iterable[str], depth=2, workers=8, max_cats=60,
                 extra_bad: Optional[re.Pattern] = None):
        super().__init__(name, cls, "commons", quota)
        self.roots, self.depth, self.workers, self.max_cats = list(roots), depth, workers, max_cats
        self.extra_bad = extra_bad  # other-class vocabulary: applied to category AND file titles

    def _api(self, params) -> Optional[dict]:
        return http_json(COMMONS_API, params={**params, "format": "json"}, timeout=40, tries=4)

    def _files(self, cat: str):
        cont = {}
        while True:
            j = self._api(
                {
                    "action": "query",
                    "generator": "categorymembers",
                    "gcmtitle": cat,
                    "gcmtype": "file",
                    "gcmlimit": 50,
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime",
                    "iiurlwidth": 500,
                    **cont,
                }
            )
            if not j:
                return
            for p in (j.get("query", {}).get("pages", {}) or {}).values():
                ii = (p.get("imageinfo") or [{}])[0]
                if ii.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                title = p.get("title", "")
                if min(ii.get("width", 0), ii.get("height", 0)) < 400 or COMMONS_BAD_TITLE.search(title):
                    continue
                if self.extra_bad is not None and self.extra_bad.search(title):
                    continue
                yield p["pageid"], ii.get("thumburl") or ii.get("url")
            cont = (j.get("continue") or {})
            if not cont:
                return

    def _subcats(self, cat: str) -> List[str]:
        j = self._api({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": "subcat", "cmlimit": 500})
        if not j:
            return []
        return [m["title"] for m in j.get("query", {}).get("categorymembers", []) if not COMMONS_BAD_CAT.search(m["title"])
                and not (self.extra_bad is not None and self.extra_bad.search(m["title"]))]

    def _fetch(self, store, pid, url):
        if not store.wants(f"commons:{pid}"):
            return
        self.attempted += 1
        r = http_get(url, timeout=40, tries=3)
        if r is not None and len(r.content) > 4000:
            store.add_raw(self.cls_idx, self.group, f"commons:{pid}", r.content)

    def run(self, store, stop):
        try:
            pool, sem = _bounded_pool(self.workers)
            seen_cats, frontier = set(), [(c, 0) for c in self.roots]
            with pool:
                while frontier and len(seen_cats) < self.max_cats:
                    cat, d = frontier.pop(0)
                    if cat in seen_cats:
                        continue
                    seen_cats.add(cat)
                    for pid, url in self._files(cat):
                        if stop.is_set() or self.full(store):
                            return
                        _submit(pool, sem, stop, self._fetch, store, pid, url)
                    if d < self.depth:
                        frontier.extend((c, d + 1) for c in self._subcats(cat))
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
        finally:
            self.done = True


# the recipe
PLANET_TERMS = re.compile(
    r"\bjupiter|\bsaturn|\bmars\b|\bmartian|\bvenus\b|\bmercury\b|\bneptune|\buranus|\bpluto|\btitan\b|\beuropa\b|"
    r"\benceladus|\bganymede|\bcallisto|\bio\b|\btriton\b|\bcharon\b|\brhea\b|\bdione\b|\btethys|\bmimas|\biapetus|"
    r"\bphobos|\bdeimos|\bceres\b|\bvesta\b|\bmoons?\b|\blunar\b|\bplanets?\b",
    re.I,
)
PLANET_NOT = re.compile(
    r"\bearth\b|saturn (v|1b|ib)\b|titan (i|ii|iii|iv|rocket)\b|\bsun\b|\bsolar (flare|eclipse|wind|storm|panel|array|cell)|"
    r"eclipse|\btransit of|comet|aurora|hurricane|exoplanet|apollo|\bmagellanic",
    re.I,
)


def build_recipe(target: int = 20000) -> List[Source]:
    """Sources aiming at `target` images per class. Quotas are upper bounds, not guarantees.

    Curated sources (NASA / ESA / Commons) are kept small and filtered hard: they carry the
    "pretty picture" look of real-world inputs. Survey sources (SDSS, DECaLS) provide volume
    with *spectroscopically / catalog confirmed* labels. Wikimedia categories for star, planet
    and quasar were audited and dropped: too many charts, diagrams and landscapes.
    """
    sdss_gal_where = "s.modelMag_r BETWEEN 13 AND 17.7 AND p.petroR90_r > 5"
    not_nebula = re.compile(r"nebula|remnant|supernova|galax|planet|black hole|quasar|cluster of galaxies", re.I)
    not_galaxy = re.compile(r"nebula|remnant|supernova|planet|\bstar cluster|globular|open cluster|milky way", re.I)
    not_nebula_title = re.compile(r"galax|quasar|planet(?!ary nebula)|jupiter|saturn|mars\b", re.I)
    return [
        # galaxy
        DecalsSource("decals_galaxies", quota=int(target * 0.55)),
        SDSSSource("sdss_galaxy", "galaxy", "sdss", int(target * 0.35), "GALAXY", sdss_gal_where, "r90", workers=16),
        NASASource(
            "nasa_galaxy",
            "galaxy",
            10**9,
            ["galaxy", "spiral galaxy", "elliptical galaxy", "interacting galaxies", "galaxy cluster", "dwarf galaxy",
             "irregular galaxy", "andromeda galaxy", "hubble deep field", "barred spiral", "whirlpool galaxy",
             "sombrero galaxy", "webb galaxy", "lenticular galaxy", "starburst galaxy"],
            re.compile(r"galax", re.I),
            exclude_title=not_galaxy,
            exclude_head=not_galaxy,
        ),
        CommonsSource(
            "commons_galaxy", "galaxy", 10**9,
            ["Category:Galaxies", "Category:Spiral galaxies", "Category:Elliptical galaxies", "Category:Hubble images of galaxies"],
            extra_bad=re.compile(
                r"milky way|galactic cent|night sky|landscape|panorama|quasar|blazar|\bagn\b|active galax|seyfert|radio galax|"
                r"supernova|nebula|planet|globular|open cluster|star cluster|dark matter|cosmic web", re.I),
        ),
        # ---- star
        SDSSSource("sdss_star_bright", "star", "sdss", int(target * 0.30), "STAR", "s.modelMag_r BETWEEN 11 AND 16", "fixed", 0.4, workers=12),
        SDSSSource("sdss_star_crowded", "star", "sdss_field", int(target * 0.30), "STAR", "s.modelMag_r BETWEEN 13 AND 18.5 AND ABS(p.b) < 22", "fixed", 0.7, workers=12),
        # Faint isolated stars at the SAME scale/brightness as the quasar source, so apparent brightness
        # (and frame scale) cannot separate star from quasar: only colour / environment can.
        SDSSSource("sdss_star_faint", "star", "sdss_faint", int(target * 0.40), "STAR", "s.modelMag_r BETWEEN 15 AND 19 AND ABS(p.b) > 25", "fixed", 0.4, workers=12),
        NASASource(
            "nasa_star",
            "star",
            10**9,
            ["star cluster", "globular cluster", "open cluster", "star field", "pleiades", "stellar cluster",
             "star trails", "binary star", "white dwarf", "red giant star", "wolf-rayet star", "dense star field"],
            re.compile(r"cluster|star field|star trails|pleiades|globular|white dwarf|binary star|red giant", re.I),
            exclude_title=not_nebula,
            exclude_head=re.compile(r"nebula|remnant|galax|computer|supercomput|supernova|planet|black hole", re.I),
        ),
        #quasar (visually the least separable class: point source + host)
        SDSSSource("sdss_quasar", "quasar", "sdss", int(target * 0.98), "QSO", "s.modelMag_r BETWEEN 15 AND 19", "fixed", 0.4, workers=18, modulus=3),
        NASASource(
            "nasa_quasar",
            "quasar",
            10**9,
            ["quasar", "active galactic nucleus", "blazar", "seyfert galaxy", "radio galaxy", "3c 273", "relativistic jet",
             "gravitational lens quasar", "agn", "bright galactic core"],
            re.compile(r"quasar|\bagn\b|active galactic|blazar|seyfert|radio galaxy|relativistic jet", re.I),
        ),
        # nebula (scarce: ~3-4k reachable unique images in total)
        NASASource(
            "nasa_nebula",
            "nebula",
            10**9,
            ["nebula", "planetary nebula", "supernova remnant", "emission nebula", "reflection nebula", "dark nebula",
             "hubble nebula", "webb nebula", "carina nebula", "orion nebula", "crab nebula", "pillars of creation",
             "helix nebula", "star forming region", "protoplanetary nebula", "hii region", "molecular cloud",
             "herbig-haro", "bubble nebula", "cat's eye nebula", "ring nebula", "eagle nebula", "stellar nursery",
             "cosmic cliffs", "cassiopeia a", "tarantula nebula", "dust cloud", "interstellar cloud"],
            re.compile(r"nebula|remnant|star.forming|stellar nurser|pillars|molecular cloud|herbig|hii|cosmic cliffs|dust cloud", re.I),
            exclude_title=not_nebula_title,
        ),
        CommonsSource(
            "commons_nebula",
            "nebula",
            10**9,
            ["Category:Nebulae", "Category:Planetary nebulae", "Category:Emission nebulae", "Category:Reflection nebulae",
             "Category:Dark nebulae", "Category:Supernova remnants", "Category:Hubble images of nebulae",
             "Category:James Webb Space Telescope images of nebulae"],
            depth=2,
            max_cats=120,
            extra_bad=re.compile(r"galax|quasar|blazar|\bagn\b|planet(?!ary nebula)|milky way|landscape|night sky|panorama|"
                                 r"globular|open cluster|star cluster|jupiter|saturn|\bmars\b|comet", re.I),
        ),
        # planet
        NASASource(
            "nasa_planet",
            "planet",
            int(target * 1.02),
            ["jupiter", "saturn", "venus", "mercury planet", "neptune", "uranus", "pluto", "titan", "europa", "enceladus",
             "ganymede", "callisto", "io volcano", "triton", "charon", "rhea", "dione", "tethys", "mimas", "iapetus",
             "phobos", "deimos", "ceres", "vesta", "mars global surveyor", "mars reconnaissance orbiter", "hirise mars",
             "viking orbiter mars", "mars orbiter", "mars crater", "mars polar", "lunar reconnaissance orbiter",
             "moon surface", "full moon", "cassini saturn", "cassini rings", "voyager jupiter", "voyager saturn",
             "juno jupiter", "new horizons", "messenger mercury", "magellan venus", "galileo jupiter", "great red spot",
             "saturn rings", "martian surface", "olympus mons", "valles marineris", "lunar crater"],
            PLANET_TERMS,
            exclude_head=PLANET_NOT,
            workers=18,
        ),
        ESASource("esa_hubble"),
    ]
