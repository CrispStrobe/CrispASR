#!/usr/bin/env python3
"""verify-remote-gguf.py — read a PUBLISHED GGUF's header back over HTTP and
report what the file actually is.

Why this exists (#437): a converter log line saying "wrote F16" is evidence
about the converter, not about the artifact a user will download. HuggingFace
has returned 200 on incomplete uploads, and an upload that half-lands leaves a
file that opens, has a plausible header, and is short. Neither the kernel log
nor `HfApi.upload_file` returning cleanly can tell those apart.

So this reads the real thing: HTTP range requests against the `resolve/main`
URL, parsing the GGUF header in place without downloading the tensor data.
From the header it reconstructs the size the file MUST have (last tensor
offset + its byte length, plus the aligned data start) and compares that to
the Content-Length the CDN reports. A truncated upload fails that check; so
does a file whose tensors are not the dtype it claims.

    python tools/verify-remote-gguf.py URL [URL ...]
    python tools/verify-remote-gguf.py --expect-dominant F16 --min-tensors 400 URL
    python tools/verify-remote-gguf.py --self-test      # controls, no network

Exit 0 only if every URL parsed, matched every --expect/--min constraint, and
the reconstructed size equals the served size. Exit 1 otherwise.

The point is that failure LOOKS DIFFERENT from success: every check prints
`ok`/`FAIL` with the two numbers it compared, and `--self-test` proves the
parser rejects a truncated file, a wrong-magic file, and a dtype mismatch
before you trust it on a real one.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import urllib.error
import urllib.request

# ── ggml type table ─────────────────────────────────────────────────────────
# (name, block_size, type_size) — enough to compute nbytes for any tensor.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 32),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
}

# GGUF metadata value types.
_GV_UINT8, _GV_INT8, _GV_UINT16, _GV_INT16 = 0, 1, 2, 3
_GV_UINT32, _GV_INT32, _GV_FLOAT32, _GV_BOOL = 4, 5, 6, 7
_GV_STRING, _GV_ARRAY, _GV_UINT64, _GV_INT64, _GV_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR = {
    _GV_UINT8: ("<B", 1), _GV_INT8: ("<b", 1),
    _GV_UINT16: ("<H", 2), _GV_INT16: ("<h", 2),
    _GV_UINT32: ("<I", 4), _GV_INT32: ("<i", 4),
    _GV_FLOAT32: ("<f", 4), _GV_BOOL: ("<?", 1),
    _GV_UINT64: ("<Q", 8), _GV_INT64: ("<q", 8),
    _GV_FLOAT64: ("<d", 8),
}


class Truncated(Exception):
    """The reader ran off the end of the available bytes."""


class BadGGUF(Exception):
    """The bytes are not a GGUF header we can parse."""


class RangeReader:
    """Sequential reader over a byte source fetched lazily in chunks.

    Backed either by an HTTP server that honours Range (the real case) or by a
    bytes object (the self-test). Both go through the same parser, so the
    self-test exercises the code that runs in production.
    """

    def __init__(self, fetch, total_size: int | None, chunk: int = 4 << 20):
        self._fetch = fetch          # fetch(start, end_inclusive) -> bytes
        self.total_size = total_size
        self._chunk = chunk
        self._buf = b""
        self._buf_start = 0
        self.pos = 0
        self.fetched = 0

    def _ensure(self, upto: int) -> None:
        if upto <= self._buf_start + len(self._buf):
            return
        start = self._buf_start + len(self._buf)
        end = max(upto, start + self._chunk) - 1
        if self.total_size is not None:
            end = min(end, self.total_size - 1)
        if end < start:
            raise Truncated(f"need byte {upto}, source has {self.total_size}")
        got = self._fetch(start, end)
        if not got:
            raise Truncated(f"empty read at {start}")
        self._buf += got
        self.fetched += len(got)
        if upto > self._buf_start + len(self._buf):
            raise Truncated(
                f"need {upto} bytes, only {self._buf_start + len(self._buf)} available"
            )

    def read(self, n: int) -> bytes:
        self._ensure(self.pos + n)
        off = self.pos - self._buf_start
        out = self._buf[off:off + n]
        if len(out) != n:
            raise Truncated(f"short read at {self.pos}: wanted {n}, got {len(out)}")
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > (64 << 20):
            raise BadGGUF(f"implausible string length {n} at offset {self.pos}")
        return self.read(n).decode("utf-8", errors="replace")


def _skip_value(r: RangeReader, vtype: int) -> object:
    if vtype in _SCALAR:
        fmt, size = _SCALAR[vtype]
        return struct.unpack(fmt, r.read(size))[0]
    if vtype == _GV_STRING:
        return r.string()
    if vtype == _GV_ARRAY:
        etype = r.u32()
        n = r.u64()
        if etype in _SCALAR:
            _, size = _SCALAR[etype]
            r.read(size * n)
            return f"<array {n}x type{etype}>"
        if etype == _GV_STRING:
            for _ in range(n):
                r.string()
            return f"<array {n}x string>"
        raise BadGGUF(f"unsupported array element type {etype}")
    raise BadGGUF(f"unsupported metadata value type {vtype}")


def parse_gguf_header(r: RangeReader) -> dict:
    """Parse magic + KV + tensor-info table. Returns a summary dict.

    Does NOT read tensor data — only the header, which is what a range request
    can cheaply fetch.
    """
    magic = r.read(4)
    if magic != b"GGUF":
        raise BadGGUF(f"bad magic {magic!r} (expected b'GGUF')")
    version = r.u32()
    if version not in (1, 2, 3):
        raise BadGGUF(f"unsupported GGUF version {version}")
    n_tensors = r.u64()
    n_kv = r.u64()
    if n_tensors > 1_000_000 or n_kv > 1_000_000:
        raise BadGGUF(f"implausible counts: {n_tensors} tensors, {n_kv} kv")

    kv: dict[str, object] = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.u32()
        kv[key] = _skip_value(r, vtype)

    alignment = kv.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0:
        alignment = 32

    dtype_hist: dict[str, int] = {}
    unknown_types: set[int] = set()
    end_of_data = 0
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        if n_dims > 4:
            raise BadGGUF(f"tensor {name!r} claims {n_dims} dims")
        dims = [r.u64() for _ in range(n_dims)]
        ttype = r.u32()
        offset = r.u64()
        info = GGML_TYPES.get(ttype)
        if info is None:
            unknown_types.add(ttype)
            tname, blk, tsz = f"TYPE{ttype}", 1, 0
        else:
            tname, blk, tsz = info
        dtype_hist[tname] = dtype_hist.get(tname, 0) + 1
        nelem = 1
        for d in dims:
            nelem *= d
        nbytes = (nelem // blk) * tsz if blk else 0
        end_of_data = max(end_of_data, offset + nbytes)
        tensors.append((name, dims, tname, offset, nbytes))

    data_start = (r.pos + alignment - 1) // alignment * alignment
    return {
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "alignment": alignment,
        "kv": kv,
        "dtype_hist": dtype_hist,
        "unknown_types": unknown_types,
        "header_bytes": r.pos,
        "data_start": data_start,
        "expected_size": data_start + end_of_data,
        "tensors": tensors,
    }


# ── HTTP plumbing ───────────────────────────────────────────────────────────

_UA = {"User-Agent": "crispasr-verify-remote-gguf/1"}


def http_size(url: str, timeout: float) -> tuple[int, str]:
    """Content-Length of the resolved object, following redirects."""
    req = urllib.request.Request(url, method="HEAD", headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cl = resp.headers.get("Content-Length")
        final = resp.geturl()
    if cl is None:
        raise BadGGUF("server did not report Content-Length")
    return int(cl), final


def make_http_fetch(url: str, timeout: float):
    def fetch(start: int, end: int) -> bytes:
        req = urllib.request.Request(
            url, headers={**_UA, "Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 206:
                # A 200 here means the server ignored Range — reading the whole
                # multi-GB file is never what we want, so refuse loudly.
                raise BadGGUF(
                    f"server ignored Range (HTTP {resp.status}); refusing full download"
                )
            return resp.read()
    return fetch


def verify_url(url: str, *, expect_dominant: str | None, min_tensors: int | None,
               forbid_dtypes: list[str], timeout: float, show: int) -> bool:
    print(f"\n{url}")
    try:
        size, final = http_size(url, timeout)
    except Exception as e:                       # noqa: BLE001
        print(f"  FAIL  HEAD: {e}")
        return False
    print(f"  served size .......... {size:,} bytes ({size / 2**30:.2f} GiB)")

    reader = RangeReader(make_http_fetch(url, timeout), size)
    try:
        h = parse_gguf_header(reader)
    except (Truncated, BadGGUF) as e:
        print(f"  FAIL  header parse: {e}")
        return False
    except Exception as e:                       # noqa: BLE001
        print(f"  FAIL  header fetch: {type(e).__name__}: {e}")
        return False

    print(f"  fetched .............. {reader.fetched:,} bytes of header")
    print(f"  gguf version ......... {h['version']}")
    print(f"  arch ................. {h['kv'].get('general.architecture', '?')}")
    print(f"  tensors .............. {h['n_tensors']}   kv: {h['n_kv']}")
    hist = ", ".join(f"{k}={v}" for k, v in sorted(
        h["dtype_hist"].items(), key=lambda kv: -kv[1]))
    print(f"  dtypes ............... {hist}")

    ok = True
    if h["unknown_types"]:
        print(f"  FAIL  unknown ggml type ids: {sorted(h['unknown_types'])}")
        ok = False

    # The load-bearing check: rebuild the file size the header implies and
    # compare it with what the CDN actually serves. A truncated upload that
    # still has a valid header fails HERE and nowhere else.
    exp = h["expected_size"]
    delta = size - exp
    if delta == 0:
        print(f"  ok    size matches header ... {exp:,} == {size:,}")
    elif 0 < delta < h["alignment"]:
        print(f"  ok    size matches header ... {exp:,} (+{delta} pad) == {size:,}")
    else:
        print(f"  FAIL  size mismatch ......... header implies {exp:,}, "
              f"server serves {size:,} (delta {delta:+,})")
        ok = False

    if min_tensors is not None:
        if h["n_tensors"] >= min_tensors:
            print(f"  ok    tensor count .......... {h['n_tensors']} >= {min_tensors}")
        else:
            print(f"  FAIL  tensor count .......... {h['n_tensors']} < {min_tensors}")
            ok = False

    if expect_dominant:
        want = expect_dominant.upper()
        top = max(h["dtype_hist"].items(), key=lambda kv: kv[1])[0] if h["dtype_hist"] else "-"
        if top == want:
            print(f"  ok    dominant dtype ........ {top} ({h['dtype_hist'][top]} tensors)")
        else:
            print(f"  FAIL  dominant dtype ........ {top}, expected {want}")
            ok = False

    for bad in forbid_dtypes:
        b = bad.upper()
        if b in h["dtype_hist"]:
            print(f"  FAIL  forbidden dtype ....... {b} present ({h['dtype_hist'][b]} tensors)")
            ok = False
        else:
            print(f"  ok    no {b} tensors")

    if show:
        print("  first tensors:")
        for name, dims, tname, off, nb in h["tensors"][:show]:
            print(f"    {name:44s} {str(dims):22s} {tname:6s} @{off} {nb}B")

    return ok


# ── self-test: prove the checks can FAIL before trusting them ───────────────

def _build_fake_gguf(*, magic=b"GGUF", ttype=1, n_extra_pad=0) -> bytes:
    """Minimal single-tensor GGUF with one KV. Used only by --self-test."""
    b = io.BytesIO()
    b.write(magic)
    b.write(struct.pack("<I", 3))          # version
    b.write(struct.pack("<Q", 1))          # n_tensors
    b.write(struct.pack("<Q", 1))          # n_kv
    key = b"general.architecture"
    b.write(struct.pack("<Q", len(key))); b.write(key)
    b.write(struct.pack("<I", _GV_STRING))
    val = b"selftest"
    b.write(struct.pack("<Q", len(val))); b.write(val)
    name = b"blk.0.weight"
    b.write(struct.pack("<Q", len(name))); b.write(name)
    b.write(struct.pack("<I", 2))          # n_dims
    b.write(struct.pack("<Q", 256)); b.write(struct.pack("<Q", 4))
    b.write(struct.pack("<I", ttype))
    b.write(struct.pack("<Q", 0))          # offset
    hdr = b.getvalue()
    start = (len(hdr) + 31) // 32 * 32
    _, blk, tsz = GGML_TYPES[ttype]
    nbytes = (256 * 4 // blk) * tsz
    return hdr + b"\0" * (start - len(hdr)) + b"\0" * (nbytes + n_extra_pad)


def _parse_bytes(data: bytes, served: int | None = None) -> dict:
    served = len(data) if served is None else served
    r = RangeReader(lambda s, e: data[s:e + 1], served)
    h = parse_gguf_header(r)
    h["served_size"] = served
    return h


def self_test() -> int:
    """Controls. Each must produce a DIFFERENT outcome, or the tool is blind."""
    fails = 0

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal fails
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
        if not cond:
            fails += 1

    print("self-test: positive control (well-formed F16)")
    good = _build_fake_gguf(ttype=1)
    h = _parse_bytes(good)
    check("parses", h["n_tensors"] == 1)
    check("dtype read as F16", h["dtype_hist"] == {"F16": 1}, str(h["dtype_hist"]))
    check("size matches", h["expected_size"] == len(good),
          f"{h['expected_size']} vs {len(good)}")

    print("self-test: discriminating control (same file, Q4_K tensor)")
    q = _build_fake_gguf(ttype=12)
    hq = _parse_bytes(q)
    check("dtype read as Q4_K, not F16", hq["dtype_hist"] == {"Q4_K": 1}, str(hq["dtype_hist"]))
    check("F16 and Q4_K differ in implied size", hq["expected_size"] != h["expected_size"])

    print("self-test: negative control (truncated upload, header intact)")
    trunc = good[: len(good) - 512]
    ht = _parse_bytes(trunc)
    check("truncation detected as size mismatch", ht["expected_size"] != len(trunc),
          f"header implies {ht['expected_size']}, served {len(trunc)}")

    print("self-test: negative control (header itself cut off)")
    try:
        _parse_bytes(good[:20])
        check("short header raises", False)
    except (Truncated, BadGGUF) as e:
        check("short header raises", True, type(e).__name__)

    print("self-test: negative control (wrong magic)")
    try:
        _parse_bytes(_build_fake_gguf(magic=b"GGML"))
        check("bad magic raises", False)
    except BadGGUF:
        check("bad magic raises", True)

    print("self-test: negative control (oversized file / trailing garbage)")
    fat = _build_fake_gguf(ttype=1, n_extra_pad=4096)
    hf = _parse_bytes(fat)
    check("extra trailing bytes detected", hf["expected_size"] != len(fat),
          f"header implies {hf['expected_size']}, served {len(fat)}")

    print(f"\nself-test: {'PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="HuggingFace resolve/main URLs")
    ap.add_argument("--expect-dominant", metavar="DTYPE",
                    help="require the most common tensor dtype to be this (e.g. F16)")
    ap.add_argument("--min-tensors", type=int, help="require at least this many tensors")
    ap.add_argument("--forbid-dtype", action="append", default=[], metavar="DTYPE",
                    help="fail if any tensor has this dtype (repeatable)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--show", type=int, default=0, help="print the first N tensor infos")
    ap.add_argument("--self-test", action="store_true",
                    help="run offline controls proving the checks can fail")
    args = ap.parse_args()

    if args.self_test:
        rc = self_test()
        if not args.urls:
            return rc
        if rc:
            print("refusing to trust remote results: self-test failed")
            return rc

    if not args.urls:
        ap.error("no URLs given (and --self-test not requested)")

    all_ok = True
    for url in args.urls:
        all_ok &= verify_url(url, expect_dominant=args.expect_dominant,
                             min_tensors=args.min_tensors,
                             forbid_dtypes=args.forbid_dtype,
                             timeout=args.timeout, show=args.show)
    print(f"\n{'ALL OK' if all_ok else 'FAILURES PRESENT'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
