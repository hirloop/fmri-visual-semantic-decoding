# -*- coding: utf-8 -*-
"""List NSD S3 objects under given prefixes with sizes (anonymous access)."""
import sys
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BUCKET = "https://natural-scenes-dataset.s3.amazonaws.com"


def list_prefix(prefix):
    """Yield (key, size) for all objects under prefix."""
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=30)
        r.raise_for_status()
        import xml.etree.ElementTree as ET
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        root = ET.fromstring(r.text)
        for c in root.findall("s3:Contents", ns):
            key = c.find("s3:Key", ns).text
            size = int(c.find("s3:Size", ns).text)
            yield key, size
        truncated = root.find("s3:IsTruncated", ns).text == "true"
        token = root.find("s3:NextContinuationToken", ns).text if truncated else None
        if not token:
            return


if __name__ == "__main__":
    prefixes = sys.argv[1:]
    for p in prefixes:
        total, count = 0, 0
        files = []
        for key, size in list_prefix(p):
            total += size
            count += 1
            files.append((key, size))
        print(f"\n== {p}  ({count} files, {total/1e9:.2f} GB)")
        for key, size in files[:12]:
            print(f"   {size/1e6:10.1f} MB  {key}")
        if len(files) > 12:
            print(f"   ... and {len(files)-12} more")
