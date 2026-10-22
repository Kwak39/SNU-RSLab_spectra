"""Session-local inputs and portable downloads. No shared user-data cache."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def json_bytes(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return item.name
        raise TypeError(type(item).__name__)
    return json.dumps(value, ensure_ascii=False, indent=2, default=convert).encode('utf-8')


def csv_bytes(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def file_pairs(uploads):
    return [(Path(item.name.replace('\\', '/')).name, item.getvalue()) for item in uploads or []]


def signature(files):
    digest = hashlib.sha256()
    for name, payload in files:
        digest.update(name.encode('utf-8'))
        digest.update(b'\0')
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def content_hashes(files):
    return sorted({hashlib.sha256(payload).hexdigest() for _, payload in files})


def write_inputs(files, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seen, paths = set(), []
    for filename, payload in files:
        name = Path(filename.replace('\\', '/')).name
        if not name or name in ('.', '..') or name.casefold() in seen:
            raise ValueError('중복되거나 유효하지 않은 파일명입니다. 파일명을 구분하세요.')
        seen.add(name.casefold())
        path = directory / name
        path.write_bytes(payload)
        paths.append(path)
    return paths


def load_files(files, loader):
    # Bytes exist on disk only during parsing; datasets keep stable logical sources.
    with tempfile.TemporaryDirectory(prefix='rslab_parse_') as temp:
        paths = write_inputs(files, temp)
        data = loader(paths)
    if isinstance(data, list):
        for spectrum in data:
            spectrum.source = Path(spectrum.source).name
    else:
        data.sources = [Path(source).name for source in data.sources]
    return data


def broken_line(axis, values):
    """Avoid drawing a line across a large discontinuity in a masked axis."""
    delta = np.diff(axis)
    cuts = np.flatnonzero(delta > 5 * np.median(delta)) + 1 if len(delta) else []
    return np.insert(axis, cuts, np.nan), np.insert(values, cuts, np.nan)


def stage_files(session, stage):
    with tempfile.TemporaryDirectory(prefix='rslab_stage_') as temp:
        folder = session.save(stage, Path(temp))
        return [(p.name, p.read_bytes()) for p in sorted(folder.glob('*.csv'))
                if p.name != 'parameters.csv']
