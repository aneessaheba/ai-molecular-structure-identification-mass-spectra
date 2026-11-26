"""Download the competition files through Kaggle's official Python API client."""
import errno
import os
import tempfile
from urllib.parse import urlsplit
from pathlib import Path

COMPETITION = 'enveda-CASMI26-molecule-id-mass-spectra'
REQUIRED_COLUMNS = {
    'train.parquet': {'ingest_lib', 'normalized_smiles', 'inchikey14', 'molecular_formula',
        'ionization_mode', 'adduct', 'precursor_mz', 'precursor_error_ppm',
        'collision_energy_ev', 'ms2_mzs', 'ms2_normalized_intensities'},
    'test.parquet': {'molecule_id', 'adduct', 'ionization_mode', 'precursor_mz',
        'collision_energy_ev', 'ms2_mzs', 'ms2_normalized_intensities'},
    'sample_submission.csv': {'molecule_id', 'smiles'},
}


def valid_file(path):
    """Reject missing, truncated or wrongly named downloads without loading all spectra."""
    import pandas as pd
    import pyarrow.parquet as pq
    path = Path(path)
    try:
        if path.suffix == '.parquet':
            parquet = pq.ParquetFile(path)
            return parquet.metadata.num_rows > 0 and REQUIRED_COLUMNS[path.name] <= set(parquet.schema_arrow.names)
        frame = pd.read_csv(path, nrows=1)
        return len(frame) > 0 and REQUIRED_COLUMNS[path.name] <= set(frame.columns)
    except FileNotFoundError:
        return False
    except (ValueError, KeyError):
        return False


def storage_label(path):
    path = Path(path).resolve()
    drive = Path(os.environ.get('CASMI_DRIVE_MOUNT', '/content/drive')).resolve()
    if path.is_relative_to(drive):
        return 'Google Drive mount'
    if path.is_relative_to('/content'):
        return 'local Colab /content disk'
    return 'configured filesystem'


def failure_origin(error):
    """Report only structured diagnostics; exception messages/URLs may contain secrets."""
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ or error.__context__
    for error in chain:
        if isinstance(error, OSError) and error.errno in {
                errno.ENOSPC, errno.EIO, errno.EACCES, errno.EPERM, errno.EROFS,
                errno.ENOENT, errno.ENOTDIR, errno.EISDIR, errno.EMFILE, errno.ENFILE,
                errno.EFBIG, getattr(errno, 'EDQUOT', -1)}:
            code = errno.errorcode.get(error.errno, str(error.errno))
            return f'destination storage: {type(error).__name__}, errno={code}'
    for error in chain:
        response = getattr(error, 'response', None)
        status = getattr(response, 'status_code', None)
        if status is not None:
            host = urlsplit(getattr(response, 'url', '') or '').hostname or 'unknown host'
            source = 'Kaggle API' if host == 'kaggle.com' or host.endswith('.kaggle.com') else 'download server'
            return f'{source}: HTTP {status}, host={host}'
    for error in chain:
        if isinstance(error, (ConnectionError, TimeoutError)) or type(error).__name__ in {
                'ConnectionError', 'ConnectTimeout', 'ReadTimeout', 'Timeout', 'SSLError', 'ProxyError'}:
            return f'network transfer: {type(error).__name__}'
    return f'unknown source inside download client: {type(chain[0]).__name__}'


def storage_error(action, path, error):
    detail = f'ERROR [storage] Cannot {action}: {path} ({storage_label(path)}; {failure_origin(error)})'
    print(detail, flush=True)
    return RuntimeError(detail)


def checked_file(path):
    try:
        return valid_file(path)
    except OSError as error:
        raise storage_error('read/validate file', path, error) from None


def ensure_competition_data(raw_dir, token_provider=None):
    """Reuse validated files; download missing files directly into Drive staging.

    token_provider is called only when a download needs a token and no token is
    already in the environment. Credentials are never written to disk or returned.
    """
    raw_dir = Path(raw_dir).resolve()
    print(f'Data destination: {raw_dir} ({storage_label(raw_dir)})', flush=True)
    print('Transfer direction: Kaggle -> destination filesystem; no browser upload to Colab.', flush=True)
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=raw_dir, prefix='.write-check-') as probe:
            probe.write(b'write check')
            probe.flush()
    except OSError as error:
        raise storage_error('create/write destination', raw_dir, error) from None
    missing = [name for name in REQUIRED_COLUMNS if not checked_file(raw_dir / name)]
    if not missing:
        print('Reusing all three validated competition files in Drive.')
        return {name: raw_dir / name for name in REQUIRED_COLUMNS}
    import kagglehub
    previous_token = os.environ.get('KAGGLE_API_TOKEN')
    token = previous_token or (token_provider() if token_provider else None)
    if not token or not token.strip():
        raise RuntimeError('Provide a Kaggle API token in the environment-values cell or hidden prompt.')
    os.environ['KAGGLE_API_TOKEN'] = token.strip()
    try:
        for name in missing:
            # Only publish a file at its canonical path after successful validation.
            staging = raw_dir / '.downloads' / name.replace('.', '_')
            try:
                staging.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise storage_error('create staging directory', staging, error) from None
            print(f'Downloading {name} from Kaggle into {staging}...', flush=True)
            try:
                downloaded = Path(kagglehub.competition_download(COMPETITION, path=name,
                    output_dir=str(staging), force_download=True)).resolve()
            except Exception as error:
                detail = failure_origin(error)
                message = f'Kaggle download failed [{detail}]. File: {name}; destination: {staging}.'
                print(f'ERROR [transfer] {message}', flush=True)
                raise RuntimeError(message + ' Check access/rules for API errors or Drive space/permissions '
                                   'for storage errors; rerun this cell.') from None
            if not downloaded.is_relative_to(staging):
                raise RuntimeError('Kaggle returned a path outside Drive staging; upgrade kagglehub and rerun.')
            print(f'Transfer finished: {name}; validating downloaded file...', flush=True)
            if not checked_file(downloaded):
                print(f'ERROR [validation] Invalid or incomplete {name} at {downloaded}.', flush=True)
                raise RuntimeError(f'Invalid or incomplete {name}; rerun to download it again.')
            try:
                downloaded.replace(raw_dir / name)
            except OSError as error:
                raise storage_error('finalize downloaded file', raw_dir / name, error) from None
            print(f'Ready: {raw_dir / name}', flush=True)
    finally:
        if previous_token is None:
            os.environ.pop('KAGGLE_API_TOKEN', None)
        else:
            os.environ['KAGGLE_API_TOKEN'] = previous_token
        del token
    return {name: raw_dir / name for name in REQUIRED_COLUMNS}
