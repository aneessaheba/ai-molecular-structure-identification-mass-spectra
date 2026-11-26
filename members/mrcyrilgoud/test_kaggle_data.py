"""Offline tests for authenticated Kaggle API downloads into Drive paths."""
import errno
import io
from contextlib import redirect_stdout
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from members.mrcyrilgoud.kaggle_data import (
    COMPETITION, REQUIRED_COLUMNS, ensure_competition_data, valid_file,
)


def write_fixture(path):
    frame = pd.DataFrame({column: ['fixture'] for column in REQUIRED_COLUMNS[path.name]})
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.parquet':
        frame.to_parquet(path)
    else:
        frame.to_csv(path, index=False)
    return str(path)


class KaggleDownloadTests(unittest.TestCase):
    def test_downloads_all_files_into_drive_and_reuses_them(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            raw = (Path(tmp) / 'Drive/repo/data/raw').resolve()
            def download(handle, *, path, output_dir, force_download):
                self.assertEqual(handle, COMPETITION)
                self.assertTrue(force_download)
                self.assertTrue(Path(output_dir).is_relative_to(raw))
                self.assertEqual(os.environ['KAGGLE_API_TOKEN'], 'test-secret')
                return write_fixture(Path(output_dir) / path)
            api = Mock(side_effect=download)
            with patch.dict('sys.modules', {'kagglehub': types.SimpleNamespace(competition_download=api)}):
                result = ensure_competition_data(raw, token_provider=lambda: 'test-secret')
                self.assertEqual(api.call_count, 3)
                self.assertEqual(set(result), set(REQUIRED_COLUMNS))
                self.assertTrue(all(valid_file(p) for p in result.values()))
                self.assertNotIn('KAGGLE_API_TOKEN', os.environ)
                api.reset_mock()
                ensure_competition_data(raw, token_provider=Mock(side_effect=AssertionError('Unexpected prompt')))
                api.assert_not_called()

    def test_corrupt_file_is_replaced_without_redownloading_valid_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'KAGGLE_API_TOKEN': 'existing-secret'}, clear=True):
            raw = Path(tmp)
            for name in REQUIRED_COLUMNS:
                write_fixture(raw / name)
            (raw / 'train.parquet').write_bytes(b'interrupted download')
            api = Mock(side_effect=lambda handle, **kw: write_fixture(Path(kw['output_dir']) / kw['path']))
            with patch.dict('sys.modules', {'kagglehub': types.SimpleNamespace(competition_download=api)}):
                ensure_competition_data(raw)
            self.assertEqual(api.call_count, 1)
            self.assertEqual(api.call_args.kwargs['path'], 'train.parquet')
            self.assertTrue(valid_file(raw / 'train.parquet'))
            self.assertEqual(os.environ['KAGGLE_API_TOKEN'], 'existing-secret')

    def test_api_failure_restores_environment_and_does_not_publish_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            api = Mock(side_effect=RuntimeError('Do not display test-secret'))
            with patch.dict('sys.modules', {'kagglehub': types.SimpleNamespace(competition_download=api)}):
                with self.assertRaisesRegex(RuntimeError, 'Kaggle download failed') as error:
                    ensure_competition_data(tmp, token_provider=lambda: 'test-secret')
            self.assertNotIn('test-secret', str(error.exception))
            self.assertNotIn('KAGGLE_API_TOKEN', os.environ)
            self.assertFalse((Path(tmp) / 'train.parquet').exists())

    def test_invalid_download_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            def invalid(handle, **kw):
                path = Path(kw['output_dir']) / kw['path']
                path.write_bytes(b'not parquet')
                return str(path)
            with patch.dict('sys.modules', {'kagglehub': types.SimpleNamespace(competition_download=invalid)}):
                with self.assertRaisesRegex(RuntimeError, 'Invalid or incomplete'):
                    ensure_competition_data(tmp, token_provider=lambda: 'test-secret')
            self.assertFalse((Path(tmp) / 'train.parquet').exists())
            self.assertNotIn('KAGGLE_API_TOKEN', os.environ)

    def test_missing_token_fails_without_network_request(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            api = Mock()
            with patch.dict('sys.modules', {'kagglehub': types.SimpleNamespace(competition_download=api)}):
                with self.assertRaisesRegex(RuntimeError, 'Provide a Kaggle API token'):
                    ensure_competition_data(tmp, token_provider=lambda: '')
            api.assert_not_called()

    def test_http_failure_logs_remote_source_without_token_or_signed_url(self):
        for host, expected in [('api.kaggle.com', 'Kaggle API'),
                               ('storage.googleapis.com', 'download server')]:
            with self.subTest(host=host), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
                error = RuntimeError('test-secret must not appear')
                error.response = types.SimpleNamespace(status_code=403,
                    url=f'https://{host}/file?token=test-secret')
                api = Mock(side_effect=error)
                output = io.StringIO()
                with patch.dict('sys.modules', {'kagglehub': types.SimpleNamespace(competition_download=api)}):
                    with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, expected + ': HTTP 403'):
                        ensure_competition_data(tmp, token_provider=lambda: 'test-secret')
                self.assertIn('ERROR [transfer]', output.getvalue())
                self.assertNotIn('test-secret', output.getvalue())
                self.assertNotIn('?token=', output.getvalue())

    def test_disk_full_during_sdk_download_logs_storage_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            outer = RuntimeError('opaque SDK failure')
            outer.__cause__ = OSError(errno.ENOSPC, 'No space')
            api = Mock(side_effect=outer)
            output = io.StringIO()
            with patch.dict('sys.modules', {'kagglehub': types.SimpleNamespace(competition_download=api)}):
                with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, 'destination storage.*ENOSPC'):
                    ensure_competition_data(tmp, token_provider=lambda: 'test-secret')
            self.assertIn('destination storage', output.getvalue())
            self.assertNotIn('KAGGLE_API_TOKEN', os.environ)

    def test_destination_permission_failure_is_logged_before_api_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with patch.object(Path, 'mkdir', side_effect=PermissionError(errno.EACCES, 'Denied')):
                with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, r'ERROR \[storage\].*EACCES'):
                    ensure_competition_data(tmp)
            self.assertIn('create/write destination', output.getvalue())

    def test_finalize_failure_logs_storage_stage(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            api = Mock(side_effect=lambda handle, **kw: write_fixture(Path(kw['output_dir']) / kw['path']))
            output = io.StringIO()
            with patch.dict('sys.modules', {'kagglehub': types.SimpleNamespace(competition_download=api)}):
                with patch.object(Path, 'replace', side_effect=OSError(errno.ENOSPC, 'Full')):
                    with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, 'finalize downloaded file'):
                        ensure_competition_data(tmp, token_provider=lambda: 'test-secret')
            self.assertIn('Transfer finished', output.getvalue())
            self.assertIn('ERROR [storage]', output.getvalue())

    def test_connection_reset_is_network_not_storage(self):
        from members.mrcyrilgoud.kaggle_data import failure_origin, storage_label
        self.assertIn('network transfer', failure_origin(ConnectionResetError(errno.ECONNRESET, 'Reset')))
        with patch.dict(os.environ, {'CASMI_DRIVE_MOUNT': '/content/drive'}):
            self.assertEqual(storage_label('/content/drive/MyDrive/data'), 'Google Drive mount')
            self.assertEqual(storage_label('/content/data'), 'local Colab /content disk')



if __name__ == '__main__':
    unittest.main()
