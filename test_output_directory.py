import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import do


class OutputDirectoryTests(unittest.TestCase):
    def test_all_outputs_use_selected_directory(self):
        names = ['dump_success', 'dump_averages', 'chart_means',
                 'chart_what_do_you_think', 'chart_before_after']
        files = ['out-success.csv', 'out-medie.csv', 'out-chart-means.png',
                 'out-chart-what-do-you-think.png', 'out-chart-after-before.png']
        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(do, **{name: unittest.mock.DEFAULT for name in names}) as mocks:
                do.save_outputs(None, None, None, None, directory)
            for name, filename in zip(names, files):
                self.assertEqual(mocks[name].call_args.args[-1], os.path.join(directory, filename))
            self.assertFalse(mocks['chart_what_do_you_think'].call_args.kwargs['use_q'])

    def test_use_q_is_forwarded_to_chart(self):
        names = ['dump_success', 'dump_averages', 'chart_means',
                 'chart_what_do_you_think', 'chart_before_after']
        with patch.multiple(do, **{name: unittest.mock.DEFAULT for name in names}) as mocks:
            do.save_outputs(None, None, None, None, '.', use_q=True)
        self.assertTrue(mocks['chart_what_do_you_think'].call_args.kwargs['use_q'])

    def test_different_input_directories_are_rejected(self):
        script = Path(do.__file__).resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(script), '--output-in-input-dir',
                                     str(Path(directory) / 'one' / 'a_PRE_x.csv'),
                                     str(Path(directory) / 'two' / 'b_PRE_x.csv')],
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('all input files to be in the same directory', result.stderr)


if __name__ == '__main__':
    unittest.main()