import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import do


class QValueTests(unittest.TestCase):
    def setUp(self):
        self.first = pd.DataFrame({name: [0, 0] for name in ['1->TU', '2->TU', '3->TU', '4->TU']})
        self.second = pd.DataFrame({
            '1->TU': [1, 1], '2->TU': [1, 1], '3->TU': [1, 1], '4->TU': [0, 0],
        })

    def test_empty_and_all_nan_families(self):
        self.assertEqual(do.calculate_qvalues({}), {})
        self.assertTrue(math.isnan(do.calculate_qvalues({'a': float('nan')})['a']))

    def test_failed_test_remains_in_family(self):
        results = [SimpleNamespace(pvalue=0.01, zstatistic=1, statistic=1), ValueError(),
                   SimpleNamespace(pvalue=0.04, zstatistic=1, statistic=1)]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(do, 'wilcoxon', side_effect=results), \
                patch.object(do, 'cliffs_delta', return_value=(0.5, 'medium')):
            filename = Path(directory) / 'averages.csv'
            do.dump_averages(self.first, self.second, filename)
            with filename.open(newline='') as csvfile:
                values = [float(row['FDR BH q-value']) for row in csv.DictReader(csvfile)]
        self.assertAlmostEqual(values[0], 0.04)
        self.assertEqual(values[1], 1.0)
        self.assertAlmostEqual(values[2], 0.08)

    def test_csv_corrects_family_and_preserves_nan(self):
        results = [SimpleNamespace(pvalue=p, zstatistic=1, statistic=1) for p in (0.01, 0.04, float('nan'))]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(do, 'wilcoxon', side_effect=results), \
                patch.object(do, 'cliffs_delta', return_value=(0.5, 'medium')):
            filename = Path(directory) / 'averages.csv'
            do.dump_averages(self.first, self.second, filename)
            with filename.open(newline='') as csvfile:
                rows = list(csv.DictReader(csvfile))
        values = [float(row['FDR BH q-value']) for row in rows]
        self.assertAlmostEqual(values[0], 0.03)
        self.assertAlmostEqual(values[1], 0.06)
        self.assertTrue(math.isnan(values[2]))
        self.assertEqual(values[3], 1.0)

    def test_chart_uses_corrected_significance(self):
        results = [SimpleNamespace(pvalue=p, zstatistic=1, statistic=1) for p in (0.01, 0.04, float('nan'))]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(do, 'wilcoxon', side_effect=results), \
                patch.object(do, 'cliffs_delta', return_value=(0.5, 'medium')):
            do.chart_what_do_you_think(self.first, self.second, 'TU', Path(directory) / 'chart.png')
            stars = plt.gcf().axes[0].lines[2].get_ydata()
            self.assertEqual(np.ma.count(stars), 1)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()