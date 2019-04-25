import sys

from main import main
from unittest.mock import patch

def test_SVMAmazonExperiment():
    with patch.object(sys, 'argv', [
        'main.py',
        '--directory',
        '..\dataset\\',
        '--experiment',
        'SVMAmazonExperiment',
    ]):
        main()
