"""Embed existing original-model quality helpers; no upload or submission."""
import base64
import gzip
from pathlib import Path
import sys
import subprocess

REVISION = '8d3b6aa393046b048643709555efc76719ee0bcc'


def main():
    root=Path(__file__).resolve().parents[3]
    names={'quality_cpu.py':'tools/kaggle/reuse-quality-cpu-20261112/quality_cpu.py',
           'chunked.py':'tools/ci-heavy/issue579_reuse/chunked.py'}
    bundle={name:base64.b64encode(gzip.compress(subprocess.check_output(['git','-C',str(root),'show',REVISION+':'+source]),mtime=0)).decode()
            for name,source in names.items()}
    text=Path(__file__).with_name('quality.py').read_text().replace('BUNDLE = None  # populated by prepare.py in the submitted copy','BUNDLE = '+repr(bundle))
    destination=Path(sys.argv[1])
    if destination.exists():raise ValueError('output already exists')
    destination.write_text(text)
    print('PREPARED_BYTES',len(text.encode()))


if __name__=='__main__':main()
