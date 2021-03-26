# MEnet <img src="./img/MEnet.png" width="20%" align="right" />

MEnet is a neural-net based deconvolution method for methylation data. MEnet can be used only for humans so far.

## Installation

1. install pytorch. [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2. git clone this repo.
3. `python setup.py install`
4. If you use the `--bismark` option, you need to install `bedtools` locally.

## Usage

### prediction

```
usage: MEnet predict [-h] -i input -m model
                     [--input_type {auto,bismark,table,array}] [-o output_dir]
                     [--bedtools BEDTOOLS] [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -i input, --input input
                        input
  -m model, --model model
                        Traind model (pickle file).
  --input_type {auto,bismark,table,array}
                        input type. (default : auto)
  -o output_dir, --output_dir output_dir
                        output directory
  --bedtools BEDTOOLS   Full path to bedtools.
  --device DEVICE       device for pytorch. (ex. cpu, cuda)
```

## For development

```
conda activate menet_dev
```

install 

```
python setup.py develop
```

run 

```
python cli.py
```

ex.

```
python cli.py train --help
```

## Contact

Yoshiaki Yasumizu ([yyasumizu@ifrec.osaka-u.ac.jp](yyasumizu@ifrec.osaka-u.ac.jp))

## Licence

This software is freely available for academic users. Usage for commercial purposes is not allowed. Please refer to the LICENCE page.

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a>
