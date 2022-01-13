#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from MEnet import utils, _version
import os
import argparse

def tile(args):
    f_input = args.input
    tile_bp = args.bp

    print(f_input, tile_bp)
    if os.path.exists('{n}.tile{t}bp.csv'.format(n=f_input.split('.bis')[0].split('_bis')[0], t=tile_bp)):
        print('Input file has already been tiled. If you need to tile it again, delete the existing file ({}).'.format(
            '{n}.tile{t}bp.csv'.format(n=f_input.split('.bis')[0].split('_bis')[0], t=tile_bp)))
        return
    else:
        print('Tiling bismark cov...')
        df_input = utils.tile_bismark(f_input, tile_bp, args.bedtools)
        df_input.to_csv(
            '{n}.tile{t}bp.csv'.format(n=f_input.split('.bis')[0].split('_bis')[0], t=tile_bp))

        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.input = 'test/predict/Minion_STR1_Fr6.bis.cov.gz'
    args.bp = 500

    tile(args)
