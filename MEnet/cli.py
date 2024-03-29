#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import MEnet


def command_tile(args):
    from .tile import tile

    print(args)
    tile(args)

def command_train(args):
    from .train import train

    train(args)


def command_predict(args):
    from .predict import predict

    predict(args)
    # print(args)


def command_help(args):
    print(parser.parse_args([args.command, '--help']))


def main():
    print_logo()
    print('MEnet version : ', MEnet._version.__version__)

    parser = argparse.ArgumentParser(description='MEnet : Neural net based cell deconvolution utilizing DNA methylation.\n\n'
    'Developer : Yoshiaki Yasumizu, Yuto Umedu (Osaka University)\n'
    'Licence : CC BY-NC 4.0')
    subparsers = parser.add_subparsers()

    # predict
    parser_predict = subparsers.add_parser('predict', help='see `predict -h`')
    parser_predict.add_argument(
        '-i', '--input', metavar='input', help='input', required=True)
    parser_predict.add_argument(
        '-m', '--model', metavar='model', help='Traind model (pickle file).', required=True)
    parser_predict.add_argument('--input_type', help='input type. (default : auto)', default='auto',
                                choices=['auto', 'bismark', 'table', 'array'])
    parser_predict.add_argument(
        '-o', '--output_dir', metavar='output_dir', help='output directory', default='out')
    parser_predict.add_argument(
        '--output_prefix', metavar='output_prefix', help='prefix of output files', default="")
    parser_predict.add_argument(
        '--bedtools', type=str, default='bedtools', help='Full path to bedtools.')
    parser_predict.add_argument(
        '--plotoff', action='store_true', default='plotoff', help='Do not generate plots.')
    # parser_predict.add_argument('--device', type=str, default=None, help='device for pytorch. (ex. cpu, cuda)')

    parser_predict.set_defaults(handler=command_predict)

    # tile 
    parser_tile = subparsers.add_parser(
        'tile', help='see `tile -h`', description="Tile bismark.cov beforehand. "
                "This function is useful for making predictions using various models for a single Bismark file.")
    parser_tile.add_argument(
        'input', action='store', help='input bismark file.')
    parser_tile.add_argument(
        '-b', '--bp', metavar='bp', type=int, help='tile size (bp)', required=True)
    parser_tile.add_argument(
        '--bedtools', type=str, default='bedtools', help='Full path to bedtools.')
    parser_tile.set_defaults(handler=command_tile)

    # train
    parser_train = subparsers.add_parser('train', help='see `train -h`')
    parser_train.add_argument(
        'input_yaml', action='store', help='input yaml file.')
    parser_train.add_argument(
        '--device', type=str, default=None, help='device for pytorch. (ex. cpu, cuda)')
    parser_train.set_defaults(handler=command_train)

    parser_help = subparsers.add_parser('help', help='see `help`')
    parser_help.add_argument(
        'command', help='command name which help is shown')
    parser_help.set_defaults(handler=command_help)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)

    else:
        parser.print_help()


def print_logo():
    print('''
                                                  
             ____                    __           
     /'\_/`\/\  _`\                 /\ \__        
    /\      \ \ \L\_\    ___      __\ \ ,_\\\\      
    \ \ \__\ \ \  _\L  /' _ `\  /'__`\ \ \/       
     \ \ \_/\ \ \ \L\ \/\ \/\ \/\  __/\ \ \_      
      \ \_\\\\ \_\ \____/\ \_\ \_\ \____\\\\ \__\\\\    
       \/_/ \/_/\/___/  \/_/\/_/\/____/ \/__/     
                                                  ''')


if __name__ == "__main__":
    main()
