#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import MEnet

def command_preprocess(args):
    print(args)

def command_train(args):
    print(args)

def command_predict(args):
    from .predict import predict

    predict(args)
    # print(args)
    

def command_help(args):
    print(parser.parse_args([args.command, '--help']))


def main():
    parser = argparse.ArgumentParser(description='MEnet')
    subparsers = parser.add_subparsers()

    parser_preprocess = subparsers.add_parser('preprocess', help='see `preprocess -h`')
    parser_preprocess.add_argument('-i', '--input', metavar='input', help='input')
    parser_preprocess.add_argument('-r', '--reference', metavar='reference', help='reference')
    parser_preprocess.set_defaults(handler=command_train)

    parser_train = subparsers.add_parser('train', help='see `train -h`')
    parser_train.add_argument('-A', '--all', action='store_true', help='all files')
    parser_train.set_defaults(handler=command_train)

    parser_predict = subparsers.add_parser('predict', help='see `predict -h`')
    parser_predict.add_argument('-i', '--input', metavar='input', help='input', required=True)
    parser_predict.add_argument('-m', '--model', metavar='model', help='Traind model (pickle file).', required=True)
    parser_predict.add_argument('--input_type', help='input type. (default : auto)', default='auto',
                                choices=['auto', 'bismark', 'table', 'array'])
    parser_predict.add_argument('--input_filetype', help='input file type. (default : auto)', default='auto',
                                choices=['auto', 'csv', 'tsv'])
    parser_predict.add_argument('-o', '--output_dir', metavar='dir_out', help='output directory', default='out') 
    
    
    parser_predict.set_defaults(handler=command_predict)

    parser_help = subparsers.add_parser('help', help='see `help -h`')
    parser_help.add_argument('command', help='command name which help is shown')
    parser_help.set_defaults(handler=command_help)
   
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)

    else:
        parser.print_help()

if __name__ == "__main__":
    print('MEnet version : ', MEnet._version.__version__)
    main()