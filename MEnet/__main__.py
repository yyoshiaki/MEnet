#!/usr/bin/env python
# coding: utf-8

import argparse
from .version import __version__

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

    parser_add = subparsers.add_parser('preprocess', help='see `preprocess -h`')
    parser_add.add_argument('-i', '--input', metavar='input', help='input')
    parser_add.add_argument('-r', '--reference', metavar='reference', help='reference')
    parser_add.set_defaults(handler=command_train)

    parser_add = subparsers.add_parser('train', help='see `train -h`')
    parser_add.add_argument('-A', '--all', action='store_true', help='all files')
    parser_add.set_defaults(handler=command_train)

    parser_commit = subparsers.add_parser('predict', help='see `predict -h`')
    parser_commit.add_argument('-i', '--input', metavar='input', help='input')
    parser_commit.add_argument('-r', '--reference', metavar='reference', help='reference')
    parser_commit.set_defaults(handler=command_predict)

    parser_help = subparsers.add_parser('help', help='see `help -h`')
    parser_help.add_argument('command', help='command name which help is shown')
    parser_help.set_defaults(handler=command_help)
   
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)

    else:
        parser.print_help()

if __name__ == "__main__":
    print('MEnet version : ', __version__)
    main()