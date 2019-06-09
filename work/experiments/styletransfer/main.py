#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser


def argparser():
    parser = ArgumentParser(description='Style Transfer')
    return parser


def main():
    args = argparser().parse_args()
    pass


if __name__ == '__main__':
    main()
