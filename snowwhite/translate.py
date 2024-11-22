#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', metavar='<file>', required=True)
parser.add_argument('--output', '-o', metavar='<file>', required=True)
args = parser.parse_args()

with open(args.input, "r", encoding="utf-8") as input_file, \
     open(args.output, "w", encoding="utf-8") as output_file:
    for line in input_file:
        transformed_line = line.strip().replace("▁", " ")
        output_file.write(transformed_line + "\n")
