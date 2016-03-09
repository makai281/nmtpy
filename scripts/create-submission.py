#!/usr/bin/env python

import argparse

INSTITUTE = "LIUM"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--task-name"     , type=str, required=True)
    parser.add_argument("-m", "--method-name"   , type=str, required=True)
    parser.add_argument("-T", "--type"          , type=str, required=True, help="(C)onstrained or (U)nconstrained")
    
