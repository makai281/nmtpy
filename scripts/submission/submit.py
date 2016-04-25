#!/usr/bin/env python

# FORMAT
# INSTITUTION-NAME_TASK-NAME_METHOD-NAME_TYPE
# INSTITUTION-NAME: Short identifier, e.g. LIUM
# TASK-NAME: 1: translation, 2: description, 3: both
# METHOD-NAME: NeuralTranslation, Moses, etc.
# TYPE: C: Constrained, U: Unconstrained

# SHEF_2_Moses_C: SHEF, description task, Moses system, Constrained

import argparse

INST = "LIUM"

def dump_results(out_file, images, hyps, method_name, task_name, _type):
    with open(out_file, 'w') as f:
        for img, hyp in zip(images, hyps):
            f.write("%s\t%s\t%s\t%s\t%s\n" % (method_name, img, hyp, task_name, _type))

def get_file_name(task_name, method_name, _type):
    s = "%s_%s_%s_%s" % (
            INST, task_name, method_name, _type)
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='submit')
    parser.add_argument('-T', '--task', type=str, required=True, help='Task 1, 2 or both(3)')
    parser.add_argument('-t', '--type', type=str, default='C', help='C(onstrained) or U(nconstrained)')
    parser.add_argument('-i', '--imgfile', type=str, default='test_images.txt', help='Test images list')
    parser.add_argument('-H', '--hypfile', type=str, required=True, help='Hypothesis file')
    parser.add_argument('-m', '--method', type=str, required=True, help='Method name')
    args = parser.parse_args()

    print 'Institution name: %s' % INST
    print 'Task name: %s' % args.task
    print 'Method name: %s' % args.method
    print 'Type: %s' % args.type
    out_file = get_file_name(args.task, args.method, args.type)

    # Open image files
    with open(args.imgfile) as fi:
        images = fi.read().strip().split("\n")

    with open(args.hypfile) as fh:
        hyps = fh.read().strip().split("\n")

    assert len(images) == len(hyps)

    dump_results(out_file, images, hyps, args.method, args.task, args.type)
