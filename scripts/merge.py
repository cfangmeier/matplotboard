#!env/bin/python
import argparse
import re
import os
import ROOT


def merge_stl_obj(obj_key, output_file, input1, input_rest, merge_func=None):
    """ Merges STL objects and saves the result into the output file, user
        must supply the merging function.
    """
    obj = input1.Get(obj_key)
    type_name_raw = str(type(obj))
    try:
        type_name = re.findall("<class 'ROOT.([^']+)'>", type_name_raw)[0]
    except IndexError:
        raise ValueError(f"Couldn't extract stl type name from {type_name_raw}")
    if merge_func is not None:
        for input_file in input_rest:
            obj_ = input_file.Get(obj_key)
            merge_func(obj, obj_)
    output_file.WriteObjectAny(obj, type_name, obj_key)


def merge_obj(obj_key, output_file, input1, input_rest):
    obj = input1.Get(obj_key)
    print('='*80)
    print(f'Merging object {obj_key} of type {type(obj)}')
    if isinstance(obj, ROOT.TH1):
        obj.SetDirectory(output_file)  # detach from input file
        for input_file in input_rest:
            obj_ = input_file.Get(obj_key)
            obj.Add(obj_)
        obj.Write()
    else:
        print(f"I don't know how to merge object of type{type(obj)}, but "
              "you can add a case in merge_obj to handle it!")


def merge_files(input_filenames, output_filename, preserve=False):
    print(f"Merging files {', '.join(input_filenames)} into {output_filename}")

    input1, *input_rest = [ROOT.TFile.Open(input_file, "READ") for input_file in input_filenames]
    output_file = ROOT.TFile.Open(output_filename, "RECREATE")
    output_file.cd()

    obj_keys = [k.GetName() for k in input1.GetListOfKeys()]
    for obj_key in obj_keys:
        if obj_key in {"_value_lookup", "_function_impl_lookup"}:
            merge_stl_obj(obj_key, output_file, input1, [])
        else:
            merge_obj(obj_key, output_file, input1, input_rest)

    for file_ in [input1, *input_rest, output_file]:
        file_.Close()
    print(f"Merge finished! Results have been saved into {output_filename}")
    if preserve:
        print("Preseve specified, leaving input files intact")
    else:
        print("Removing input files...", end='', flush=True)
        for filename in input_filenames:
            os.remove(filename)
        print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add = parser.add_argument

    add('output_file')
    add('input_files', nargs='+')
    add('--preserve', '-p', action='store_true')

    args = parser.parse_args()
    merge_files(args.input_files, args.output_file, args.preserve)
