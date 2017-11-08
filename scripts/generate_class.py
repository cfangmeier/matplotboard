#!/usr/bin/env python3


def generate_collection_class(obj_name, obj_attrs):
    src = []

    src += f'''\
struct {obj_name};

class {obj_name}Collection {{
  public:
    class iter {{
      public:
        iter(const {obj_name}Collection* collection, size_t idx)
          :collection(collection), idx(idx) {{ }}
        iter operator++() {{ ++idx; return *this; }}
        bool operator!=(const iter & other) {{ return idx != other.idx; }}
        const {obj_name} operator*() const;
      private:
        const {obj_name}Collection* collection;
        size_t idx;
    }};

'''.splitlines()

    for field in obj_attrs['fields']:
        name = field['name']
        type_ = field['type']
        src.append(f'    Value<vector<{type_}>>* val_{name};')

    src.append(f'\n    {obj_name}Collection() {{ }}\n')

    src.append('    void init(TrackingDataSet& tds){')
    for field in obj_attrs['fields']:
        name = field['name']
        type_ = field['type']
        prefix = obj_attrs['treename_prefix']+'_'
        src.append(f'        val_{name} = tds.track_branch_obj<vector<{type_}>>("{prefix}{name}");')
    src.append('    }\n')
    first_obj_name = list(obj_attrs['fields'])[0]['name']
    src.append(f'    size_t size() const {{ return val_{first_obj_name}->get_value().size();}}\n')
    src.append(f'    const {obj_name} operator[](size_t) const;')
    src.append('    iter begin() const { return iter(this, 0); }')
    src.append('    iter end() const { return iter(this, size()); }')
    src.append('};')

    src += f'''
struct {obj_name} {{
    const {obj_name}Collection* collection;
    const size_t idx;
    {obj_name}(const {obj_name}Collection* collection, const size_t idx)
      :collection(collection), idx(idx) {{ }}\n
'''.splitlines()

    for field in obj_attrs['fields']:
        name = field['name']
        type_ = field['type']
        src.append(f'    const {type_}& {name}() const {{return collection->val_{name}->get_value().at(idx);}}')
    src.append('};')

    src.append(f'''
const {obj_name} {obj_name}Collection::iter::operator*() const {{
    return {{collection, idx}};
}}
const {obj_name} {obj_name}Collection::operator[](size_t idx) const {{
    return {{this, idx}};
}}
''')
    return '\n'.join(src)


def generate_header(input_filename, output_filename):
    from datetime import datetime

    return f'''\
/** {output_filename} created on {datetime.now()} by generate_class.py
 * AVOID EDITING THIS FILE BY HAND!! Instead edit {input_filename} and re-run
 * generate_class.py
 */
#include "filval/filval.hpp"
#include "filval/root/filval.hpp"

#include<cmath>

#include "TrackingNtuple.h"

using namespace std;
using namespace fv;
using namespace fv::root;

typedef TreeDataSet<TrackingNtuple> TrackingDataSet;
'''


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('input_file', help='An input YAML file defining the objects to generate')

    args = parser.parse_args()
    classes = []
    with open(args.input_file) as fi:
        for obj, attrs in yaml.load(fi).items():
            classes.append(generate_collection_class(obj, attrs))
    output_filename = args.input_file.replace('.yaml', '.hpp')
    with open(output_filename, 'w') as fo:
        fo.write(generate_header(args.input_file, output_filename))
        for class_ in classes:
            fo.write(class_)
