import pydotplus.graphviz as pdp
import sys
import re


def parse(str_in, alias=None):
    str_in = "("+str_in+")"

    functions = []
    ends = {}
    nests = {}
    names = {}
    styles = {}
    parens = []
    name = ""
    name_start = 0
    name_end = 0
    for i, char in enumerate(str_in):
        if char == "(":
            nests[name_start] = []
            if parens:
                nests[parens[-1]].append(name_start)
            names[name_start] = name  # save function name
            styles[name_start] = {"shape": "ellipse"}
            name = ""
            parens.append(name_start)
        elif char == ")":
            if name:
                ends[name_start] = name_end
                names[name_start] = name
                styles[name_start] = {"shape": "rectangle"}
                nests[parens[-1]].append(name_start)
                name = ""
            ends[parens.pop()] = i
        elif char in ",:":
            if name:
                ends[name_start] = name_end
                names[name_start] = name
                if char == ",":
                    styles[name_start] = {"shape": "rectangle"}
                else:
                    styles[name_start] = {"shape": "invhouse"}
                    functions.append(name)
                nests[parens[-1]].append(name_start)
                name = ""
        else:
            if not name:
                name_start = i
            name += char
            name_end = i

    # clean up duplicate sub-trees
    text = {}
    for start, end in ends.items():
        s = str_in[start:end+1]
        if s in text:
            dup_id = text[s]
            names.pop(start)
            if start in nests:
                nests.pop(start)
            for l in nests.values():
                for i in range(len(l)):
                    if l[i] == start:
                        l[i] = dup_id
        else:
            text[s] = start

    names.pop(0)
    nests.pop(0)

    g = pdp.Dot()

    for id_, name in names.items():
        g.add_node(pdp.Node(str(id_), label=name, **styles[id_]))
    for group_id, children in nests.items():
        for child_id in children:
            g.add_edge(pdp.Edge(str(group_id), str(child_id)))
    if alias:
        g.add_node(pdp.Node(alias, shape="plain", pos="0,0!"))
    return g, functions


if __name__ == '__main__':
    aliases = {}
    ali_re = re.compile(r"ALIAS::\"([^\"]*)\" referring to \"([^\"]*)\"")

    with open(sys.argv[1]) as f:
        for line in f.readlines():
            res = ali_re.findall(line)
            if res:
                aliases[res[0][1]] = res[0][0]
                continue
    for name, alias in aliases.items():
        graph, _ = parse(name, alias)
        fname = "val_graph_{}.gif".format(alias)
        with open(fname, "wb") as f:
            try:
                f.write(graph.create_gif())
            except Exception as e:
                print(e)
                print(graph.to_string())
