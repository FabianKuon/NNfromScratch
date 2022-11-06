from graphviz import Digraph, DOT_BINARY


class Value:
    """
    Class representation for mathematical operations in a neural net
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')


def trace(root: float):
    """
    builds a set of all nodes and edges in a graph.
    """
    nodes, edges = set(), set()

    def build(value):
        if value not in nodes:
            nodes.add(value)
            for child in value._prev:
                edges.add((child, value))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root: float) -> DOT_BINARY:
    """
    builds the dot object containing a directonal graph.
    """
    dot = Digraph(format='svg', graph_attr={
                  'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{ data %.4f }" %
                 (node.data, ), shape='record')
        if node._op:
            # if this value is a result of some operatoin, create an op node for it
            dot.node(name=uid + node._op, label=node._op)
            # and connect this node to it
            dot.edge(uid + node._op, uid)

    for node1, node2 in edges:
        dot.edge(str(id(node1)), str(id(node2)) + node2._op)

    return dot


if __name__ == "__main__":
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a*b + c
    draw_dot(d).render(directory='doctest-output', view=True)
