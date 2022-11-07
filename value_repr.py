"""Class representing a neural network implementation without a ML library"""
import math
from graphviz import Digraph, DOT_BINARY


class Value:
    """
    Class representation for mathematical operations in a neural net
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # enables addition of Value instance with e.g. integers
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __subtr__(self, other):
        return self * (-other)

    def __mul__(self, other):
        # enables multiplication of Value instance with e.g. integers
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)), "only supporting int and float for power as of now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad = other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        """Function which allows for 2*a=2.__mul__(a) which would otherwise not be possible"""
        return self * other

    def __truediv__(self, other):
        """
        Function for division operation of Value instances
        """
        return self * other**-1

    def tanh(self):
        """
        Tanh evaluation of Values instance
        """
        tanh = (math.exp(2*self.data) - 1)/(math.exp(2*self.data) + 1)
        out = Value(tanh, (self, ), 'tanh')

        def _backward():
            self.grad += (1.0 - tanh**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """
        exp function for Values instance
        """
        out = Value(math.exp(self.data, (self,), 'exp'))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """
        Backward propagation through neural net
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


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
        dot.node(
            name=uid,
            label=f"{{ {node.label} | data {node.data:.4f} | grad {node.grad:.4f} }}",
            shape='record')

        if node._op:
            # if this value is a result of some operatoin, create an op node for it
            dot.node(name=uid + node._op, label=node._op)
            # and connect this node to it
            dot.edge(uid + node._op, uid)

    for node1, node2 in edges:
        dot.edge(str(id(node1)), str(id(node2)) + node2._op)

    return dot


if __name__ == "__main__":
    # inputs x1, x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights w1, w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bais of the neuron
    b = Value(6.8813735870195432, label='b')
    # x1*wa1 + x2*w2 + b
    x1w1 = x1*w1
    x1w1.label = 'x1+w1'
    x2w2 = x2*w2
    x1w1.label = 'x2+w2'
    x1w1x2w2 = x1w1+x2w2
    x1w1x2w2.label = 'x1*w1 + x2*w2'
    n = x1w1x2w2 + b
    n.label = 'n'
    o = n.tanh()
    o.label = 'o'
    draw_dot(o).render(directory='doctest-output')
    o.backward()
