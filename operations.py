import math
class Operation(object):
    def partial_a(self):
        """ Computes the partial derivative of this operation with respect to a: d(op)/da"""
        raise NotImplementedError

    def partial_b(self):
        """ Computes the partial derivative of this operation with respect to b: d(op)/db"""
        raise NotImplementedError

    def __call__(self, a, b):
        """ Computes the forward pass of this operation: op(a, b) -> output"""
        raise NotImplementedError

    def backprop(self, grad):
        """ Calls .backprop for self.a and self.b, passing to it dF/da and
            dF/db, respectively, where F represents the terminal `Number`-instance node 
            in the computational graph, which originally invoked `F.backprop().             
            """
        self.a.backprop(self.partial_a() * grad)  # backprop: dF/d(op)*d(op)/da -> dF/da
        self.b.backprop(self.partial_b() * grad)

    def null_gradients(self):
        for attr in self.__dict__:
            var = getattr(self, attr)
            if hasattr(var, 'null_gradients'):
                var.null_gradients()

class Add(Operation):
    def __repr__(self): return "+"

    def __call__(self, a, b):
        self.a = a
        self.b = b
        return a.data + b.data
    
    def partial_a(self):
        """ Returns d(a + b)/da """
        return 1
    
    def partial_b(self):
        """ Returns d(a + b)/db """
        return 1


class Multiply(Operation):
    def __repr__(self): return "*"

    def __call__(self, a, b):
        self.a = a
        self.b = b
        return a.data * b.data
    
    def partial_a(self):
        """ Returns d(a * b)/da as int or float"""
        return self.b.data
    
    def partial_b(self):
        """ Returns d(a * b)/db as int or float"""
        return self.a.data


class Subtract(Operation):
    def __repr__(self):
        return "-"

    def __call__(self, a, b):
        self.a = a
        self.b = b
        return a.data - b.data
    
    def partial_a(self):
        return 1
    
    def partial_b(self):
        return -1
    
class Divide(Operation):
    def __repr__(self): return "/"

    def __call__(self, a, b):
        self.a = a
        self.b = b
        return a.data / b.data
    
    def partial_a(self):
        return 1/(self.b.data)
    
    def partial_b(self):
        return - self.a.data / (self.b.data ** 2)


class Power(Operation):
    def __repr__(self): return "**"

    def __call__(self, a, b):
        self.a = a
        self.b = b
        return a.data ** b.data
    
    def partial_a(self):
        return self.b.data * (self.a.data ** (self.b.data-1))
    
    def partial_b(self):
        return (self.a.data ** self.b.data) * math.log(self.a.data)

"xxx"