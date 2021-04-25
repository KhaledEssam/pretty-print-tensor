import inspect
from inspect import getframeinfo, stack
from typing import Dict, List, Optional, Union

from prettytable import PrettyTable
from torch import nn


def debuginfo():
    caller = getframeinfo(stack()[2][0])
    return f"{caller.filename}:{caller.lineno}"


# Sieve of Eratosthenes
# Code by David Eppstein, UC Irvine, 28 Feb 2002
# http://code.activestate.com/recipes/117119/


def gen_primes():
    """Generate an infinite sequence of prime numbers."""
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}

    # The running integer that's checked for primeness
    q = 2

    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            #
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next
            # multiples of its witnesses to prepare for larger
            # numbers
            #
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]

        q += 1


def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [
            var_name
            for var_name, var_val in fi.frame.f_locals.items()
            if var_val is var
        ]
        if len(names) > 0:
            return names[0]


class PPT(object):
    def __init__(self) -> None:
        self.n2d: Dict[str, int] = {}
        self.d2n: Dict[str, int] = {}

    def defvars(
        self,
        variables: Union[str, List],
        values: Optional[Dict[str, int]] = None,
        sep: str = ",",
    ) -> Dict[str, int]:
        if isinstance(variables, str):
            variables = variables.strip().split(sep)
        if values is None:
            values = {}
        primes = gen_primes()
        names = [i.strip() for i in variables]
        dims = [next(primes) if name not in values else values[name] for name in names]
        self.n2d = {n: d for n, d in zip(names, dims)}
        self.d2n = {d: n for n, d in zip(names, dims)}
        print("Defined Variables:")
        vs = PrettyTable(field_names=["Variable Name", "Value"])
        vs.add_rows([[k, v] for k, v in self.n2d.items()])
        print(vs)
        return self.n2d

    def __call__(self, *args):
        def type_str(t) -> str:
            return t.type().replace("torch.", "")

        if isinstance(args[0], nn.Module):
            for i in args:
                i.pp = self
            return args if len(args) > 1 else args[0]

        table = PrettyTable(field_names=["Caller", "Variable Name", "Type", "Shape"])
        caller = debuginfo()
        for t in args:
            name = retrieve_name(t)
            dims = t.size()
            dim_names = []
            for d in dims:
                if d in self.d2n:
                    dim_names.append(self.d2n[d])
                else:
                    mul = 2
                    should_break = False
                    while not should_break:
                        for _k, _v in self.d2n.items():
                            if _k * mul == d:
                                dim_names.append(f"{mul} × {_v}")
                                should_break = True
                                break
                        mul += 1

            table.add_row([caller, name, type_str(t), f"[{', '.join(dim_names)}]"])
        print(table)
