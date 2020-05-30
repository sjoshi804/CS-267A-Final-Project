# pylint: disable=too-few-public-methods,too-many-public-methods


class Node:
    def accept(self, visitor):
        visitor.methods[type(self).__name__](self)


class Program(Node):
    def __init__(self, functions, main_expr):
        self.functions = functions
        self.main_expr = main_expr


class BracketedExpr(Node):
    def __init__(self, expr):
        self.expr = expr


class TrueLiteral(Node):
    pass


class FalseLiteral(Node):
    pass


class IntLiteral(Node):
    def __init__(self, size, value):
        self.size = size
        self.value = value


class Discrete(Node):
    def __init__(self, probabilities):
        self.probabilities = probabilities


class Binop(Node):
    def __init__(self, lhs, rhs, binop):
        self.lhs = lhs
        self.rhs = rhs
        self.binop = binop


class Pair(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Fst(Node):
    def __init__(self, expr):
        self.expr = expr


class Snd(Node):
    def __init__(self, expr):
        self.expr = expr


class Not(Node):
    def __init__(self, expr):
        self.expr = expr


class Flip(Node):
    def __init__(self, probability):
        self.probability = probability


class Observe(Node):
    def __init__(self, expr):
        self.expr = expr


class If(Node):
    def __init__(self, guard_expr, then_expr, else_expr):
        self.guard_expr = guard_expr
        self.then_expr = then_expr
        self.else_expr = else_expr


class Let(Node):
    def __init__(self, identifier, equal_expr, in_expr):
        self.identifier = identifier
        self.equal_expr = equal_expr
        self.in_expr = in_expr


class Call(Node):
    def __init__(self, function, args):
        self.function = function
        self.args = args


class Identifier(Node):
    def __init__(self, name):
        self.name = name


class BoolType(Node):
    pass


class PairType(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class IntType(Node):
    def __init__(self, size):
        self.size = size


class Arg(Node):
    def __init__(self, identifier, typ):
        self.identifier = identifier
        self.typ = typ


class Function(Node):
    def __init__(self, name, args, expr):
        self.name = name
        self.args = args
        self.expr = expr


class PrintVisitor:
    def __init__(self):
        self.text = ""
        self.methods = {}
        self.methods["Program"] = self.visit_program
        self.methods["BracketedExpr"] = self.visit_bracketed_expr
        self.methods["TrueLiteral"] = self.visit_true_literal
        self.methods["FalseLiteral"] = self.visit_false_literal
        self.methods["IntLiteral"] = self.visit_int_literal
        self.methods["Discrete"] = self.visit_discrete
        self.methods["Binop"] = self.visit_binop
        self.methods["Pair"] = self.visit_pair
        self.methods["Fst"] = self.visit_fst
        self.methods["Snd"] = self.visit_snd
        self.methods["Not"] = self.visit_not
        self.methods["Flip"] = self.visit_flip
        self.methods["Observe"] = self.visit_observe
        self.methods["If"] = self.visit_if
        self.methods["Let"] = self.visit_let
        self.methods["Call"] = self.visit_call
        self.methods["Identifier"] = self.visit_identifier
        self.methods["BoolType"] = self.visit_bool_type
        self.methods["PairType"] = self.visit_pair_type
        self.methods["IntType"] = self.visit_int_type
        self.methods["Arg"] = self.visit_arg
        self.methods["Function"] = self.visit_function

    def visit_program(self, node):
        for f in node.functions:
            f.accept(self)
            self.text += "\n\n"
        node.main_expr.accept(self)

    def visit_bracketed_expr(self, node):
        self.text += "( "
        node.expr.accept(self)
        self.text += " )"

    def visit_true_literal(self, _):
        self.text += "true"

    def visit_false_literal(self, _):
        self.text += "false"

    def visit_int_literal(self, node):
        self.text += "int(" + str(node.size) + ", " + str(node.value) + ")"

    def visit_discrete(self, node):
        self.text += "discrete("
        for p in node.probabilities[:-1]:
            self.text += "{:.4f}".format(p)
            self.text += ", "
        self.text += "{:.4f}".format(node.probabilities[-1])
        self.text += ")"

    def visit_binop(self, node):
        node.lhs.accept(self)
        self.text += " " + node.binop + " "
        node.rhs.accept(self)

    def visit_pair(self, node):
        self.text += "( "
        node.left.accept(self)
        self.text += ", "
        node.right.accept(self)
        self.text += " )"

    def visit_fst(self, node):
        self.text += "fst "
        node.expr.accept(self)

    def visit_snd(self, node):
        self.text += "snd "
        node.expr.accept(self)

    def visit_not(self, node):
        self.text += "!"
        node.expr.accept(self)

    def visit_flip(self, node):
        self.text += "flip " + "{:.7f}".format(node.probability)

    def visit_observe(self, node):
        self.text += "observe "
        node.expr.accept(self)

    def visit_if(self, node):
        self.text += "if "
        node.guard_expr.accept(self)
        self.text += " then "
        node.then_expr.accept(self)
        self.text += " else "
        node.else_expr.accept(self)

    def visit_let(self, node):
        self.text += "let "
        node.identifier.accept(self)
        self.text += " = "
        node.equal_expr.accept(self)
        self.text += " in "
        node.in_expr.accept(self)

    def visit_call(self, node):
        self.text += node.function.name
        self.text += "("
        for a in node.args[:-1]:
            a.accept(self)
            self.text += ", "
        node.args[-1].accept(self)
        self.text += ")"

    def visit_identifier(self, node):
        self.text += node.name

    def visit_bool_type(self, _):
        self.text += "bool"

    def visit_pair_type(self, node):
        self.text += "( "
        node.left.accept(self)
        self.text += ", "
        node.right.accept(self)
        self.text += " )"

    def visit_int_type(self, node):
        self.text += "int(" + str(node.size) + ")"

    def visit_arg(self, node):
        node.identifier.accept(self)
        self.text += ": "
        node.typ.accept(self)

    def visit_function(self, node):
        self.text += "fun "
        self.text += node.name
        self.text += "("
        for a in node.args[:-1]:
            a.accept(self)
            self.text += ", "
        node.args[-1].accept(self)
        self.text += ") {"
        node.expr.accept(self)
        self.text += "}"


def bandit():
    functions = []
    action = Identifier("ACTION")
    optimal_now = Identifier("OPTIMAL_NOW")
    main_expr = Let(
        action, Discrete([0.25, 0.25, 0.25, 0.25]),
        Let(
            optimal_now,
            If(Binop(action, IntLiteral(4, 3), "=="), Flip(1.0),
               Flip(0.000045)),
            Let(Identifier("_"), Observe(optimal_now), action)))
    p = Program(functions, main_expr)
    v = PrintVisitor()
    p.accept(v)
    print(v.text)


def states():
    state = Identifier("STATE")
    action = Identifier("ACTION")
    optimal_now = Identifier("OPTIMAL_NOW")
    next_state = Identifier("NEXT_STATE")
    next_action = Identifier("NEXT_ACTION")
    optimal_next = Identifier("OPTIMAL_NEXT")

    transition = Function(
        "transition", [Arg(state, IntType(8)),
                       Arg(action, IntType(4))],
        If(
            Binop(Binop(state, IntLiteral(8, 0), "=="),
                  Binop(action, IntLiteral(4, 3), "=="), "&&"),
            Discrete([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.50]),
            If(
                Binop(Binop(state, IntLiteral(8, 0), "=="),
                      Binop(action, IntLiteral(4, 1), "=="), "&&"),
                Discrete([0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25]),
                Discrete([0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]))))

    main_expr = Let(
        state, IntLiteral(8, 0),
        Let(
            action, Discrete([0.25, 0.25, 0.25, 0.25]),
            Let(
                optimal_now, Flip(0.000045),
                Let(
                    next_state, Call(transition, [state, action]),
                    Let(
                        next_action, Discrete([0.25, 0.25, 0.25, 0.25]),
                        Let(
                            optimal_next,
                            If(
                                Binop(
                                    Binop(next_state, IntLiteral(8, 7), "=="),
                                    Binop(next_action, IntLiteral(4, 3), "=="),
                                    "&&"), Flip(1.0), Flip(0.000045)),
                            Let(
                                Identifier("_"),
                                Observe(
                                    BracketedExpr(
                                        Binop(optimal_now, optimal_next,
                                              "&&"))), action)))))))

    p = Program([transition], main_expr)
    v = PrintVisitor()
    p.accept(v)
    print(v.text)


if __name__ == "__main__":
    # bandit()
    states()
