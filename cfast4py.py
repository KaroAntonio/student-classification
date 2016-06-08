import sys
import ast

# TODO: does not work on comprehensions
# TODO: number of params for functions
# TODO: add break, continue

def toCFAST(text, name='', delim='\n'):
    '''Produce a string representation of a CFAST.
    '''
    myast = ast.parse(text, name)
    v=MyVisitor(myast, delim=delim)
    return str(v)


class MyVisitor(ast.NodeVisitor):
    '''Visitor subclass of NodeVisitor that produces a CFAST in the
    instance variable cfast. Can set a delimiter (delim) so that the
    CFAST can be one line, or many lines.
    '''

    tabstop = 4

    def __init__(self, ast, delim='\n', debug=False):
        self.ast = ast
        self.DEBUG=debug
        self.delim=delim

        self.cfast=[]
        #self.depth = 0
        self.depth = 4
        super(MyVisitor, self).__init__()

    def generic_visit(self, node):
        '''Recursively visit each node and add to the cfast variable.

        TODO: should we include a placeholder when there is a statement vs. no statements inside a block?
        '''
        name=type(node).__name__
        if self.DEBUG:
            print('debug; visiting:',name)
        if name == 'Module':
            ast.NodeVisitor.generic_visit(self, node)
        elif name in ['FunctionDef', 'For', 'While']:
            self.cfast.append(' ' * self.depth + name)
            self.depth += self.tabstop
            ast.NodeVisitor.generic_visit(self, node)
            self.depth -= self.tabstop
        elif name == 'If':
            self.cfast.append(' ' * self.depth + name)
            self.depth += self.tabstop
            for n in node.body:
                self.visit(n)
            self.depth -= self.tabstop

            if node.orelse:     # Handle Else
                self.cfast.append(' ' * self.depth + 'Else')
                self.depth += self.tabstop
                for n in node.orelse:
                    self.visit(n)
                self.depth -= self.tabstop
        elif name == 'Return':
            self.cfast.append(' ' * self.depth + name)
            ast.NodeVisitor.generic_visit(self, node)
        else:
            #print('CATCHALL', name)
            ast.NodeVisitor.generic_visit(self, node)

    def __str__(self):
        if not self.cfast:
            self.visit(self.ast)
        return "FileAST\n" + self.delim.join(self.cfast)


def main():
    filename='test1.py'
    if len(sys.argv)>1:
        filename=sys.argv[1]
    text=open(filename).read()
    cfast=toCFAST(readtext(filename), filename)

    print(cfast)


if __name__=='__main__':
    main()
