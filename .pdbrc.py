import pdb
class Config(pdb.DefaultConfig):
    sticky_by_default = True
    editor = "vim"
    # pygments_formatter_class = "pygments.formatters.TerminalTrueColorFormatter"
    # pygments_formatter_kwargs = {"style": "solarized-light"}

    def __init__(self):
        try:
            from pygments.formatters import terminal

        except ImportError:
            pass
        else:
            self.colorscheme = terminal.TERMINAL_COLORS.copy()
            self.colorscheme.update({
                terminal.Keyword:            ('darkred',     'red'),
                terminal.Number:             ('darkyellow',  'yellow'),
                terminal.String:             ('brown',       'green'),
                terminal.Name.Function:      ('darkgreen',   'blue'),
                terminal.Name.Namespace:     ('teal',        'white'),
                })
