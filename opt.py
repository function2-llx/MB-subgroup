from typing import List

import click

class TypedChoice(click.Choice):
    def __init__(self, choices, type: click.ParamType):
        from click.types import convert_type
        super().__init__(choices)
        self.type = type

    def convert(self, value, param, ctx):
        # Match through normalization and case sensitivity
        # first do token_normalize_func, then lowercase
        # preserve original `value` to produce an accurate message in
        # `self.fail`
        normed_value = value
        normed_choices = {choice: choice for choice in self.choices}

        if ctx is not None and ctx.token_normalize_func is not None:
            normed_value = ctx.token_normalize_func(value)
            normed_choices = {
                ctx.token_normalize_func(normed_choice): original
                for normed_choice, original in normed_choices.items()
            }

        if not self.case_sensitive:
            from click.types import PY2
            if PY2:
                lower = str.lower
            else:
                lower = str.casefold

            normed_value = lower(normed_value)
            normed_choices = {
                lower(normed_choice): original
                for normed_choice, original in normed_choices.items()
            }

        if normed_value in normed_choices:
            return normed_choices[normed_value]

        self.fail(
            "invalid choice: {}. (choose from {})".format(
                value, ", ".join(self.choices)
            ),
            param,
            ctx,
        )
