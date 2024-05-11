import inspect
from typing import Dict, Optional, Any

import pixeltable
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from .function import Function
from .signature import Signature, Parameter


class QueryTemplateFunction(Function):
    """A parameterized query/DataFrame from which an executable DataFrame is created with a function call."""

    def __init__(
            self, df: 'pixeltable.DataFrame', py_signature: inspect.Signature, self_path: Optional[str] = None,
            name: Optional[str] = None):
        self.df = df
        self.self_name = name
        self.param_types = df.parameters()

        # verify default values
        import pixeltable.exprs as exprs
        self.defaults: Dict[str, exprs.Literal] = {}  # key: param name, value: default value converted to a Literal
        for py_param in py_signature.parameters.values():
            if py_param.default is inspect.Parameter.empty:
                continue
            assert py_param.name in self.param_types
            param_type = self.param_types[py_param.name]
            try:
                literal_default = exprs.Literal(py_param.default, col_type=param_type)
                self.defaults[py_param.name] = literal_default
            except TypeError as e:
                msg = str(e)
                raise excs.Error(f"Default value for parameter '{py_param.name}': {msg[0].lower() + msg[1:]}")

        # construct signature
        assert len(self.param_types) == len(py_signature.parameters)
        fn_params = [
            Parameter(py_param.name, self.param_types[py_param.name], py_param.kind)
            for py_param in py_signature.parameters.values()
        ]
        signature = Signature(return_type=ts.JsonType(), parameters=fn_params)

        super().__init__(signature, py_signature=py_signature, self_path=self_path)

    def exec(self, *args: Any, **kwargs: Any) -> Any:
        bound_args = self.py_signature.bind(*args, **kwargs).arguments
        # apply defaults, otherwise we might have Parameters left over
        bound_args.update(
            {param_name: default for param_name, default in self.defaults.items() if param_name not in bound_args})
        bound_df = self.df.bind(bound_args)
        result = bound_df.collect()
        return list(result)

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def _as_dict(self) -> Dict:
        if self.self_path is not None:
            return super()._as_dict()
        return {
            'name': self.name,
            'expr': self.expr.as_dict(),
            **super()._as_dict()
        }

    @classmethod
    def _from_dict(cls, d: Dict) -> Function:
        if 'expr' not in d:
            return super()._from_dict(d)
        import pixeltable.exprs as exprs
        return cls(exprs.Expr.from_dict(d['expr']), name=d['name'])
