# revise-tf

## Jupyter Notebook

### CSS settings

```python
from IPython.display import display, HTML

css = """
  .CodeMirror pre, .output pre {
    font-family: 'Roboto Mono', Monaco, 源真ゴシック等幅 monospace;
  }
"""

display(HTML('<link href="https://fonts.googleapis.com/css?family=Roboto+Mono" rel="stylesheet">'))
display(HTML('<style type="text/css">%s</style>' % css))
```

### Run python script

`%run -i '01_start.py'`

## venv/pipfile note
`$ pipenv install [package | (or initialize)]`
`$ pipenv shell # activate`
`$ exit # deactivate`

`$ time pipenv install -r requirements.txt --skip-lock`
https://github.com/pypa/pipenv/issues/1914

https://github.com/pypa/pipenv/issues/84

https://tekunabe.hatenablog.jp/entry/2018/12/28/vscode_venv_default_rolad
