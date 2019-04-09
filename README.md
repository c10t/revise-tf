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
