# text-tagging-ml
multi-label text classification with Tensorflow


## Usage

> Command Line

```python main.py --checkpoint "./ml_model/" --inference true --i_title "안녕하세요" --i_content "올ㅋ"```

> Results

```
[*] Model Size : 31.82 M params
[+] charcnn model loaded
[*] Reading checkpoints...
[+] global step : 520  successfully loaded
[*] Prediction
  [*] title        : 안녕하세요
  [*] content      : 올ㅋ
  [*] big category : current-affairs
  [*] sub category : current-affairs
```