# andas库一些基础功能

安装pandas

```python
pip install pandas
import pandas as pd

df = pd.read_csv('data.csv')
#读取csv
df = pd.read_excel('data.xlsx')
#读取excle

df = pd.DataFrame('name（string）', columns=['A', 'B'])
#创建一个 DataFrame 对象。DataFrame 是 Pandas 中用于数据处理和分析的主要数据结构，类似于 Excel 表格或 SQL 表，它由行和列组成。
df.to_csv('name.csv', index=False)
#将对象保存为csv
```

