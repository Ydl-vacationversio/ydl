# langchain调用大模型

```python
import getpass
import os

#apikey、模型型号、接口网址需要修改
os.environ["OPENAI_API_KEY"] = 'apikey'
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o",base_url='接口网址')
```

使用langchain：[langchain](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/#using-language-models)

```python
from langchain_core.prompts import ChatPromptTemplate
system_template = "你是一个精通计算机的专家，请根据专业知识回答我下面的问题"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
#text是你要参考的文本，可以在其他地方定义

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
#配置输出

chain = prompt_template | model | parser
#将提示模板、模型、输出 chain到一起

answer = chain.invoke({"text": text})
#在已经设定好提示词的情况下，直接改变输入的文本问题，调用chain.invoke实现对大模型的提问。大模型的输出保存在answer里

```