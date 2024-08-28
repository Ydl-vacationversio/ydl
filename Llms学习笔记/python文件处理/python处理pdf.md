# 处理pdf（PyMuPDF）

[出处](https://blog.csdn.net/qq_41185868/article/details/134587863?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522172476362816800182131991%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=172476362816800182131991&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-134587863-null-null.142^v100^control&utm_term=PyMuPDF&spm=1018.2226.3001.4187)

首先安装PyMuPDF

```python
pip install PyMuPDF
```

基础操作：

```python
import fitz
#导入
    
doc = fitz.open(filename)
#打开文件，filename可以是xxxx.pdf等 
pangenumber=doc.page_count
#获取pdf页数

for i in tqdm(doc): 
    text = i.get_text()
        #循环遍历pdf，得到每一页的信息

document.save("output.pdf")
# 保存PDF
document.close()
# 关闭文档

page.set_rotation(90)  # 旋转90度
# 旋转页面

clip = fitz.Rect(0, 100, 500, 600)  # 定义裁剪区域
page = document[0]
page.set_cropbox(clip)
# 裁剪页面

pdf1 = fitz.open("file1.pdf")
pdf2 = fitz.open("file2.pdf")
pdf1.insert_pdf(pdf2)
pdf1.save("merged_output.pdf")
# 合并PDF文档
    
# 提取页面中的图片
for page in document:
    image_list = page.get_images(full=True)
for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = document.extract_image(xref)
        image_bytes = base_image["image"]
# 将图片保存到文件
with open(f"image{page.number+1}_{img_index}.png", "wb") as img_file:
            img_file.write(image_bytes)

```

