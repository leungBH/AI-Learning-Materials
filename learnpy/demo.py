"""
#获取字符的整数表示。
print(ord('A'))
print(ord('B'))
print(ord('中'))
print(ord('文'))

#小明的成绩从去年的72分提升到了今年的85分，请计算小明成绩提升的百分点，并用字符串格式化显示出'xx.x%'，只保留小数点后1位：
r = (85-72)/72*100
print('%.1f%%' % r)


#list
classmates = ['A','B','C']
print(len(classmates))
print(classmates[0])
print(classmates[1])
print(classmates[2])

#tuple
classmates = ("A","B","C")
print(len(classmates))
print(classmates[0])
print(classmates[1])
print(classmates[2])
"""

classmates = ("A","B","C")
for name in classmates:
    print(name)
    
#dict

#set    
    